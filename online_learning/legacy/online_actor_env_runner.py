#!/home/vision/miniconda3/envs/robodiff/bin/python
"""
온라인 학습 Actor — 박스삽입(오른팔, wrench) 태스크 + 페달 servo 교정.

bae_eval_real_robot_rightarm_insert_plug.py 의 검증된 제어 루프(박스삽입 env
DualarmRealEnv, wrench obs)를 따르고, 온라인 훅 + 페달 servo 핸드오프를 추가한다.

  (1) 매 에피소드 시작 시 weights mailbox를 확인해 새 버전이 있으면 policy 가중치를
      hot-swap (learner가 fine-tune한 결과를 실시간 반영).
  (2) 유지(keep)한 에피소드를 achieved-pose로 relabel(오른팔 wrench 포함)해서 단일 demo
      HDF5로 만들어 learner에 전송 (online_learning/relabel_utils.relabel_..._box).
  (3) 페달로 servo(servo_rightarm_imp_online.py)에 로봇을 넘겨 사람이 직접 교정.
      교정 구간은 stage=1로 기록되어 correction 데이터가 된다.

페달/키 조작 (컨트롤 창을 클릭해 포커스를 준 뒤):
  a : 핸드오프 — actor 제어 정지(로봇 명령 발행 중단) + /teleop_control=1 (servo가 현재
      팔 위치를 기준으로 VR delta servoing). 이 구간 obs가 stage=1(correction)로 기록됨.
  b : 복귀    — /teleop_control=0 (servo hold) + actor 제어 재개(현재 포즈로 재동기화).
  c : 홈복귀  — /teleop_control=2 (servo가 preset home으로 이동). 계속 teleop 상태.
  S : 에피소드 종료 + 유지(relabel 후 learner로 전송)
  D : 에피소드 종료 + 폐기
  Q/Esc/Ctrl+C : 종료
  (타임아웃(max_duration)으로 끝난 에피소드도 유지·전송된다.)

⚠️ 함께 실행: 다른 터미널에서 servo_rightarm_imp_online.py 를 띄워둬야 페달 servo가 동작.
   그리고 두 노드가 같은 /right_dsr_controller/task_space_command 로 발행하므로, 핸드오프
   시 actor 컨트롤러가 발행을 멈춰야(pause) 안 싸운다 — 이 스크립트가 자동으로 처리.

실행:
  source ~/dualarm_ws/install/setup.bash
  # 터미널1: python online_learning/servo_rightarm_imp_online.py
  # 터미널2: python online_learning/online_learner.py -i <box_ckpt>
  # 터미널3:
  python online_learning/online_actor_env_runner.py \
      -i data/outputs/260708_insert_box_wrench_abs/epoch=0900-train_loss=0.000.ckpt \
      --steps_per_inference 12 --frequency 10 --num_inference_steps 12
"""
# ============================================================
# huggingface_hub 버전 충돌 회피 (다른 import보다 먼저)
# ============================================================
import sys
local_paths = [p for p in sys.path if '.local' in p]
sys.path = [p for p in sys.path if '.local' not in p]
import huggingface_hub  # noqa: F401
sys.path = local_paths + sys.path

import os
import time
import glob
import select
import termios
import tty

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import dill
import hydra
import click
import cv2
import av
from multiprocessing.managers import SharedMemoryManager
from omegaconf import OmegaConf

from diffusion_policy.real_world.bae_real_env_rightarm_hand_insert_plug import DualarmRealEnv
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, get_real_obs_dict, get_real_relative_obs_dict,
    get_abs_action_from_relative)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace

from online_learning.legacy import config_online as C
from online_learning.legacy.mailbox import FileMailbox
from online_learning.legacy.relabel_utils import relabel_last_episode_to_hdf5_box, pose_quat_to_9d

OmegaConf.register_new_resolver("eval", eval, replace=True)

# 오른팔 realsense (박스삽입 eval과 동일)
RIGHT_CAMERA_SERIALS = ['126122270712']
WIN = "actor_control"


class TeleopControlPub:
    """servo_rightarm_imp_online.py 로 /teleop_control(UInt8)을 쏘는 소형 rclpy 노드.
    env 자식 프로세스들이 모두 fork된 뒤(=env warmup 후) 생성해야 안전하다."""
    def __init__(self):
        import rclpy
        from std_msgs.msg import UInt8
        self._rclpy = rclpy
        self._UInt8 = UInt8
        self._own_init = False
        if not rclpy.ok():
            rclpy.init(args=None)
            self._own_init = True
        self.node = rclpy.create_node('actor_teleop_control_pub')
        self.pub = self.node.create_publisher(UInt8, '/teleop_control', 10)
        print("[Actor] /teleop_control publisher 준비 완료")

    def publish(self, value):
        msg = self._UInt8()
        msg.data = int(value)
        # 확실히 전달되도록 몇 번 쏘고 스핀
        for _ in range(3):
            self.pub.publish(msg)
            self._rclpy.spin_once(self.node, timeout_sec=0.01)
        print(f"[Actor] /teleop_control <- {value}")

    def close(self):
        try:
            self.node.destroy_node()
        except Exception:
            pass
        if self._own_init:
            try:
                self._rclpy.shutdown()
            except Exception:
                pass


def maybe_hotswap_weights(mailbox, policy, current_version, device):
    """새 가중치 버전이 있으면 policy에 로드. 새 버전 번호 반환."""
    latest = mailbox.get_latest_weight_version()
    if latest is None or latest == current_version:
        return current_version
    payload = mailbox.load_weights(latest, map_location=device)
    if payload is None:
        return current_version
    missing, unexpected = policy.load_state_dict(payload["state_dict"], strict=False)
    policy.eval().to(device)
    print(f"[Actor] 가중치 hot-swap: v{current_version} -> v{latest} "
          f"(learner demo={payload.get('num_demos', '?')}, "
          f"missing={len(missing)}, unexpected={len(unexpected)})")
    return latest


def wait_videos_finalized(video_dir, timeout=30.0, poll=0.5):
    """end_episode 후 mp4 인코더가 moov atom을 다 쓸 때까지 대기."""
    mp4s = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if not mp4s:
        print(f"[Actor][WARN] 영상 파일 없음: {video_dir}")
        return
    deadline = time.monotonic() + timeout
    for path in mp4s:
        while True:
            try:
                with av.open(path) as container:
                    next(container.decode(video=0))
                break
            except Exception:
                if time.monotonic() > deadline:
                    print(f"[Actor][WARN] 영상 마무리 대기 타임아웃(>{timeout}s): {path}")
                    break
                time.sleep(poll)
    print(f"[Actor] 영상 마무리 확인 완료: {video_dir}")


class KeyReader:
    """터미널을 cbreak 모드로 두고 논블로킹으로 한 글자씩 읽는다.
    cv2 GUI(QT5)는 멀티프로세싱 fork와 교착되므로 창 대신 stdin으로 페달/키를 받는다.
    ★ 페달/키보드가 이 '터미널'에 포커스된 상태에서 입력해야 한다."""
    def __init__(self):
        self.fd = None
        self.old = None
        self.enabled = False

    def __enter__(self):
        try:
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)          # 한 글자씩, echo 없음, Ctrl+C(ISIG)는 유지
            self.enabled = True
        except Exception as e:
            print(f"[Actor][WARN] 터미널 raw 모드 실패({e}) — 키 입력이 안 될 수 있음.")
        return self

    def poll(self):
        """입력이 있으면 한 글자(str) 반환, 없으면 None."""
        if not self.enabled:
            return None
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            return sys.stdin.read(1)
        return None

    def __exit__(self, *a):
        if self.old is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
            except Exception:
                pass


@click.command()
@click.option('--input', '-i', default=None,
              help='박스삽입 체크포인트(.ckpt). 미지정 시 config_online.BASE_CKPT.')
@click.option('--output', '-o', default=None,
              help='에피소드 저장 폴더. 미지정 시 config_online.ACTOR_OUTPUT_DIR.')
@click.option('--robot_ip', '-ri', default="192.168.111.50", help="Robot IP (호환용).")
@click.option('--steps_per_inference', '-si', default=12, type=int,
              help="한 번의 inference에서 실행할 action 스텝 수.")
@click.option('--max_duration', '-md', default=60, type=float,
              help='에피소드 최대 길이(초). 초과 시 자동 종료(유지·전송).')
@click.option('--frequency', '-f', default=10.0, type=float, help="제어 주기(Hz).")
@click.option('--num_inference_steps', '-ni', default=12, type=int, help="DDIM denoising 스텝 수.")
def main(input, output, robot_ip, steps_per_inference,
         max_duration, frequency, num_inference_steps):
    ckpt_path = input if input is not None else C.BASE_CKPT
    output_dir = output if output is not None else C.ACTOR_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    mailbox = FileMailbox(C.ONLINE_WORKDIR)
    device = torch.device(C.DEVICE if torch.cuda.is_available() else "cpu")

    # ── base policy 로드 ──
    print(f"[Actor] base 체크포인트 로드: {ckpt_path}")
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    # 박스삽입 ckpt는 cfg.name='box_insert_virtual_action'(‘diffusion’ 미포함)이라
    # 이름만 보면 안 되고 workspace/policy _target_ 도 함께 확인한다 (bae eval과 동일).
    _wt = str(OmegaConf.select(cfg, "_target_", default="")).lower()
    _pt = str(OmegaConf.select(cfg, "policy._target_", default="")).lower()
    if not any('diffusion' in t for t in (str(cfg.name).lower(), _wt, _pt)):
        raise RuntimeError(f"Unsupported policy type: {cfg.name} / {cfg._target_}")
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.eval().to(device)
    policy.num_inference_steps = num_inference_steps
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    obs_pose_repr = OmegaConf.select(cfg, "task.pose_repr.obs_pose_repr", default=None)
    action_pose_repr = OmegaConf.select(cfg, "task.pose_repr.action_pose_repr", default=None)
    print(f"[Actor] pose_repr: obs={obs_pose_repr}, action={action_pose_repr}")

    weight_version = -1
    dt = 1.0 / frequency
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)   # (width, height)
    n_obs_steps = cfg.n_obs_steps
    action_dim = cfg.task.shape_meta.action.shape[0]

    def make_obs_dict(obs):
        if obs_pose_repr == 'relative':
            d = get_real_relative_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta)
        else:
            d = get_real_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta)
        return dict_apply(d, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))

    with SharedMemoryManager() as shm_manager:
        with DualarmRealEnv(
            output_dir=output_dir,
            robot_ip=robot_ip,
            frequency=frequency,
            camera_serial_numbers=RIGHT_CAMERA_SERIALS,
            n_obs_steps=n_obs_steps,
            shape_meta=cfg.task.shape_meta,
            obs_image_resolution=obs_res,
            obs_float32=True,
            init_joints=False,
            # ★ multi_cam_vis(별도 프로세스 cv2 창)를 켜면, 이 메인 프로세스가 만드는
            #   컨트롤 cv2.namedWindow 와 OpenCV GUI 백엔드가 교착(deadlock)되어
            #   namedWindow가 리턴하지 않는다. actor는 자체 컨트롤 창으로 페달/키를
            #   받아야 하므로 vis는 끈다(카메라는 계속 녹화됨, 라이브 프리뷰만 없음).
            enable_multi_cam_vis=False,
            record_raw_video=False,
            thread_per_video=3,
            video_crf=21,
            shm_manager=shm_manager,
        ) as env:
            # 참고: 이 rush 버전 DualarmRealEnv는 record_wrist_wrench 파라미터가 없다.
            #   obs의 wrench_wrist_R(6,32) 기록은 shape_meta(wrench 키)로 결정되며
            #   replay_buffer에 그대로 저장되어 relabel에 쓰인다.
            print("[Actor] env READY (realsense+robot 초기화 완료).")
            cv2.setNumThreads(1)   # cv2 GUI(QT5)는 fork와 교착 → 창은 안 만들고 stdin으로 입력
            print("[Actor] 센서 워밍업...")
            time.sleep(2.0)

            # ── warmup inference ──
            print("[Actor] 워밍업 추론...")
            obs = env.get_obs()
            with torch.no_grad():
                policy.reset()
                result = policy.predict_action(make_obs_dict(obs))
                a = result['action'][0].detach().cpu().numpy()
                assert a.shape[-1] == action_dim, \
                    f"Expected action dim={action_dim}, got {a.shape[-1]}"
                del result

            # env 자식들이 모두 뜬 뒤 teleop publisher 생성 (fork-after-init 회피)
            teleop = TeleopControlPub()

            print("[Actor] 준비 완료 (이 '터미널'에 포커스 두고 페달/키 입력).")
            print("        a=servo핸드오프  b=actor복귀  c=홈  |  s=유지+전송  d=폐기  q=종료")

            mode = 'policy'   # 'policy' | 'teleop'  (process_key가 nonlocal로 갱신)

            def process_key(key):
                """a/b/c 핸드오프는 즉시 처리(로봇 pause/resume + servo publish).
                s/d/q는 신호('keep'/'discard'/'quit')를 반환. 없으면 None."""
                nonlocal mode
                if key is None:
                    return None
                if key in ('a', 'A'):
                    if mode != 'teleop':
                        env.pause_robot(); teleop.publish(1); mode = 'teleop'
                        print("[Actor] 핸드오프 → servo (correction 기록 시작)")
                elif key in ('b', 'B'):
                    if mode != 'policy':
                        teleop.publish(4); env.resume_robot(); mode = 'policy'
                        print("[Actor] 복귀 → actor 정책 제어")
                elif key in ('c', 'C'):
                    env.pause_robot(); teleop.publish(2); mode = 'teleop'
                    print("[Actor] 홈복귀(servo). 복귀하려면 b.")
                elif key in ('s', 'S'):
                    return 'keep'
                elif key in ('d', 'D'):
                    return 'discard'
                elif key in ('q', 'Q', '\x1b'):
                    return 'quit'
                return None

            kr = KeyReader()
            kr.__enter__()
            try:
                while True:
                    # ── 에피소드 시작 전: 새 가중치 확인 & hot-swap ──
                    weight_version = maybe_hotswap_weights(mailbox, policy, weight_version, device)

                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    precise_wait(eval_t_start - 1.0 / 30, time_func=time.time)
                    print("[Actor] 에피소드 시작!")

                    iter_idx = 0
                    mode = 'policy'       # process_key가 nonlocal로 갱신
                    keep = True
                    quit_all = False
                    frame_latency = 1.0 / 30
                    episode_done = None   # None | 'keep' | 'discard' | 'quit'
                    while True:
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']

                        # 루프 진입 시 키 반영 (a/b/c 핸드오프는 process_key가 즉시 처리)
                        sig = process_key(kr.poll())
                        if sig is not None:
                            episode_done = sig; break

                        # ── 제어 ──
                        if mode == 'policy':
                            with torch.no_grad():
                                result = policy.predict_action(make_obs_dict(obs))
                                action = result['action'][0].detach().cpu().numpy()  # [H,9]

                            this_target_poses = np.zeros((len(action), action.shape[-1]),
                                                         dtype=np.float64)
                            this_target_poses[:] = action[:]
                            if action_pose_repr == 'relative':
                                if np.any(np.abs(this_target_poses[..., :3]) > 0.5):
                                    print("[Actor][WARN] 비정상적으로 큰 relative translation!")
                                this_target_poses = get_abs_action_from_relative(
                                    action=this_target_poses, env_obs=obs)

                            action_timestamps = (
                                np.arange(len(action), dtype=np.float64)) * dt + obs_timestamps[-1]
                            curr_time = time.time()
                            is_new = action_timestamps > (curr_time + 0.01)
                            if np.sum(is_new) == 0:
                                this_target_poses = this_target_poses[[-1]]
                                next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                                action_timestamps = np.array([eval_t_start + next_step_idx * dt])
                            else:
                                this_target_poses = this_target_poses[is_new][:steps_per_inference]
                                action_timestamps = action_timestamps[is_new][:steps_per_inference]

                            env.exec_actions(actions=this_target_poses,
                                             timestamps=action_timestamps,
                                             stages=np.zeros(len(this_target_poses), dtype=np.int64))
                            cycle_steps = steps_per_inference
                        else:
                            # teleop: servo가 로봇을 몰고, actor는 achieved pose를
                            # correction(stage=1)으로 기록만 한다 (로봇에 명령 X).
                            cur_pose = np.asarray(obs['robot_pose_R'])[-1]
                            cur_quat = np.asarray(obs['robot_quat_R'])[-1]
                            act9 = pose_quat_to_9d(cur_pose[None], cur_quat[None]).astype(np.float64)
                            env.exec_actions(actions=act9,
                                             timestamps=np.array([obs_timestamps[-1] + 2 * dt]),
                                             stages=np.array([1], dtype=np.int64),
                                             record_only=True)
                            cycle_steps = 1

                        # timeout
                        if time.monotonic() - t_start > max_duration:
                            episode_done = 'keep'
                            print("[Actor] 타임아웃 종료 (유지)"); break

                        # ── 키 반응형 대기 (핸드오프 지연 최소화: ~5ms 주기 폴링) ──
                        #   정책 추론은 steps_per_inference*dt 마다 하지만, 페달은 이 대기
                        #   동안 계속 폴링해 즉시 처리한다. 모드가 바뀌면 대기를 끊고 바로 다음 사이클.
                        t_cycle_end = t_start + (iter_idx + cycle_steps) * dt
                        wait_mode = mode
                        while time.monotonic() < t_cycle_end - frame_latency:
                            sig = process_key(kr.poll())
                            if sig in ('keep', 'discard', 'quit'):
                                episode_done = sig; break
                            if mode != wait_mode:
                                break
                            time.sleep(0.005)
                        if episode_done is not None:
                            break
                        iter_idx += cycle_steps

                    # ── 에피소드 종료 처리 ──
                    if episode_done == 'discard':
                        env.drop_episode(); keep = False
                        print("[Actor] 에피소드 종료 (폐기)")
                    else:
                        env.end_episode(); keep = True
                        if episode_done == 'quit':
                            quit_all = True
                            print("[Actor] 종료 요청 (마지막 에피소드 유지)")
                        elif episode_done == 'keep':
                            print("[Actor] 에피소드 종료 (유지)")

                    # 에피소드 종료 시 servo release + actor 재개 상태로 복원
                    if mode == 'teleop':
                        teleop.publish(4)
                        env.resume_robot()
                        mode = 'policy'

                    # ── 유지된 에피소드를 relabel 후 learner로 전송 ──
                    if keep and C.SEND_TRANSITIONS:
                        try:
                            _ep_idx = env.replay_buffer.n_episodes - 1
                            wait_videos_finalized(
                                os.path.join(str(env.output_dir), "videos", str(_ep_idx)))
                            ep_hdf5 = relabel_last_episode_to_hdf5_box(
                                env.replay_buffer,
                                env_output_dir=str(env.output_dir),
                                out_path=os.path.join(C.ONLINE_WORKDIR, "last_episode.hdf5"),
                                frequency=frequency,
                                out_res=obs_res)
                            mailbox.send_episode(ep_hdf5)
                            print(f"[Actor] 에피소드 전송 완료: {ep_hdf5}")
                        except Exception as e:
                            print(f"[Actor] 에피소드 전송 실패: {e}")

                    if quit_all:
                        break

                    # ── 다음 에피소드 대기 ──
                    print("[Actor] Enter/b=다음 에피소드 / q=종료 (터미널 포커스)")
                    _next = False
                    while True:
                        k = kr.poll()
                        if k in ('b', 'B', '\r', '\n'):
                            _next = True; break
                        if k in ('q', 'Q', '\x1b'):
                            break
                        time.sleep(0.02)
                    if not _next:
                        break

            except KeyboardInterrupt:
                print("\n[Actor] 중단.")
                try:
                    env.end_episode()
                except Exception:
                    pass
            finally:
                try:
                    kr.__exit__(None, None, None)   # 터미널 모드 복원
                except Exception:
                    pass
                try:
                    teleop.publish(4)   # servo release(idle)로 안전 종료
                except Exception:
                    pass
                teleop.close()

    print("[Actor] 종료.")


if __name__ == "__main__":
    main()
