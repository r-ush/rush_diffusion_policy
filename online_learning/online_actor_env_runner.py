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

손(hand) 제어 (--use_hand):
  16D(팔9 + 오른손7) 체크포인트를 쓸 때 --use_hand 를 준다. 정책 제어 중엔 정책이
  /hand_joint_controller/joint_state_command 로 손을 민다. 교정('a')으로 넘어가면 팔은
  servo가, 손은 마누스 글러브가 담당한다 — actor는 pause로 팔·손 명령 발행을 모두 멈추므로
  마누스와 안 싸운다. 사람이 민 achieved 손자세(hand_pose_R)가 correction(stage=1) 타깃이
  되어 relabel 시 16D action(pose9 + hand7)으로 저장된다.

  ★ 마누스는 정책 손 발행과 '같은 토픽'을 쓰므로, 마누스 노드를 --gate-teleop 로 띄운다.
    그러면 마누스는 항상 켜둔 채로도 /teleop_control 이 교정 모드(1·2)일 때만 손을 발행하고,
    정책 모드(4)에선 침묵한다 → 평상시 actor 손 / 교정시 마누스 손 이 자동 전환된다.
    (--gate-teleop 없이 띄우면 200Hz로 항상 발행해 정책 손을 덮어써 버린다.)

실행:
  source ~/dualarm_ws/install/setup.bash
  # 터미널1: python online_learning/servo_rightarm_imp_online.py
  # 터미널2: python online_learning/online_learner.py -i <box_hand_ckpt>
  # 터미널3(손 쓸 때, 항상 켜둠): (manus_ws)$ python manus_to_aidin_rush.py --gate-teleop
  # 터미널4:
  ONLINE_WORKDIR=data/online_runs/run_hand \
  python online_learning/online_actor_env_runner.py \
      -i data/outputs/260710_insert_box_hand_wrench_abs/epoch=0900-train_loss=0.001.ckpt \
      --steps_per_inference 12 --frequency 10 --num_inference_steps 12 --use_hand
  # (learner/actor 모두 같은 ONLINE_WORKDIR·같은 16D ckpt, 손 실행은 새 workdir로 시작)
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
    add_wrench_obs_noise,
    get_real_obs_resolution, get_real_obs_dict, get_real_relative_obs_dict,
    get_abs_action_from_relative)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace

from online_learning import config_online as C
from online_learning.mailbox import FileMailbox
from online_learning.relabel_utils import relabel_last_episode_to_hdf5_box, pose_quat_to_9d
from analysis.modality_attribution.record_infer_obs import InferenceObsRecorder

OmegaConf.register_new_resolver("eval", eval, replace=True)

# 오른팔 realsense (박스삽입 eval과 동일)
RIGHT_CAMERA_SERIALS = ['126122270712']
WIN = "actor_control"

# hand 모드: 팔 9D(pos3+rot6d) + 오른손 7D
RIGHT_ARM_ACTION_DIM = 9
RIGHT_HAND_POLICY_DIM = 7


def _validate_hand_mode(shape_meta, use_hand):
    """체크포인트 shape_meta가 --use_hand 설정과 맞는지 검증 (bae eval과 동일)."""
    action_dim = int(shape_meta['action']['shape'][0])
    obs_meta = shape_meta['obs']
    hand_meta = obs_meta.get('hand_pose_R')
    hand_shape = None if hand_meta is None else tuple(hand_meta.get('shape', ()))

    expected_action_dim = RIGHT_ARM_ACTION_DIM + (
        RIGHT_HAND_POLICY_DIM if use_hand else 0
    )
    if action_dim != expected_action_dim:
        if not use_hand and action_dim == RIGHT_ARM_ACTION_DIM + RIGHT_HAND_POLICY_DIM:
            raise click.UsageError(
                'This checkpoint has a 7-DoF right-hand action. Re-run with --use_hand.'
            )
        raise click.UsageError(
            f'Hand mode={use_hand} expects action dim {expected_action_dim}, '
            f'but the checkpoint uses {action_dim}.'
        )

    if use_hand and hand_shape != (RIGHT_HAND_POLICY_DIM,):
        raise click.UsageError(
            f'--use_hand requires obs.hand_pose_R shape [{RIGHT_HAND_POLICY_DIM}], '
            f'but the checkpoint uses {hand_shape}.'
        )
    if not use_hand and hand_meta is not None:
        raise click.UsageError(
            'This checkpoint consumes hand_pose_R. Re-run with --use_hand.'
        )


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
    """새 가중치 버전이 있으면 policy에 로드. 새 버전 번호 반환.
    가중치 shape가 안 맞으면(예: 팔9D↔손16D 혼용된 낡은 workdir) 크래시 대신 스킵한다."""
    latest = mailbox.get_latest_weight_version()
    if latest is None or latest == current_version:
        return current_version
    payload = mailbox.load_weights(latest, map_location=device)
    if payload is None:
        return current_version
    try:
        missing, unexpected = policy.load_state_dict(payload["state_dict"], strict=False)
    except RuntimeError as e:
        print(f"[Actor][WARN] v{latest} 가중치 로드 실패 → hot-swap 스킵(현재 v{current_version} 유지).\n"
              f"  보통 base ckpt와 action 차원이 다른 '낡은 mailbox' 때문입니다 (예: 팔9D↔손16D).\n"
              f"  learner를 actor와 '같은' ckpt로 재시작하고 ONLINE_WORKDIR를 새 폴더로 비우세요.\n"
              f"  ({type(e).__name__}: {str(e).splitlines()[0]})")
        return current_version
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
@click.option('--wrench_noise_force_mean', '--wrench-noise-force-mean', default=0.0, type=float, show_default=True, help='정책 wrench force 채널에 더할 상수 offset(N).')
@click.option('--wrench_noise_force_uniform_min', '--wrench-noise-force-uniform-min', default=None, type=float, help='정책 wrench force 채널 균등노이즈 최소(N).')
@click.option('--wrench_noise_force_uniform_max', '--wrench-noise-force-uniform-max', default=None, type=float, help='정책 wrench force 채널 균등노이즈 최대(N).')
@click.option('--wrench_noise_force_std', '--wrench-noise-force-std', default=0.0, type=float, show_default=True, help='정책 wrench force 채널 가우시안 노이즈 std(N). <=0이면 비활성.')
@click.option('--wrench_noise_torque_mean', '--wrench-noise-torque-mean', default=0.0, type=float, show_default=True, help='정책 wrist torque 채널에 더할 상수 offset(Nm).')
@click.option('--wrench_noise_torque_uniform_min', '--wrench-noise-torque-uniform-min', default=None, type=float, help='정책 wrist torque 채널 균등노이즈 최소(Nm).')
@click.option('--wrench_noise_torque_uniform_max', '--wrench-noise-torque-uniform-max', default=None, type=float, help='정책 wrist torque 채널 균등노이즈 최대(Nm).')
@click.option('--wrench_noise_torque_std', '--wrench-noise-torque-std', default=0.0, type=float, show_default=True, help='정책 wrist torque 채널 가우시안 노이즈 std(Nm). <=0이면 비활성.')
@click.option('--wrench_noise_seed', '--wrench-noise-seed', default=None, type=int, help='정책 wrench 관측 노이즈용 RNG seed(옵션).')
@click.option('--record_wrist_wrench/--no_record_wrist_wrench', default=True, help="에피소드 timeseries HDF5에 오른손목 FT 기록.")
@click.option('--use_hand/--no_use_hand', default=False, help='오른손 7-DoF 관측·행동 제어 활성화.')
@click.option('--record_infer_obs/--no_record_infer_obs', default=True,
              help='정책 추론 obs를 <output>/eval_debug/episode_XXXXXX_infer_obs.hdf5로 덤프(modality attribution용).')
def main(input, output, robot_ip, steps_per_inference,
         max_duration, frequency, num_inference_steps,
         wrench_noise_force_mean, wrench_noise_force_std,
         wrench_noise_force_uniform_min, wrench_noise_force_uniform_max,
         wrench_noise_torque_mean, wrench_noise_torque_std,
         wrench_noise_torque_uniform_min, wrench_noise_torque_uniform_max,
         wrench_noise_seed, record_wrist_wrench, use_hand, record_infer_obs):
    ckpt_path = input if input is not None else C.BASE_CKPT
    output_dir = output if output is not None else C.ACTOR_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    mailbox = FileMailbox(C.ONLINE_WORKDIR)
    device = torch.device(C.DEVICE if torch.cuda.is_available() else "cpu")

    # ── base policy 로드 ──
    print(f"[Actor] base 체크포인트 로드: {ckpt_path}")
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    _validate_hand_mode(cfg.task.shape_meta, use_hand)
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
    wrench_noise_rng = np.random.default_rng(wrench_noise_seed)
    print(f"[Actor] right hand control: {'enabled' if use_hand else 'disabled'}, "
          f"action_dim={action_dim}")
    print(f"[Actor] wrench noise force mean/uniform/std: {wrench_noise_force_mean} / "
          f"({wrench_noise_force_uniform_min},{wrench_noise_force_uniform_max}) / {wrench_noise_force_std}")
    print(f"[Actor] wrench noise torque mean/uniform/std: {wrench_noise_torque_mean} / "
          f"({wrench_noise_torque_uniform_min},{wrench_noise_torque_uniform_max}) / {wrench_noise_torque_std}")

    def make_obs_dict_np(obs):
        # 정책에 실제로 들어가는 numpy obs dict (get_real_obs_dict + wrench noise, 배치차원 전).
        # infer_obs 덤프(record_infer_obs)가 이 스냅샷을 저장한다.
        if obs_pose_repr == 'relative':
            d = get_real_relative_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta)
        else:
            d = get_real_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta)
        d = add_wrench_obs_noise(
            d,
            cfg.task.shape_meta,
            rng=wrench_noise_rng,
            force_mean=wrench_noise_force_mean,
            force_uniform_min=wrench_noise_force_uniform_min,
            force_uniform_max=wrench_noise_force_uniform_max,
            force_std=wrench_noise_force_std,
            torque_mean=wrench_noise_torque_mean,
            torque_uniform_min=wrench_noise_torque_uniform_min,
            torque_uniform_max=wrench_noise_torque_uniform_max,
            torque_std=wrench_noise_torque_std,
        )
        return d

    def obs_np_to_tensor(d):
        return dict_apply(d, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))

    def make_obs_dict(obs):
        return obs_np_to_tensor(make_obs_dict_np(obs))

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
            record_wrist_wrench=record_wrist_wrench,
            use_hand=use_hand,
            shm_manager=shm_manager,
        ) as env:
            # obs의 wrench_wrist_R(6,32) 기록은 shape_meta(wrench 키)로 결정되어 replay_buffer에
            #   저장되고 relabel에 쓰인다. record_wrist_wrench=True면 그와 별개로 최신 6축 손목
            #   FT를 per-episode timeseries HDF5로도 남긴다(진단용). use_hand=True면 정책이 손도
            #   제어하고 hand_pose_R(7)이 obs/replay_buffer에 기록된다.
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
                    inference_index = 0   # 정책 추론 카운터 (infer_obs 라벨)
                    obs_recorder = InferenceObsRecorder() if record_infer_obs else None
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
                            obs_dict_np = make_obs_dict_np(obs)
                            with torch.no_grad():
                                result = policy.predict_action(obs_np_to_tensor(obs_dict_np))
                                action = result['action'][0].detach().cpu().numpy()  # [H,9 또는 16]
                            # 이 추론에 실제로 들어간 obs를 덤프(로봇 없이 replay_offline로 재생/분석)
                            if obs_recorder is not None:
                                obs_recorder.add(inference_index, obs_dict_np,
                                                 obs_timestamps, eval_t_start)
                                inference_index += 1

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
                            # teleop: servo가 팔을, (use_hand면) manus가 손을 몰고,
                            # actor는 achieved pose(+hand)를 correction(stage=1)으로 기록만
                            # 한다 (로봇에 명령 X). 사람이 민 결과가 그대로 학습 타깃.
                            cur_pose = np.asarray(obs['robot_pose_R'])[-1]
                            cur_quat = np.asarray(obs['robot_quat_R'])[-1]
                            act = pose_quat_to_9d(cur_pose[None], cur_quat[None]).astype(np.float64)
                            if use_hand:
                                cur_hand = np.asarray(obs['hand_pose_R'])[-1].astype(np.float64)
                                act = np.concatenate([act, cur_hand[None]], axis=1)   # (1,16)
                            env.exec_actions(actions=act,
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

                    # ── 정책 추론 obs 덤프 저장 (유지된 에피소드) ──
                    #   <output>/eval_debug/episode_XXXXXX_infer_obs.hdf5 →
                    #   analysis.modality_attribution.replay_offline --obs 로 재생/분석.
                    if keep and obs_recorder is not None and len(obs_recorder) > 0:
                        try:
                            _ep_idx = env.replay_buffer.n_episodes - 1
                            obs_recorder.save(output_dir, _ep_idx)
                        except Exception as e:
                            print(f"[Actor] infer_obs 덤프 저장 실패: {e}")

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
                                out_res=obs_res,
                                use_hand=use_hand)
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
