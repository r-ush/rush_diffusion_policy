#!/usr/bin/env python
"""
온라인 residual DAgger Actor (박스삽입, 오른팔+손, wrench) — slow+fast residual 판.

finetune_teleop_actor_env_runner.py(full-finetune 판)의 검증된 제어/교정 골격을 따르되,
추론을 **frozen slow chunk + per-step fast residual** 로 바꾸고, hot-swap 을 **head-only**
로, 전송을 **residual 포맷** 으로 바꾼 것. 추론 조립은 검증된
diffusion_policy/residual_policy/eval_real_robot_rightarm_insert_plug.py 의 헬퍼를 재사용.

핵심 설계 — one-step-per-tick:
  residual 은 매 tick 최신 force 를 봐야 하므로 **한 tick 에 한 스텝**만 실행한다(eval 과 동일).
  덕분에 "그 tick 의 obs" 와 "그 tick 에 쓴 slow_pred" 가 1:1 로 정렬되어, learner 가 기대하는
  residual 포맷(residual_relabel_utils)이 정확히 만들어진다.

온라인 훅 3개:
  1. head hot-swap (에피소드 경계): full state_dict 아니라 head+normalizer(+force_encoder)만.
  2. slow_pred 로깅 (매 tick): 그 스텝의 base(slow 예측 abs pose9) 를 프레임에 같이 기록.
  3. relabel+전송 (유지 시): write_residual_episode_hdf5 → mailbox.send_episode.

★★★ 이 파일은 로봇/카메라/servo 없이 실행 검증 불가(하드웨어 필요). 문법·구조는 맞췄으나
    로봇 앞에서 아래 [VERIFY] 표시 지점을 반드시 실측 확인/튜닝할 것.
    상세 절차: online_learning/RESIDUAL_ONLINE_ROBOT_RUNBOOK.md

실행 (터미널4):
  RESIDUAL_ONLINE_WORKDIR=data/online_runs/run_hand_residual \
  RESIDUAL_SLOW_CKPT=/media/rush/.../260714_insert_box_hand_rel/epoch=0500-train_loss=0.002.ckpt \
  /home/rush/anaconda3/envs/bae_robodiff/bin/python \
    online_learning/residual_teleop_actor_env_runner.py --use_hand \
      --steps_per_inference 6 --frequency 10 --num_inference_steps 12
"""
# huggingface_hub 버전 충돌 회피 (다른 import보다 먼저) — online actor 와 동일
import sys
local_paths = [p for p in sys.path if '.local' in p]
sys.path = [p for p in sys.path if '.local' not in p]
import huggingface_hub  # noqa: F401
sys.path = local_paths + sys.path

import os
import time
import copy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import dill
import hydra
import click
from hydra import compose, initialize_config_dir
from multiprocessing.managers import SharedMemoryManager
from omegaconf import OmegaConf

from diffusion_policy.real_world.bae_real_env_rightarm_hand_insert_plug import DualarmRealEnv
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution,
    get_abs_action_from_relative,
    get_relative_action_from_abs,
)
from diffusion_policy.residual_policy.pose_util import apply_residual_action_to_pose9

# residual 추론 obs 조립 헬퍼 재사용 (eval 스크립트, 모듈 로드는 하드웨어 import 없음)
from diffusion_policy.residual_policy.eval_real_robot_rightarm_insert_plug import (
    _get_policy_obs_dict,
    _latest_obs_only,
    _strip_fast_only_obs,
    _build_fixed_context_fast_obs,
    _to_torch_obs,
)
# 교정/키 입력 헬퍼 재사용 (finetune teleop actor)
from online_learning.finetune_teleop_actor_env_runner import (
    TeleopControlPub,
    KeyReader,
    _validate_hand_mode,
)
from online_learning import config_residual_online as C
from online_learning.mailbox import FileMailbox
from online_learning.residual_relabel_utils import write_residual_episode_hdf5, RAW_OBS_KEYS

OmegaConf.register_new_resolver("eval", eval, replace=True)

RIGHT_CAMERA_SERIALS = ['126122270712']   # 오른팔 realsense (박스삽입 eval과 동일)


def maybe_hotswap_residual_head(mailbox, fast_policy, current_version, device):
    """새 head 가중치가 있으면 fast_policy 에 로드(head+normalizer+선택 force_encoder만).
    slow(frozen)는 안 건드린다. 새 버전 번호 반환."""
    latest = mailbox.get_latest_weight_version()
    if latest is None or latest == current_version:
        return current_version
    payload = mailbox.load_weights(latest, map_location=device)
    if payload is None or "head_state" not in payload:
        return current_version
    try:
        fast_policy.head.load_state_dict(
            {k: v.to(device) for k, v in payload["head_state"].items()})
        fast_policy.normalizer.load_state_dict(
            {k: v.to(device) for k, v in payload["normalizer_state"].items()}, strict=False)
        if payload.get("force_encoder_state") is not None and fast_policy.force_encoder is not None:
            fast_policy.force_encoder.load_state_dict(
                {k: v.to(device) for k, v in payload["force_encoder_state"].items()})
    except RuntimeError as e:
        print(f"[Actor][WARN] head v{latest} 로드 실패 → hot-swap 스킵. ({type(e).__name__}: "
              f"{str(e).splitlines()[0]})")
        return current_version
    fast_policy.eval().to(device)
    print(f"[Actor] head hot-swap v{current_version} -> v{latest} "
          f"(learner demo={payload.get('num_demos', '?')})")
    return latest


def _fast_head_ready(fast_policy):
    """learner 가 1회 이상 학습해 normalizer(obs+action 통계)를 채워 발행했는지.
    learner 는 시작 시 학습 전(demo=0) v0 을 한 번 발행하는데, 그때 normalizer 는
    아직 set_normalizer 전이라 비어 있다. 빈 normalizer 로 predict_action 하면
    _build_head_input 의 normalize 가 image0 키에서 죽는다(AttributeError).
    -> normalizer 가 비어 있으면 residual 을 적용하지 않고 slow-only 로 돌며 교정만 모은다.
    (교정 데이터 0개일 때 residual=0 은 DAgger 부트스트랩상으로도 올바른 동작.)"""
    try:
        return len(fast_policy.normalizer.params_dict) > 0
    except AttributeError:
        return False


def _to_uint8_image(img):
    """live obs 이미지를 residual 데이터셋이 기대하는 uint8 HWC 로.
    [VERIFY] env(obs_float32=True) 이미지 스케일을 로봇에서 확인:
      0~1 이면 *255, 0~255 이면 그대로. 아래는 max 로 추정(안전판)."""
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    if img.max() <= 1.5:      # [0,1] 로 추정
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


@click.command()
@click.option('--input', '-i', default=None, help='slow(base) ckpt. 미지정 시 config_residual_online.SLOW_CKPT.')
@click.option('--output', '-o', default=None, help='에피소드 저장 폴더. 미지정 시 C.ACTOR_OUTPUT_DIR.')
@click.option('--config_name', default=None, help='residual hydra config. 미지정 시 C.RESIDUAL_CONFIG_NAME.')
@click.option('--robot_ip', '-ri', default="192.168.111.50")
@click.option('--steps_per_inference', '-si', default=6, type=int, help='slow chunk 재계획 전 실행할 스텝 수.')
@click.option('--max_duration', '-md', default=60, type=float, help='에피소드 최대 길이(초).')
@click.option('--frequency', '-f', default=10.0, type=float, help='제어 주기(Hz).')
@click.option('--num_inference_steps', '-ni', default=12, type=int, help='slow DDIM 스텝 수.')
@click.option('--residual_translation_cap', default=0.05, type=float, help='[VERIFY] residual 병진 캡(m).')
@click.option('--residual_rotation_cap', default=0.4, type=float, help='[VERIFY] residual 회전 캡(rad).')
@click.option('--use_hand/--no_use_hand', default=True, help='오른손 7-DoF 관측·행동(16D base).')
@click.option('--record_wrist_wrench/--no_record_wrist_wrench', default=True)
def main(input, output, config_name, robot_ip, steps_per_inference, max_duration,
         frequency, num_inference_steps, residual_translation_cap, residual_rotation_cap,
         use_hand, record_wrist_wrench):
    slow_ckpt = input if input is not None else C.SLOW_CKPT
    output_dir = output if output is not None else C.ACTOR_OUTPUT_DIR
    cfg_name = config_name if config_name is not None else C.RESIDUAL_CONFIG_NAME
    os.makedirs(output_dir, exist_ok=True)

    mailbox = FileMailbox(C.ONLINE_WORKDIR)
    device = torch.device(C.DEVICE if torch.cuda.is_available() else "cpu")

    # ── slow(base) cfg 로드 (env shape_meta / obs 구성 / n_obs_steps 용) ──
    print(f"[Actor] slow(base) ckpt: {slow_ckpt}")
    slow_payload = torch.load(open(slow_ckpt, "rb"), pickle_module=dill, weights_only=False)
    slow_cfg = slow_payload["cfg"]
    del slow_payload
    _validate_hand_mode(slow_cfg.task.shape_meta, use_hand)
    env_shape_meta = slow_cfg.task.shape_meta                 # image0,pose,quat,hand,wrench
    obs_res = get_real_obs_resolution(env_shape_meta)
    n_obs_steps = slow_cfg.n_obs_steps

    # ── residual(fast) 정책 인스턴스화 (내부에서 slow 가중치 로드) ──
    config_dir = os.path.join(ROOT, "diffusion_policy", "config")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        res_cfg = compose(config_name=cfg_name)
    res_cfg.task.slow_ckpt_path = slow_ckpt      # '=' 포함 경로라 override 대신 직접 대입
    OmegaConf.resolve(res_cfg)
    print("[Actor] residual 정책 인스턴스화 ...")
    fast_policy = hydra.utils.instantiate(res_cfg.policy).to(device).eval()
    slow_policy = fast_policy.slow_policy
    slow_policy.num_inference_steps = num_inference_steps
    slow_policy.n_action_steps = slow_policy.horizon - slow_policy.n_obs_steps + 1

    fast_shape_meta = res_cfg.task.shape_meta
    slow_shape_meta = _strip_fast_only_obs(fast_shape_meta)   # base_action_rel 제거 = raw obs
    base_action_key = fast_policy.base_action_key
    slow_obs_pose_repr = getattr(slow_policy, "obs_pose_repr",
                                 OmegaConf.select(slow_cfg, "task.pose_repr.obs_pose_repr", default="abs"))
    slow_action_pose_repr = getattr(slow_policy, "action_pose_repr", "relative")
    fast_obs_pose_repr = OmegaConf.select(res_cfg, "task.pose_repr.obs_pose_repr", default=slow_obs_pose_repr)
    fast_n_obs = int(getattr(fast_policy, "n_obs_steps", 16))
    action_dim = env_shape_meta.action.shape[0]              # 16(hand) or 9
    dt = 1.0 / frequency
    print(f"[Actor] action_dim={action_dim}, use_hand={use_hand}, slow n_obs={n_obs_steps}, "
          f"fast n_obs={fast_n_obs}, repr slow obs/action={slow_obs_pose_repr}/{slow_action_pose_repr}")

    def cap_residual(residual6):
        """[VERIFY] residual delta6 캡 — 방향 유지, 크기만 상한(초기 발산 안전)."""
        r = np.asarray(residual6, dtype=np.float64).copy()
        tn = np.linalg.norm(r[:3])
        if tn > residual_translation_cap:
            r[:3] = r[:3] / tn * residual_translation_cap
        rn = np.linalg.norm(r[3:6])
        if rn > residual_rotation_cap:
            r[3:6] = r[3:6] / rn * residual_rotation_cap
        return r

    weight_version = -1

    with SharedMemoryManager() as shm_manager:
        with DualarmRealEnv(
            output_dir=output_dir, robot_ip=robot_ip, frequency=frequency,
            camera_serial_numbers=RIGHT_CAMERA_SERIALS, n_obs_steps=n_obs_steps,
            shape_meta=env_shape_meta, obs_image_resolution=obs_res, obs_float32=True,
            init_joints=False, enable_multi_cam_vis=False, record_raw_video=False,
            thread_per_video=3, video_crf=21, record_wrist_wrench=record_wrist_wrench,
            use_hand=use_hand, shm_manager=shm_manager,
        ) as env:
            print("[Actor] env READY. 센서 워밍업...")
            time.sleep(2.0)

            # ── warmup: slow + fast 한 번 ──
            obs = env.get_obs()
            with torch.no_grad():
                slow_obs = _to_torch_obs(_get_policy_obs_dict(obs, slow_shape_meta, slow_obs_pose_repr), device)
                slow_res = slow_policy.predict_action(slow_obs)
                assert slow_res["action"].shape[-1] == action_dim, \
                    f"slow action dim {slow_res['action'].shape[-1]} != {action_dim}"
                del slow_res
            teleop = TeleopControlPub()
            print("[Actor] 준비 완료. a=교정핸드오프 b=복귀 c=홈 | s=유지+전송 d=폐기 q=종료")

            mode = 'policy'

            # 핸드오프 정착 지연: pause/resume 은 큐 기반이라 컨트롤러가 실제로 발행을
            # 멈추기까지 최대 한 제어 사이클 걸린다. servo/manus 로 넘기기 전에 이만큼
            # 기다려야 같은 토픽(task_space_command / hand joint_state_command)에 컨트롤러와
            # servo/manus 가 동시에 쏘는 창이 사라진다(개입 순간 로봇 튐 방지).
            HANDOFF_SETTLE_S = 0.15

            def process_key(key):
                nonlocal mode
                if key is None:
                    return None
                if key in ('a', 'A'):
                    if mode != 'teleop':
                        # 먼저 컨트롤러 발행 중단 → 정착 → 그 다음에 servo/manus 인계
                        env.pause_robot(); time.sleep(HANDOFF_SETTLE_S)
                        teleop.publish(1); mode = 'teleop'
                        print("[Actor] 핸드오프 → servo (correction 기록)")
                elif key in ('b', 'B'):
                    if mode != 'policy':
                        # 먼저 servo/manus 중단 → 정착 → 그 다음에 컨트롤러 재개(현재 pose 재동기화)
                        teleop.publish(4); time.sleep(HANDOFF_SETTLE_S)
                        env.resume_robot(); mode = 'policy'
                        print("[Actor] 복귀 → 정책 제어")
                elif key in ('c', 'C'):
                    env.pause_robot(); time.sleep(HANDOFF_SETTLE_S)
                    teleop.publish(2); mode = 'teleop'
                    print("[Actor] 홈복귀(servo). b로 복귀.")
                elif key in ('s', 'S'):
                    return 'keep'
                elif key in ('d', 'D'):
                    return 'discard'
                elif key in ('q', 'Q', '\x1b'):
                    return 'quit'
                return None

            kr = KeyReader(); kr.__enter__()
            try:
                while True:
                    # ── 에피소드 시작 전 head hot-swap ──
                    weight_version = maybe_hotswap_residual_head(mailbox, fast_policy, weight_version, device)
                    head_ready = _fast_head_ready(fast_policy)
                    if not head_ready:
                        print("[Actor] 학습된 residual head 아직 없음(normalizer 비어있음) → "
                              "이번 에피소드는 slow-only 실행 + 교정 데이터만 수집. "
                              f"(a=교정 → s=전송, {C.MIN_EPISODES_BEFORE_TRAIN}개 이상 쌓이면 learner 학습→발행)")

                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    precise_wait(eval_t_start - 1.0 / 30, time_func=time.time)
                    print("[Actor] 에피소드 시작!")

                    iter_idx = 0
                    mode = 'policy'
                    episode_done = None
                    frames = []                     # per-tick 프레임(RAW_OBS + slow_pred)
                    slow_abs_seq = None             # 현재 slow chunk (abs, [H, action_dim])
                    slow_step = 0                   # chunk 내 다음 실행 스텝
                    fast_ctx = None                 # fixed-context fast obs(chunk 시작 obs)
                    base_hist, wr_hist, ld_hist = [], {}, {}

                    while True:
                        obs = env.get_obs()
                        obs_ts = obs['timestamp']
                        sig = process_key(kr.poll())
                        if sig is not None:
                            episode_done = sig; break

                        # ── slow chunk 재계획 ──
                        #   정책 모드: chunk 소진 시에만(부드러운 궤적, slow 연산 amortize).
                        #   교정 모드: 매 tick 신선한 slow 예측을 base 로 → residual =
                        #     achieved(다음) - 신선한 slow 예측 = 순수 교정 신호(stale drift 없음).
                        #     (교정은 사람 속도라 tick 당 slow 추론 ~0.2s 여도 무방.)
                        usable = 0 if slow_abs_seq is None else min(steps_per_inference, len(slow_abs_seq))
                        if slow_abs_seq is None or slow_step >= usable or mode == 'teleop':
                            with torch.no_grad():
                                slow_obs = _to_torch_obs(
                                    _get_policy_obs_dict(obs, slow_shape_meta, slow_obs_pose_repr), device)
                                slow_action = slow_policy.predict_action(slow_obs)["action"][0].cpu().numpy()
                            if slow_action_pose_repr == "relative":
                                slow_abs_seq = get_abs_action_from_relative(action=slow_action, env_obs=obs)
                            else:
                                slow_abs_seq = slow_action
                            slow_step = 0
                            # fast fixed-context = 이 obs 기준
                            fast_ctx = _latest_obs_only(
                                _get_policy_obs_dict(obs, fast_shape_meta, fast_obs_pose_repr), fast_shape_meta)
                            base_hist, wr_hist, ld_hist = [], {}, {}

                        slow_abs = slow_abs_seq[slow_step]                 # (action_dim,)
                        slow_pose9 = np.asarray(slow_abs[:9], dtype=np.float64)
                        slow_hand = np.asarray(slow_abs[9:], dtype=np.float64) if action_dim > 9 else None

                        # ── fast residual (학습된 head 있을 때만; 없으면 slow-only=0) ──
                        if head_ready:
                            base_action_rel = get_relative_action_from_abs(action=slow_pose9[None], env_obs=obs)[0]
                            fast_obs_np = _build_fixed_context_fast_obs(
                                context_obs_dict_np=fast_ctx,
                                latest_obs_dict_np=_latest_obs_only(
                                    _get_policy_obs_dict(obs, fast_shape_meta, fast_obs_pose_repr), fast_shape_meta),
                                base_action_rel=base_action_rel,
                                base_action_history=base_hist, wrench_history=wr_hist, low_dim_history=ld_hist,
                                fast_policy=fast_policy, max_steps=fast_n_obs)
                            with torch.no_grad():
                                residual6 = fast_policy.predict_action(
                                    _to_torch_obs(fast_obs_np, device))["action"][0, 0].cpu().numpy()
                            residual6 = cap_residual(residual6)
                        else:
                            residual6 = np.zeros(6, dtype=np.float64)
                        final_pose9 = apply_residual_action_to_pose9(slow_pose9, residual6)
                        if slow_hand is not None:
                            final_action = np.concatenate([final_pose9, slow_hand]).astype(np.float64)
                        else:
                            final_action = final_pose9.astype(np.float64)

                        # ── 실행(정책 모드만) / 프레임 기록(항상) ──
                        action_ts = obs_ts[-1] + dt
                        if mode == 'policy':
                            env.exec_actions(actions=final_action[None],
                                             timestamps=np.array([action_ts]),
                                             stages=np.zeros(1, dtype=np.int64))
                        # teleop 모드: 로봇은 servo/사람이 몰고, actor는 명령 안 함(프레임만 기록)

                        frames.append({
                            "image0": _to_uint8_image(np.asarray(obs["image0"])[-1]),
                            "robot_pose_R": np.asarray(obs["robot_pose_R"])[-1].astype(np.float32),
                            "robot_quat_R": np.asarray(obs["robot_quat_R"])[-1].astype(np.float32),
                            "hand_pose_R": (np.asarray(obs["hand_pose_R"])[-1].astype(np.float32)
                                            if "hand_pose_R" in obs else np.zeros(7, np.float32)),
                            "wrench_wrist_R": np.asarray(obs["wrench_wrist_R"])[-1].astype(np.float32),
                            "slow_pred_target_abs": slow_pose9.astype(np.float32),
                        })
                        slow_step += 1

                        if time.monotonic() - t_start > max_duration:
                            episode_done = 'keep'; print("[Actor] 타임아웃(유지)"); break

                        # ── 다음 tick 까지 키 반응형 대기 ──
                        t_cycle_end = t_start + (iter_idx + 1) * dt
                        wait_mode = mode
                        while time.monotonic() < t_cycle_end - 1.0 / 30:
                            s2 = process_key(kr.poll())
                            if s2 in ('keep', 'discard', 'quit'):
                                episode_done = s2; break
                            if mode != wait_mode:
                                break
                            time.sleep(0.005)
                        if episode_done is not None:
                            break
                        iter_idx += 1
                        # 모드가 바뀌었으면 slow chunk 무효화(새 상황에서 재계획)
                        if mode != wait_mode:
                            slow_abs_seq = None

                    # ── 에피소드 종료 처리 ──
                    keep = episode_done != 'discard'
                    if episode_done == 'discard':
                        env.drop_episode(); print("[Actor] 폐기")
                    else:
                        env.end_episode()
                        print("[Actor] 종료(유지)" if episode_done != 'quit' else "[Actor] 종료 요청(마지막 유지)")
                    if mode == 'teleop':
                        teleop.publish(4); env.resume_robot(); mode = 'policy'

                    # ── residual 포맷으로 relabel + 전송 ──
                    if keep and C.SEND_TRANSITIONS and len(frames) >= 2:
                        try:
                            ep = {k: np.stack([f[k] for f in frames], axis=0)
                                  for k in RAW_OBS_KEYS + ["slow_pred_target_abs"]}
                            out = write_residual_episode_hdf5(
                                os.path.join(C.ONLINE_WORKDIR, "last_episode.hdf5"), ep, demo_name="demo_0")
                            mailbox.send_episode(out)
                            print(f"[Actor] residual 에피소드 전송: {out} ({len(frames)} frames)")
                        except Exception as e:
                            print(f"[Actor] 전송 실패: {e}")
                    elif keep:
                        print(f"[Actor] 프레임 부족({len(frames)}) — 전송 스킵")

                    if episode_done == 'quit':
                        break
                    print("[Actor] Enter/b=다음 에피소드 / q=종료")
                    nxt = False
                    while True:
                        k = kr.poll()
                        if k in ('b', 'B', '\r', '\n'):
                            nxt = True; break
                        if k in ('q', 'Q', '\x1b'):
                            break
                        time.sleep(0.02)
                    if not nxt:
                        break
            except KeyboardInterrupt:
                print("\n[Actor] 중단.")
                try:
                    env.end_episode()
                except Exception:
                    pass
            finally:
                kr.__exit__()
                try:
                    teleop.close()
                except Exception:
                    pass
    print("[Actor] 종료.")


if __name__ == "__main__":
    main()
