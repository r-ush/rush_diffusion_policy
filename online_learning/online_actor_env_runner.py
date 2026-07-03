#!/home/vision/miniconda3/envs/robodiff/bin/python
"""
온라인 학습 Actor (실제 로봇, 오른팔 임피던스에서 실행).

CR-DAgger의 residual_online_learning_env_runner.py에 대응. rush_eval_real_robot_imp_right.py
의 검증된 제어 루프를 그대로 따르고, 두 가지 온라인 훅만 추가한다:

  (1) 매 에피소드 시작 시 weights mailbox를 확인해 새 버전이 있으면 policy 가중치를
      hot-swap (learner가 fine-tune한 결과를 실시간 반영).
  (2) 사람이 correction한 에피소드를 유지(keep)하면, achieved-pose로 relabel한 뒤
      단일 demo HDF5로 만들어 mailbox로 learner에 전송. (이미지는 zarr가 아니라
      videos/*.mp4 에서 디코드 — online_learning/relabel_utils 참고)

조작 (rush_eval_real_robot_imp.py와 동일):
  C : correction 토글 (correction ON일 때 팔을 물리적으로 밀어 교정. stage=1로 기록)
  S : 에피소드 종료 + 유지(relabel 후 learner로 전송)
  D : 에피소드 종료 + 폐기
  Ctrl+C : 종료
  (타임아웃(max_duration)으로 끝난 에피소드도 유지·전송된다.)

⚠️ 이 스크립트는 실제 로봇/카메라(RightarmRealEnvImp)가 있어야 동작한다. 로봇 없이
   learner+통신 로직만 검증하려면 online_learning/smoke_test_no_robot.py 를 실행.

실행:
  conda activate robodiff
  cd /home/rush/rush_diffusion_policy
  # 터미널1: python online_learning/online_learner.py -i <ckpt>
  # 터미널2:
  python online_learning/online_actor_env_runner.py \
      -i data/outputs/logistics_abs_10/epoch=0800-train_loss=0.000.ckpt \
      --steps_per_inference 12 --frequency 10 --num_inference_steps 12
"""
# ============================================================
# huggingface_hub 버전 충돌 회피 (오른팔 eval과 동일 shim, 다른 import보다 먼저)
# ============================================================
import sys
local_paths = [p for p in sys.path if '.local' in p]
sys.path = [p for p in sys.path if '.local' not in p]
import huggingface_hub  # noqa: F401
sys.path = local_paths + sys.path

import os
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import dill
import hydra
import click
import cv2
from multiprocessing.managers import SharedMemoryManager
from omegaconf import OmegaConf

from diffusion_policy.real_world.rush_real_env_rightarm_imp import RightarmRealEnvImp
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, get_real_obs_dict, get_real_relative_obs_dict,
    get_abs_action_from_relative)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace

from online_learning import config_online as C
from online_learning.mailbox import FileMailbox
from online_learning.relabel_utils import relabel_last_episode_to_hdf5

OmegaConf.register_new_resolver("eval", eval, replace=True)


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


@click.command()
@click.option('--input', '-i', default=None,
              help='Path to base checkpoint (.ckpt). 미지정 시 config_online.BASE_CKPT 사용.')
@click.option('--output', '-o', default=None,
              help='에피소드 저장 폴더. 미지정 시 config_online.ACTOR_OUTPUT_DIR 사용.')
@click.option('--robot_ip', '-ri', default="192.168.111.50",
              help="Robot IP (RightarmRealEnvImp 호환용).")
@click.option('--steps_per_inference', '-si', default=12, type=int,
              help="한 번의 inference에서 실행할 action 스텝 수.")
@click.option('--max_duration', '-md', default=60, type=float,
              help='에피소드 최대 길이(초). 초과 시 자동 종료(유지·전송).')
@click.option('--frequency', '-f', default=10.0, type=float,
              help="제어 주기(Hz). 영상 프레임 정렬(relabel)에도 사용.")
@click.option('--num_inference_steps', '-ni', default=12, type=int,
              help="DDIM denoising 스텝 수.")
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
    if 'diffusion' not in cfg.name:
        raise RuntimeError(f"Unsupported policy type: {cfg.name}")
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.eval().to(device)
    policy.num_inference_steps = num_inference_steps
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    # pose 표현 (abs/relative) — 오른팔 eval과 동일하게 cfg에서 결정
    obs_pose_repr = OmegaConf.select(cfg, "task.pose_repr.obs_pose_repr", default=None)
    action_pose_repr = OmegaConf.select(cfg, "task.pose_repr.action_pose_repr", default=None)
    print(f"[Actor] pose_repr: obs={obs_pose_repr}, action={action_pose_repr}")

    weight_version = -1
    dt = 1.0 / frequency
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)   # (width, height)=(320,240)
    n_obs_steps = cfg.n_obs_steps

    def make_obs_dict(obs):
        if obs_pose_repr == 'relative':
            d = get_real_relative_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta)
        else:
            d = get_real_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta)
        return dict_apply(d, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))

    with SharedMemoryManager() as shm_manager:
        with RightarmRealEnvImp(
            output_dir=output_dir,
            robot_ip=robot_ip,
            frequency=frequency,
            n_obs_steps=n_obs_steps,
            obs_image_resolution=obs_res,
            obs_float32=True,
            enable_multi_cam_vis=True,
            record_raw_video=False,
            thread_per_video=3,
            video_crf=21,
            shm_manager=shm_manager,
        ) as env:
            cv2.setNumThreads(1)
            print("[Actor] 센서 워밍업...")
            time.sleep(2.0)

            # ── warmup inference ──
            print("[Actor] 워밍업 추론...")
            obs = env.get_obs()
            with torch.no_grad():
                policy.reset()
                result = policy.predict_action(make_obs_dict(obs))
                a = result['action'][0].detach().cpu().numpy()
                assert a.shape[-1] == 9, f"Expected action dim=9, got {a.shape[-1]}"
                del result
            print("[Actor] 준비 완료. C=correction 토글 / S=유지 / D=폐기 / Ctrl+C=종료")

            while True:
                # ── 에피소드 시작 전: 새 가중치 확인 & hot-swap ──
                weight_version = maybe_hotswap_weights(mailbox, policy, weight_version, device)

                try:
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    precise_wait(eval_t_start - 1.0 / 30, time_func=time.time)
                    print("[Actor] 에피소드 시작!")

                    iter_idx = 0
                    correction_active = False
                    keep = True
                    while True:
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']

                        # ── inference ──
                        with torch.no_grad():
                            result = policy.predict_action(make_obs_dict(obs))
                            action = result['action'][0].detach().cpu().numpy()  # [H,9]

                        this_target_poses = np.zeros((len(action), action.shape[-1]),
                                                     dtype=np.float64)
                        this_target_poses[:] = action[:]

                        # relative action -> abs target pose (오른팔 eval과 동일)
                        if action_pose_repr == 'relative':
                            if np.any(np.abs(this_target_poses[..., :3]) > 0.5):
                                print("[Actor][WARN] 비정상적으로 큰 relative translation!")
                            this_target_poses = get_abs_action_from_relative(
                                action=this_target_poses, env_obs=obs)

                        # ── 타이밍: 이미 지난 action 제거 + steps_per_inference 절단 ──
                        action_timestamps = (
                            np.arange(len(action), dtype=np.float64)) * dt + obs_timestamps[-1]
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + 0.01)
                        if np.sum(is_new) == 0:
                            this_target_poses = this_target_poses[[-1]]
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamps = np.array([eval_t_start + next_step_idx * dt])
                            print("[Actor][WARN] Over budget! 마지막 action 실행.")
                        else:
                            this_target_poses = this_target_poses[is_new][:steps_per_inference]
                            action_timestamps = action_timestamps[is_new][:steps_per_inference]

                        # ── 키 입력: correction 토글 / 종료 ──
                        key_stroke = cv2.pollKey()
                        if key_stroke in (ord('c'), ord('C')):
                            correction_active = not correction_active
                            print(f"[Actor] correction={'ON (사람 교정중)' if correction_active else 'off'}")

                        # ── execute ──
                        stages = np.full(len(this_target_poses),
                                         1 if correction_active else 0, dtype=np.int64)
                        env.exec_actions(actions=this_target_poses,
                                         timestamps=action_timestamps, stages=stages)

                        if key_stroke in (ord('s'), ord('S')):
                            env.end_episode(); keep = True
                            print("[Actor] 에피소드 종료 (유지)"); break
                        if key_stroke in (ord('d'), ord('D')):
                            env.drop_episode(); keep = False
                            print("[Actor] 에피소드 종료 (폐기)"); break

                        if time.monotonic() - t_start > max_duration:
                            env.end_episode(); keep = True
                            print("[Actor] 타임아웃 종료 (유지)"); break

                        precise_wait(t_cycle_end - 1.0 / 30)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("\n[Actor] 중단.")
                    env.end_episode()
                    break

                # ── 유지된 에피소드를 relabel 후 learner로 전송 ──
                if keep and C.SEND_TRANSITIONS:
                    try:
                        ep_hdf5 = relabel_last_episode_to_hdf5(
                            env.replay_buffer,
                            env_output_dir=str(env.output_dir),
                            out_path=os.path.join(C.ONLINE_WORKDIR, "last_episode.hdf5"),
                            frequency=frequency,
                            out_res=obs_res)
                        mailbox.send_episode(ep_hdf5)
                    except Exception as e:
                        print(f"[Actor] 에피소드 전송 실패: {e}")

                print("[Actor] Enter=다음 에피소드 / Ctrl+C=종료")
                try:
                    sys.stdin.readline()
                except (KeyboardInterrupt, EOFError):
                    break

    print("[Actor] 종료.")


if __name__ == "__main__":
    main()
