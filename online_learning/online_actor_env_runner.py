#!/usr/bin/env python
"""
온라인 학습 Actor (실제 로봇에서 실행).

CR-DAgger의 residual_online_learning_env_runner.py에 대응. rush_eval_real_robot_imp.py의
검증된 제어 루프를 기반으로, 두 가지 온라인 훅을 추가:

  (1) 매 에피소드 시작 시 weights mailbox를 확인해 새 버전이 있으면 policy 가중치를
      hot-swap (learner가 fine-tune한 결과를 실시간 반영).
  (2) 사람이 correction한 에피소드를 유지(keep)하면, achieved-pose로 relabel한 뒤
      단일 demo HDF5로 만들어 mailbox로 learner에 전송.

조작 (rush_eval_real_robot_imp.py와 동일):
  C : correction 토글 (페달을 이 키에 매핑해도 됨)
  S : 에피소드 종료 + 유지(전송)
  D : 에피소드 종료 + 폐기
  Q / Ctrl+C : 종료

⚠️ 이 스크립트는 실제 로봇/카메라(LeftarmRealEnvImp)가 있어야 동작한다. 로봇 없이
   learner+통신 로직만 검증하려면 online_learning/smoke_test_no_robot.py 를 실행.

실행:
  conda activate robodiff
  cd /home/rush/rush_diffusion_policy
  # 터미널1: python online_learning/online_learner.py
  # 터미널2:
  python online_learning/online_actor_env_runner.py
"""
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import dill
import hydra
import cv2
from multiprocessing.managers import SharedMemoryManager
from omegaconf import OmegaConf

from diffusion_policy.real_world.rush_real_env_leftarm_imp import LeftarmRealEnvImp
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, get_real_obs_dict)
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
          f"(learner demo={payload.get('num_demos','?')}, "
          f"missing={len(missing)}, unexpected={len(unexpected)})")
    return latest


def main():
    mailbox = FileMailbox(C.ONLINE_WORKDIR)
    device = torch.device(C.DEVICE if torch.cuda.is_available() else "cpu")

    # base policy 로드
    print(f"[Actor] base 체크포인트 로드: {C.BASE_CKPT}")
    payload = torch.load(open(C.BASE_CKPT, "rb"), pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.eval().to(device)
    policy.num_inference_steps = 16
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    weight_version = -1

    dt = 1.0 / 10.0
    steps_per_inference = 6
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps

    with SharedMemoryManager() as shm_manager:
        with LeftarmRealEnvImp(
            output_dir=C.ACTOR_OUTPUT_DIR,
            robot_ip="192.168.111.50",
            frequency=10,
            n_obs_steps=n_obs_steps,
            obs_image_resolution=obs_res,
            obs_float32=True,
            enable_multi_cam_vis=True,
            record_raw_video=False,
            shm_manager=shm_manager,
        ) as env:
            cv2.setNumThreads(1)
            time.sleep(2.0)
            print("[Actor] 준비 완료.")

            while True:
                # ── 에피소드 시작 전: 새 가중치 확인 & hot-swap ──
                weight_version = maybe_hotswap_weights(mailbox, policy, weight_version, device)

                try:
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    precise_wait(eval_t_start - 1.0/30, time_func=time.time)
                    print("[Actor] 에피소드 시작!")

                    iter_idx = 0
                    correction_active = False
                    keep = True
                    while True:
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']

                        with torch.no_grad():
                            obs_dict_np = get_real_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta)
                            obs_dict = dict_apply(obs_dict_np,
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            action = result['action'][0].detach().cpu().numpy()  # [H,9]

                        this_target_poses = np.zeros((len(action), 9), dtype=np.float64)
                        this_target_poses[:] = action[:]
                        action_timestamps = (np.arange(len(action), dtype=np.float64)) * dt + obs_timestamps[-1]
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + 0.01)
                        if np.sum(is_new) == 0:
                            this_target_poses = this_target_poses[[-1]]
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamps = np.array([eval_t_start + next_step_idx * dt])
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        # ── 키 입력: correction 토글 / 종료 ──
                        key_stroke = cv2.pollKey()
                        if key_stroke in (ord('c'), ord('C')):
                            correction_active = not correction_active
                            print(f"[Actor] correction={'ON' if correction_active else 'off'}")

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

                        precise_wait(t_cycle_end - 1.0/30)
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
                            out_path=os.path.join(C.ONLINE_WORKDIR, "last_episode.hdf5"))
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
