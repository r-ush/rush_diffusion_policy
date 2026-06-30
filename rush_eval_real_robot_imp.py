#!/home/vision/miniconda3/envs/robodiff/bin/python

# 실행코드 (왼팔 only inference)
# python rush_eval_real_robot.py --input data/outputs/260429_0102_wo_imp/epoch=0800-train_loss=0.001.ckpt --output data/results

"""
Usage:
(robodiff)$ python rush_eval_real_robot.py -i <ckpt_path> -o <save_dir>

학습 체크포인트 정보:
  - obs: image0 (3x240x320), robot_pose_L (3,), robot_quat_L (4,)
  - action: shape (9,) = pos_L(3) + rot6d_L(6), abs 표현
  - 왼팔만 inference, 오른팔은 현재 위치 유지

Recording control:
  Click the opencv window (make sure it's in focus).
  Press "S" to stop evaluation.
  Press "Q" or Ctrl+C to exit program.
"""

# %%
# ============================================================
# huggingface_hub 버전 충돌 문제 해결 (local v1.x vs conda v0.11.1)
# 1. 임시로 .local 경로를 sys.path에서 제외
# 2. huggingface_hub를 conda 환경에서 먼저 임포트
# 3. 다시 sys.path 복구하여 torch(최신 쿠다지원) 등 다른 패키지는 정상 로드되게 함
# ============================================================
import sys
local_paths = [p for p in sys.path if '.local' in p]
sys.path = [p for p in sys.path if '.local' not in p]

import huggingface_hub

# 다시 원래대로 복구 (우선순위 유지)
sys.path = local_paths + sys.path

import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

from diffusion_policy.real_world.rush_real_env_leftarm_imp import LeftarmRealEnvImp
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution,
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

OmegaConf.register_new_resolver("eval", eval, replace=True)


def rot6d_to_rotvec(rot6d: np.ndarray) -> np.ndarray:
    """rotation 6d representation -> rotation vector"""
    a1 = rot6d[:3]
    a2 = rot6d[3:]
    b1 = a1 / np.linalg.norm(a1)
    a2_proj = np.dot(b1, a2) * b1
    b2 = a2 - a2_proj
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    R_mat = np.stack((b1, b2, b3), axis=1)
    return R.from_matrix(R_mat).as_rotvec()


@click.command()
@click.option('--input',  '-i', required=True,  help='Path to checkpoint (.ckpt)')
@click.option('--output', '-o', required=True,  help='Directory to save recording')
@click.option('--robot_ip', '-ri', default="192.168.111.50",
              help="Robot IP (not used by LeftarmRealEnv but kept for compatibility)")
@click.option('--vis_camera_idx', default=0, type=int,
              help="Which RealSense camera to visualize.")
@click.option('--steps_per_inference', '-si', default=6, type=int,
              help="Number of action steps to execute per inference (action horizon).")
@click.option('--max_duration', '-md', default=60, type=float,
              help='Max duration for each episode in seconds.')
@click.option('--frequency', '-f', default=10, type=float,
              help="Control frequency in Hz.")
@click.option('--num_inference_steps', '-ni', default=16, type=int,
              help="DDIM denoising steps (more → slower but potentially better).")
def main(input, output, robot_ip,
         vis_camera_idx, steps_per_inference, max_duration,
         frequency, num_inference_steps):

    # ===================== 1. Load checkpoint =====================
    ckpt_path = input
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # ===================== 2. Policy setup ========================
    if 'diffusion' not in cfg.name:
        raise RuntimeError(f"Unsupported policy type: {cfg.name}")

    policy: BaseImagePolicy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device('cuda')
    policy.eval().to(device)

    # DDIM inference steps
    policy.num_inference_steps = num_inference_steps
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    print(f"[INFO] Policy: {cfg.name}")
    print(f"[INFO] horizon={policy.horizon}, n_obs_steps={policy.n_obs_steps}, "
          f"n_action_steps={policy.n_action_steps}")

    # ===================== 3. Env setup ===========================
    dt = 1.0 / frequency
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)   # (width, height) = (320, 240)
    n_obs_steps = cfg.n_obs_steps   # 2

    print(f"[INFO] obs_res: {obs_res}")
    print(f"[INFO] n_obs_steps: {n_obs_steps}")
    print(f"[INFO] steps_per_inference: {steps_per_inference}")

    with SharedMemoryManager() as shm_manager:
        with LeftarmRealEnvImp(
            output_dir=output,
            robot_ip=robot_ip,
            frequency=frequency,
            n_obs_steps=n_obs_steps,
            obs_image_resolution=obs_res,         # (320, 240)
            obs_float32=True,
            enable_multi_cam_vis=True,
            record_raw_video=False,
            thread_per_video=3,
            video_crf=21,
            shm_manager=shm_manager
        ) as env:
            cv2.setNumThreads(1)

            print("[INFO] Waiting for sensors to warm up...")
            time.sleep(2.0)

            # =================== 4. Warmup inference ====================
            print("[INFO] Warming up policy inference...")
            obs = env.get_obs()

            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                obs_dict = dict_apply(obs_dict_np,
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action'][0].detach().cpu().numpy()
                assert action.shape[-1] == 9, \
                    f"Expected action dim=9, got {action.shape[-1]}"
                del result

            print("[INFO] Ready! Press 'S' to stop episode, Ctrl+C to quit.")

            # =================== 5. Control loop =======================
            while True:
                try:
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start      = time.monotonic() + start_delay

                    env.start_episode(eval_t_start)
                    frame_latency = 1.0 / 30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("[INFO] Episode started!")

                    iter_idx = 0
                    while True:
                        # --- timing ---
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # --- get obs ---
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f"[INFO] Obs latency: {time.time() - obs_timestamps[-1]:.4f}s")

                        # --- inference ---
                        with torch.no_grad():
                            t_inf = time.time()
                            obs_dict_np = get_real_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta)
                            obs_dict = dict_apply(obs_dict_np,
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            # action: [Horizon, 9]  (pos_L(3) + rot6d_L(6), abs)
                            action = result['action'][0].detach().cpu().numpy()
                            print(f"[DEBUG] AI Action[0] (pos): {action[0, :3]}")
                            print(f"[DEBUG] AI Action[-1] (pos): {action[-1, :3]}")
                            print(f"[INFO] Inference latency: {time.time() - t_inf:.4f}s")

                        # --- policy action (9,) ---
                        this_target_poses = np.zeros(
                            (len(action), 9), dtype=np.float64)
                        this_target_poses[:] = action[:]

                        # --- timing filter: 이미 지난 action 제거 ---
                        action_timestamps = (
                            np.arange(len(action), dtype=np.float64)
                        ) * dt + obs_timestamps[-1]
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + 0.01)

                        if np.sum(is_new) == 0:
                            # 전부 지나버린 경우: 마지막 action이라도 실행
                            this_target_poses = this_target_poses[[-1]]
                            next_step_idx = int(
                                np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamps = np.array(
                                [eval_t_start + next_step_idx * dt])
                            print("[WARN] Over budget! Executing last action.")
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps  = action_timestamps[is_new]

                        # --- execute ---
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps)
                        print(f"[INFO] Submitted {len(this_target_poses)} action steps.")

                        # --- keypress check ---
                        key_stroke = cv2.pollKey()
                        if key_stroke == ord('s') or key_stroke == ord('S'):
                            env.end_episode()
                            print("[INFO] Episode stopped by user.")
                            break

                        # --- timeout check ---
                        if time.monotonic() - t_start > max_duration:
                            env.end_episode()
                            print("[INFO] Episode terminated by timeout.")
                            break

                        # --- wait for next cycle ---
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("\n[INFO] Interrupted! Stopping robot.")
                    env.end_episode()
                    break

                # 에피소드 끝나면 다음 에피소드 대기
                print("[INFO] Press Enter to start next episode, or Ctrl+C to quit.")
                try:
                    import sys
                    sys.stdin.readline()   # 엔터 대기 (builtin input() 과 click 'input' 파라미터 충돌 방지)
                except (KeyboardInterrupt, EOFError):
                    break

    print("[INFO] Done.")


# %%
if __name__ == '__main__':
    main()
