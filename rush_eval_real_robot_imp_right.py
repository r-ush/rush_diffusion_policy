#!/home/vision/miniconda3/envs/robodiff/bin/python

# 실행코드 (오른팔 only inference)
# python rush_eval_real_robot_imp_right.py --input data/outputs/logistics_abs_10/epoch=0800-train_loss=0.000.ckpt --output data/results --steps_per_inference 12 --frequency 10 --num_inference_steps 12

"""
Usage:
(robodiff)$ python rush_eval_real_robot_imp_right.py -i <ckpt_path> -o <save_dir>

학습 체크포인트 정보 (logistics_abs_10):
  - obs: image0 (3x240x320), robot_pose_L (3,), robot_quat_L (4,)
    * obs key 이름은 오른팔 데이터라도 shape_meta 라벨(_L)을 그대로 사용
  - action: shape (9,) = pos(3) + rot6d(6), abs 표현
  - 오른팔만 inference

Recording control:
  Click the opencv window (make sure it's in focus).
  Press "C" to toggle correction mode ON/OFF. Turn it ON right before you
    physically push/guide the arm to correct it, and OFF once you let go.
    This flag is saved per control-step in the episode's `stage` field
    (0=policy, 1=human correction), used later by
    data_process/rush_replay_buffer_to_correction_hdf5.py to relabel data.
  Press "S" to stop the episode and KEEP the recorded data.
  Press "D" to stop the episode and DISCARD the recorded data.
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

from diffusion_policy.real_world.rush_real_env_rightarm_imp import RightarmRealEnvImp
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution,
    get_real_obs_dict,
    get_real_relative_obs_dict,
    get_abs_action_from_relative)
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
              help="Robot IP (not used by RightarmRealEnv but kept for compatibility)")
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
    action_offset = 0
    delta_action = False

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

    # ===================== 2.5 Relative/abs pose repr =================
    # If not present in cfg, defaults to abs behavior.
    obs_pose_repr = OmegaConf.select(cfg, "task.pose_repr.obs_pose_repr", default=None)
    action_pose_repr = OmegaConf.select(cfg, "task.pose_repr.action_pose_repr", default=None)
    print(f"[DEBUG] cfg.task.pose_repr: {OmegaConf.select(cfg, 'task.pose_repr', default='NOT FOUND')}")
    print(f"[DEBUG] obs_pose_repr={obs_pose_repr}, action_pose_repr={action_pose_repr}")
    print(f"[DEBUG] action shape_meta: {cfg.task.shape_meta.action}")
    if obs_pose_repr is not None or action_pose_repr is not None:
        print(f"[INFO] pose_repr: obs={obs_pose_repr}, action={action_pose_repr}")

    # ===================== 3. Env setup ===========================
    dt = 1.0 / frequency
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)   # (width, height) = (320, 240)
    n_obs_steps = cfg.n_obs_steps   # 2

    print(f"[INFO] obs_res: {obs_res}")
    print(f"[INFO] n_obs_steps: {n_obs_steps}")
    print(f"[INFO] steps_per_inference: {steps_per_inference}")

    with SharedMemoryManager() as shm_manager:
        with RightarmRealEnvImp(
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
                if obs_pose_repr == 'relative':
                    obs_dict_np = get_real_relative_obs_dict(
                        env_obs=obs, shape_meta=cfg.task.shape_meta)
                else:
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
                    correction_active = False
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
                            if obs_pose_repr == 'relative':
                                obs_dict_np = get_real_relative_obs_dict(
                                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                            else:
                                obs_dict_np = get_real_obs_dict(
                                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                            obs_dict = dict_apply(obs_dict_np,
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            # action: [Horizon, 9]  (pos(3) + rot6d(6), abs)
                            action = result['action'][0].detach().cpu().numpy()
                            print(f"[DEBUG] AI Action[0] (pos): {action[0, :3]}")
                            print(f"[DEBUG] AI Action[-1] (pos): {action[-1, :3]}")

                            # ---- orientation 디버그 (오른팔) ----
                            # action[:, 3:9] = rot6d (abs). 컨트롤러가 이걸 rotvec->ZYX euler(deg)로 보냄.
                            try:
                                cmd_rotvec0 = rot6d_to_rotvec(action[0, 3:9])
                                cmd_euler0 = R.from_rotvec(cmd_rotvec0).as_euler('ZYX', degrees=True)
                                print(f"[DEBUG] AI Action[0] rot6d  : {np.round(action[0, 3:9], 4)}")
                                print(f"[DEBUG] AI Action[0] rotvec : {np.round(cmd_rotvec0, 4)}")
                                print(f"[DEBUG] AI Action[0] euler ZYX(deg): {np.round(cmd_euler0, 2)}")
                                # 현재 팔 자세 (obs quat -> euler) 와 비교
                                curr_quat = np.asarray(obs['robot_quat_L'])[-1]  # [x,y,z,w]
                                curr_euler = R.from_quat(curr_quat).as_euler('ZYX', degrees=True)
                                print(f"[DEBUG] Current obs quat_L  : {np.round(curr_quat, 4)}")
                                print(f"[DEBUG] Current obs euler ZYX(deg): {np.round(curr_euler, 2)}")
                                print(f"[DEBUG] euler delta (cmd-curr): {np.round(cmd_euler0 - curr_euler, 2)}")
                            except Exception as _e:
                                print(f"[DEBUG] orientation debug skipped: {_e}")
                            # -------------------------------------

                            print(f"[INFO] Inference latency: {time.time() - t_inf:.4f}s")

                        # convert policy action to env actions
                        if delta_action:   # False
                            assert len(action) == 1
                            this_target_poses = np.zeros((len(action), action.shape[-1]), dtype=np.float64)
                            this_target_poses[:, :action.shape[-1]] = action
                        else:
                            this_target_poses = np.zeros((len(action), action.shape[-1]), dtype=np.float64)
                            this_target_poses[:, :action.shape[-1]] = action

                        # --- relative action -> abs target pose ---
                        # Keep env API unchanged: always execute absolute (pos+rot6d).
                        if action_pose_repr == 'relative':
                            try:
                                # Uses latest robot pose from env obs as reference.
                                print("Generated Action (Relative):", this_target_poses[0])
                                print(f"[DEBUG] Current robot abs pose_L: {obs['robot_pose_L'][-1]}")
                                print(f"[DEBUG] Current robot abs quat_L: {obs['robot_quat_L'][-1]}")
                                # Ensure we clip/check for absurdly large movements here
                                if np.any(np.abs(this_target_poses[..., :3]) > 0.5):
                                    print("WARNING: Unusually large relative translation detected!")

                                this_target_poses = get_abs_action_from_relative(
                                    action=this_target_poses,
                                    env_obs=obs
                                )
                                print("Converted Absolute Target Pose:", this_target_poses[0])
                            except Exception as e:
                                raise RuntimeError(
                                    "Failed to convert relative action to absolute. "
                                    "Check ckpt pose_repr/action definition and shape_meta. "
                                    f"Original error: {e}"
                                )

                        # deal with timing
                        # the same step actions are always the target for
                        # timing: already passed actions are removed
                        action_timestamps = (
                            np.arange(len(action), dtype=np.float64) + action_offset
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
                            this_target_poses = this_target_poses[is_new][:steps_per_inference]
                            action_timestamps  = action_timestamps[is_new][:steps_per_inference]

                        # --- keypress check (correction 토글 / 종료) ---
                        # C: correction 모드 토글. ON일 때 팔을 물리적으로 밀어 교정하면
                        #    해당 스텝이 stage=1로 기록되어, 이후
                        #    data_process/rush_replay_buffer_to_correction_hdf5.py 가
                        #    achieved-pose로 relabel 학습 데이터를 만든다.
                        key_stroke = cv2.pollKey()
                        if key_stroke == ord('c') or key_stroke == ord('C'):
                            correction_active = not correction_active
                            print(f"[INFO] Correction mode: "
                                  f"{'ON (human correcting)' if correction_active else 'OFF'}")

                        # --- execute ---
                        stages = np.full(len(this_target_poses),
                                         1 if correction_active else 0, dtype=np.int64)
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps,
                            stages=stages)
                        print(f"[INFO] Submitted {len(this_target_poses)} action steps. "
                              f"correction={correction_active}")

                        if key_stroke == ord('s') or key_stroke == ord('S'):
                            env.end_episode()
                            print("[INFO] Episode stopped by user (kept).")
                            break
                        if key_stroke == ord('d') or key_stroke == ord('D'):
                            env.drop_episode()
                            print("[INFO] Episode stopped by user (DISCARDED).")
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
