#!/usr/bin/env python3
"""
Dataset-based Inference Script for Diffusion Policy (Pose Only)

학습된 checkpoint를 이용해서, HDF5 dataset의 observation을 순서대로 던져주며 inference를 수행합니다.
예측된 action sequence와 GT action을 비교하고, 결과를 저장합니다.

사용법:
  cd /home/rush/diffusion_policy
  source venv_dp/bin/activate
  python bae_infer_from_dataset.py \
    --checkpoint /media/rush/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7/260429_0102/outputs/epoch=0800-train_loss=0.001.ckpt \
    --dataset /media/rush/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7/260429_0102/diffusion_data_leftarm_pose_only.hdf5 \
    --demo_idx 0 \
    --output_dir ./inference_results
"""

import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import argparse
import numpy as np
import torch
import dill
import h5py
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import hydra

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply

OmegaConf.register_new_resolver("eval", eval, replace=True)


# ─────────────────────────────────────────────
# 1. Checkpoint 로드 유틸
# ─────────────────────────────────────────────
def load_policy_from_checkpoint(ckpt_path: str, device: str = "cuda"):
    """
    checkpoint에서 policy(EMA 우선)와 cfg를 꺼내 반환합니다.
    """
    ckpt_path = pathlib.Path(ckpt_path)
    print(f"[INFO] Loading checkpoint: {ckpt_path}")

    payload = torch.load(str(ckpt_path), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]

    # Workspace 인스턴스 생성 & weight 로드
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # EMA 모델 우선 사용
    if cfg.training.use_ema and workspace.ema_model is not None:
        policy: BaseImagePolicy = workspace.ema_model
        print("[INFO] Using EMA model for inference.")
    else:
        policy: BaseImagePolicy = workspace.model
        print("[INFO] Using base model for inference.")

    policy.eval().to(torch.device(device))

    # DDIM inference steps
    policy.num_inference_steps = 8   # cfg에서 8로 설정되어 있음
    # 실행할 action 수: horizon - n_obs_steps + 1
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    print(f"[INFO] Policy horizon={policy.horizon}, n_obs_steps={policy.n_obs_steps}, "
          f"n_action_steps={policy.n_action_steps}, num_inference_steps={policy.num_inference_steps}")
    return policy, cfg


# ─────────────────────────────────────────────
# 2. HDF5 Demo 로드 유틸
# ─────────────────────────────────────────────
def load_demo(hdf5_path: str, demo_idx: int):
    """
    HDF5에서 demo를 불러와 numpy dict로 반환합니다.

    Returns:
        obs_dict: {
            'image0':       (T, 240, 320, 3)  uint8
            'robot_pose_L': (T, 3)            float64
            'robot_quat_L': (T, 4)            float64
        }
        gt_actions: (T, 9)  float64  – GT action (robot_pose_L + rot6d)
        T: 총 timestep 수
    """
    with h5py.File(hdf5_path, "r") as f:
        all_demos = sorted(f["data"].keys(),
                           key=lambda x: int(x.replace("demo_", "")))
        if demo_idx >= len(all_demos):
            raise ValueError(f"demo_idx={demo_idx} >= 총 demo 수 {len(all_demos)}")
        demo_key = all_demos[demo_idx]
        print(f"[INFO] Loading {demo_key}  (총 {len(all_demos)}개 demo)")

        demo = f[f"data/{demo_key}"]
        obs_dict = {
            "image0":       np.array(demo["obs/image0"]),       # (T,H,W,3) uint8
            "robot_pose_L": np.array(demo["obs/robot_pose_L"]), # (T,3)
            "robot_quat_L": np.array(demo["obs/robot_quat_L"]), # (T,4)
        }
        gt_actions = np.array(demo["actions"])  # (T,9)

    T = obs_dict["image0"].shape[0]
    print(f"[INFO] Demo timesteps: {T}")
    return obs_dict, gt_actions, T


# ─────────────────────────────────────────────
# 3. obs 전처리 → policy 입력 형태로 변환
# ─────────────────────────────────────────────
def make_obs_tensor(obs_dict_np, t, n_obs_steps, device):
    """
    timestep t 기준으로 n_obs_steps 만큼의 window를 만들어
    policy.predict_action()에 넣을 수 있는 텐서 dict를 반환합니다.

    image: (T,H,W,C) uint8  →  (1, n_obs_steps, C, H, W) float32 [0,1]
    low_dim: (T, D)          →  (1, n_obs_steps, D) float32
    """
    T = obs_dict_np["image0"].shape[0]

    # 시작 인덱스 (부족하면 첫 프레임 패딩)
    indices = [max(0, t - n_obs_steps + 1 + i) for i in range(n_obs_steps)]

    imgs = obs_dict_np["image0"][indices]          # (n_obs, H, W, 3) uint8
    imgs = imgs.astype(np.float32) / 255.0         # [0,1]
    imgs = np.transpose(imgs, (0, 3, 1, 2))        # (n_obs, 3, H, W)
    imgs = imgs[np.newaxis]                         # (1, n_obs, 3, H, W)

    pose = obs_dict_np["robot_pose_L"][indices].astype(np.float32)  # (n_obs, 3)
    quat = obs_dict_np["robot_quat_L"][indices].astype(np.float32)  # (n_obs, 4)
    pose = pose[np.newaxis]  # (1, n_obs, 3)
    quat = quat[np.newaxis]  # (1, n_obs, 4)

    obs_tensor = {
        "image0":       torch.from_numpy(imgs).to(device),
        "robot_pose_L": torch.from_numpy(pose).to(device),
        "robot_quat_L": torch.from_numpy(quat).to(device),
    }
    return obs_tensor


# ─────────────────────────────────────────────
# 4. 결과 시각화 & 저장
# ─────────────────────────────────────────────
def save_results(pred_actions_all, gt_actions_all, output_dir, demo_idx):
    """
    예측 action과 GT action을 비교하는 그래프를 저장합니다.
    pred_actions_all: list of (n_action_steps, 9)
    gt_actions_all:   (T, 9)
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 각 inference step에서 첫번째 예측만 사용해 trajectory 재구성
    pred_traj = np.array([p[0] for p in pred_actions_all])  # (num_steps, 9)
    gt_traj   = gt_actions_all[:len(pred_traj)]              # (num_steps, 9)

    action_labels_xyz  = ["x", "y", "z"]
    action_labels_rot6d = [f"rot6d_{i}" for i in range(6)]
    all_labels = action_labels_xyz + action_labels_rot6d

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for i in range(9):
        ax = axes[i]
        ax.plot(gt_traj[:, i],   label="GT",   color="steelblue",  linewidth=2)
        ax.plot(pred_traj[:, i], label="Pred", color="orangered",  linewidth=2, linestyle="--")
        ax.set_title(f"Action dim {i}: {all_labels[i]}", fontsize=11)
        ax.set_xlabel("Timestep")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    mse_per_dim = np.mean((pred_traj - gt_traj) ** 2, axis=0)
    mse_total   = float(np.mean(mse_per_dim))
    fig.suptitle(f"Demo {demo_idx} — Inference vs GT  |  MSE={mse_total:.6f}", fontsize=13)
    plt.tight_layout()

    save_path = output_dir / f"inference_demo{demo_idx}.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Plot → {save_path}")

    # 수치 저장
    np_path = output_dir / f"inference_demo{demo_idx}.npz"
    np.savez(str(np_path), pred=pred_traj, gt=gt_traj)
    print(f"[SAVED] NumPy → {np_path}")
    print(f"[RESULT] MSE per dim: {mse_per_dim}")
    print(f"[RESULT] MSE total  : {mse_total:.6f}")

    return mse_total


# ─────────────────────────────────────────────
# 5. 메인 Inference 루프
# ─────────────────────────────────────────────
def run_inference(policy, cfg, obs_dict_np, gt_actions, device, step_every=1):
    """
    sliding window 방식으로 매 timestep마다 inference를 수행합니다.

    step_every: 몇 step마다 inference할지 (기본 1 = 매 step)
    """
    n_obs_steps = cfg.n_obs_steps  # 2
    T = obs_dict_np["image0"].shape[0]

    pred_actions_list = []

    print(f"\n[INFO] Running inference over {T} timesteps (step_every={step_every})...")

    with torch.no_grad():
        policy.reset()
        for t in range(T):
            if t % step_every != 0:
                continue

            obs_tensor = make_obs_tensor(obs_dict_np, t, n_obs_steps, device)

            result = policy.predict_action(obs_tensor)
            # action: (1, n_action_steps, 9) → (n_action_steps, 9)
            action_pred = result["action"][0].detach().cpu().numpy()

            pred_actions_list.append(action_pred)

            # 진행 상황 출력
            if t % 10 == 0 or t == T - 1:
                gt_a = gt_actions[t]
                pr_a = action_pred[0]
                print(f"  t={t:3d}  GT=[{gt_a[0]:.4f},{gt_a[1]:.4f},{gt_a[2]:.4f},...]  "
                      f"Pred=[{pr_a[0]:.4f},{pr_a[1]:.4f},{pr_a[2]:.4f},...]")

    return pred_actions_list


# ─────────────────────────────────────────────
# 6. Entry point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Offline inference from HDF5 dataset")
    parser.add_argument("--checkpoint", "-c", required=True,
                        help="Path to .ckpt file")
    parser.add_argument("--dataset", "-d", required=True,
                        help="Path to HDF5 dataset file")
    parser.add_argument("--demo_idx", "-n", type=int, default=0,
                        help="Which demo to use (0-indexed, default=0)")
    parser.add_argument("--output_dir", "-o", default="./inference_results",
                        help="Directory to save results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (default: cuda if available)")
    parser.add_argument("--step_every", type=int, default=1,
                        help="Inference every N steps (default=1, every step)")
    args = parser.parse_args()

    print(f"[INFO] Device: {args.device}")

    # 1) Policy 로드
    policy, cfg = load_policy_from_checkpoint(args.checkpoint, args.device)

    # 2) 데이터 로드
    obs_dict_np, gt_actions, T = load_demo(args.dataset, args.demo_idx)

    # 3) Inference
    pred_actions_list = run_inference(
        policy, cfg, obs_dict_np, gt_actions, args.device,
        step_every=args.step_every
    )

    # 4) 결과 저장
    mse = save_results(pred_actions_list, gt_actions, args.output_dir, args.demo_idx)
    print(f"\n[DONE] Inference complete. Total MSE={mse:.6f}")


if __name__ == "__main__":
    main()
