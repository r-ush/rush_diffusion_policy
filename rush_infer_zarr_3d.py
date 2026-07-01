#!/usr/bin/env python3
"""
Zarr 에피소드 데이터셋 기반 Inference 3D 시각화

학습된 checkpoint를 이용해 raw zarr 에피소드의 관측을 순서대로 넣고,
모델이 예측하는 미래 trajectory를 3D로 보여줍니다.

3D plot 구성:
  - 파란 실선   : GT 현재 EE 위치 (ee_pose_se3 → FK 결과)
  - 주황 점선   : GT 목표 위치 (command_pose_se3, 학습 label)
  - 컬러 마커   : inference 시작점 (에피소드 초→말: 파랑→빨강)
  - 컬러 실선   : 해당 시점에서 모델이 예측한 미래 trajectory

사용법:
  conda activate robodiff
  cd ~/rush_diffusion_policy
  python rush_infer_zarr_3d.py \\
    --checkpoint data/outputs/logistic_box_unet_abs/checkpoints/epoch=0400-train_loss=0.002.ckpt \\
    --dataset_root /home/rush/Desktop/Datasets/20260630_195919 \\
    --episode_idx 0 \\
    --infer_every 20 \\
    --output_dir ./inference_results_3d
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
import zarr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go
from PIL import Image
from omegaconf import OmegaConf
import hydra

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply

OmegaConf.register_new_resolver("eval", eval, replace=True)

# ─────────────────────────────────────────────
# 설정 상수
# ─────────────────────────────────────────────
STRIDE        = 3           # 30Hz → 10Hz
IMG_OUT_SIZE  = (320, 240)  # (W, H)
CAMERA_NAME   = "camera_0_D405"


# ══════════════════════════════════════════════
# 1. Checkpoint 로드
# ══════════════════════════════════════════════
def load_policy(ckpt_path: str, device: str):
    payload = torch.load(ckpt_path, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    if cfg.training.use_ema and workspace.ema_model is not None:
        policy: BaseImagePolicy = workspace.ema_model
        print("[INFO] EMA 모델 사용")
    else:
        policy: BaseImagePolicy = workspace.model
        print("[INFO] 기본 모델 사용")

    policy.eval().to(torch.device(device))
    policy.num_inference_steps = 8

    print(f"[INFO] horizon={policy.horizon}, n_obs_steps={policy.n_obs_steps}, "
          f"n_action_steps={policy.n_action_steps}")
    return policy, cfg


# ══════════════════════════════════════════════
# 2. Zarr 에피소드 로드 & 전처리
# ══════════════════════════════════════════════
def se3_to_pos(se3_matrices):
    """SE3 (N,4,4) → 위치 (N,3) [m]"""
    return se3_matrices[:, :3, 3].copy()


def se3_to_quat(se3_matrices):
    """SE3 (N,4,4) → 쿼터니언 (N,4) [x,y,z,w]"""
    rotmats = se3_matrices[:, :3, :3]
    quat = R.from_matrix(rotmats).as_quat()
    quat[quat[:, 3] < 0] *= -1
    return quat


def resize_images(imgs_nhwc, out_wh=(320, 240)):
    out_w, out_h = out_wh
    out = np.empty((len(imgs_nhwc), out_h, out_w, imgs_nhwc.shape[3]), dtype=np.uint8)
    for i, img in enumerate(imgs_nhwc):
        out[i] = np.array(Image.fromarray(img).resize(out_wh, Image.LANCZOS))
    return out


def load_episode(dataset_root: str, episode_idx: int):
    """
    zarr 에피소드 폴더에서 데이터를 로드해 전처리 후 반환.

    반환:
      ee_pos_full   : (N_full, 3)  - 전체 raw EE 위치 (stride 전, 3D 참조용)
      cmd_pos_full  : (N_full, 3)  - 전체 raw command 위치 (stride 전)
      obs_dict      : {'image0', 'robot_pose_L', 'robot_quat_L'}  (T=downsampled)
      act_gt_pos    : (T, 3)  - GT action 위치 (command_pose → 9D 중 pos만)
      T             : downsampled 총 timestep
    """
    ep_dirs = sorted([
        d for d in os.listdir(dataset_root)
        if d.startswith("episode_") and
        os.path.isdir(os.path.join(dataset_root, d))
    ])
    if episode_idx >= len(ep_dirs):
        raise ValueError(f"episode_idx={episode_idx} >= 총 {len(ep_dirs)}개 에피소드")

    ep_path = os.path.join(dataset_root, ep_dirs[episode_idx])
    print(f"[INFO] 에피소드 로드: {ep_dirs[episode_idx]}")

    # --- raw zarr 로드 ---
    ee_se3  = zarr.open(os.path.join(ep_path, "robot/ee_pose_se3.zarr"), "r")[:]
    cmd_se3 = zarr.open(os.path.join(ep_path, "robot/command_pose_se3.zarr"), "r")[:]
    rgb     = zarr.open(os.path.join(ep_path, f"{CAMERA_NAME}/rgb.zarr"), "r")[:]

    N = min(len(ee_se3), len(rgb))     # 721

    # 전체 trajectory (시각화용, stride 전)
    ee_pos_full  = se3_to_pos(ee_se3[:N])
    cmd_pos_full = se3_to_pos(cmd_se3[:N])

    # --- downsampled (학습 포맷과 동일) ---
    # obs: [0, N-1), action: [1, N)   (길이 N-1)
    ee_obs  = ee_se3[:N-1:STRIDE]
    cmd_act = cmd_se3[1:N:STRIDE]
    rgb_obs = rgb[:N-1:STRIDE]

    T = min(len(ee_obs), len(cmd_act), len(rgb_obs))
    ee_obs  = ee_obs[:T]
    cmd_act = cmd_act[:T]
    rgb_obs = rgb_obs[:T]

    # obs
    ee_pos  = se3_to_pos(ee_obs).astype(np.float32)
    ee_quat = se3_to_quat(ee_obs).astype(np.float32)
    images  = resize_images(rgb_obs, IMG_OUT_SIZE)

    # GT action 위치 (command의 translation)
    act_gt_pos = se3_to_pos(cmd_act).astype(np.float32)

    obs_dict = {
        "image0":       images,   # (T, H, W, 3) uint8
        "robot_pose_L": ee_pos,   # (T, 3)
        "robot_quat_L": ee_quat,  # (T, 4)
    }

    print(f"[INFO] Downsampled: {T} steps  |  Full trajectory: {N} frames")
    return ee_pos_full, cmd_pos_full, obs_dict, act_gt_pos, T


# ══════════════════════════════════════════════
# 3. Inference
# ══════════════════════════════════════════════
def make_obs_tensor(obs_dict, t, n_obs_steps, device):
    """t 시점 기준 n_obs_steps 윈도우 → policy 입력 텐서"""
    T = obs_dict["image0"].shape[0]
    idx = [max(0, t - n_obs_steps + 1 + i) for i in range(n_obs_steps)]

    imgs = obs_dict["image0"][idx].astype(np.float32) / 255.0
    imgs = np.transpose(imgs, (0, 3, 1, 2))[np.newaxis]   # (1, n_obs, 3, H, W)

    pose = obs_dict["robot_pose_L"][idx].astype(np.float32)[np.newaxis]   # (1, n_obs, 3)
    quat = obs_dict["robot_quat_L"][idx].astype(np.float32)[np.newaxis]   # (1, n_obs, 4)

    return {
        "image0":       torch.from_numpy(imgs).to(device),
        "robot_pose_L": torch.from_numpy(pose).to(device),
        "robot_quat_L": torch.from_numpy(quat).to(device),
    }


def run_inference_at_points(policy, cfg, obs_dict, infer_every, device):
    """
    infer_every 스텝마다 inference를 수행.

    반환:
      results: list of dict
        - 't'        : inference 시작 timestep (downsampled)
        - 'obs_pos'  : 해당 t의 실제 EE 위치 (3,)
        - 'pred_pos' : 예측된 미래 위치 (n_action_steps, 3)
    """
    n_obs_steps  = cfg.n_obs_steps
    n_act_steps  = policy.n_action_steps
    T = obs_dict["image0"].shape[0]
    results = []

    print(f"[INFO] Inference: 총 {T} steps, {infer_every} step마다 실행")
    print(f"       → 예측 horizon={n_act_steps} steps")

    with torch.no_grad():
        policy.reset()
        for t in range(0, T, infer_every):
            obs_tensor = make_obs_tensor(obs_dict, t, n_obs_steps, device)
            result = policy.predict_action(obs_tensor)
            # (1, n_action_steps, 9) → (n_action_steps, 9)
            action_pred = result["action"][0].detach().cpu().numpy()
            pred_pos = action_pred[:, :3]   # position 성분만 추출

            results.append({
                "t":        t,
                "obs_pos":  obs_dict["robot_pose_L"][t],
                "pred_pos": pred_pos,
            })

    print(f"[INFO] 총 {len(results)}회 inference 완료")
    return results


# ══════════════════════════════════════════════
# 4. 3D 시각화
# ══════════════════════════════════════════════
def plot_3d(ee_pos_full, cmd_pos_full, obs_dict, act_gt_pos,
            infer_results, episode_idx, output_dir):
    """
    3D 경로 + 2D XYZ 시계열 subplot 저장.

    색상 규칙:
      파란 실선  = GT 현재 EE (ee_pose FK)
      주황 점선  = GT 목표 (command_pose, 학습 label)
      컬러(cool→warm) = inference 시작점(마커) + 예측 경로(선)
                        에피소드 초반=파랑, 후반=빨강
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    T_full  = len(ee_pos_full)
    T_down  = len(obs_dict["robot_pose_L"])
    n_infer = len(infer_results)

    # inference 시작점별 색상 (coolwarm colormap)
    cmap   = cm.get_cmap("coolwarm", n_infer)
    colors = [cmap(i / max(n_infer - 1, 1)) for i in range(n_infer)]

    # ── Figure 1: 3D trajectory ────────────────────────────────────
    fig3d = plt.figure(figsize=(14, 10))
    ax3d  = fig3d.add_subplot(111, projection="3d")

    # GT current EE (full resolution, stride 전 원본)
    ax3d.plot(ee_pos_full[:, 0], ee_pos_full[:, 1], ee_pos_full[:, 2],
              color="steelblue", linewidth=2.0, label="GT EE (current)", alpha=0.8)

    # GT command/desired (full resolution)
    ax3d.plot(cmd_pos_full[:, 0], cmd_pos_full[:, 1], cmd_pos_full[:, 2],
              color="darkorange", linewidth=1.5, linestyle="--",
              label="GT Command (desired)", alpha=0.8)

    # Inference 결과
    for i, res in enumerate(infer_results):
        t       = res["t"]
        obs_p   = res["obs_pos"]
        pred_p  = res["pred_pos"]   # (n_action_steps, 3)
        c       = colors[i]
        t_sec   = t * STRIDE / 30.0  # 실제 시각(초)

        # 시작점 마커
        ax3d.scatter(obs_p[0], obs_p[1], obs_p[2],
                     color=c, s=80, zorder=5, marker="o",
                     edgecolors="black", linewidths=0.5)

        # 예측 trajectory (시작점에서 첫 pred까지 연결 포함)
        traj_x = np.concatenate([[obs_p[0]], pred_p[:, 0]])
        traj_y = np.concatenate([[obs_p[1]], pred_p[:, 1]])
        traj_z = np.concatenate([[obs_p[2]], pred_p[:, 2]])
        ax3d.plot(traj_x, traj_y, traj_z,
                  color=c, linewidth=1.2, alpha=0.85,
                  label=f"t={t}({t_sec:.1f}s)" if i < 6 else "_")

        # 끝점 작은 삼각형
        ax3d.scatter(pred_p[-1, 0], pred_p[-1, 1], pred_p[-1, 2],
                     color=c, s=40, marker="^", zorder=4)

    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.set_title(f"Episode {episode_idx}  |  Inference 3D Trajectory\n"
                   f"● = inference start,  △ = prediction end,  색상: 초반(파랑)→후반(빨강)",
                   fontsize=12)
    ax3d.legend(loc="upper left", fontsize=8, ncol=2)

    # colorbar (시간 축)
    sm = plt.cm.ScalarMappable(cmap="coolwarm",
                               norm=plt.Normalize(vmin=0,
                                                  vmax=(T_down - 1) * STRIDE / 30.0))
    sm.set_array([])
    cbar = fig3d.colorbar(sm, ax=ax3d, shrink=0.5, pad=0.1)
    cbar.set_label("Inference 시작 시각 [s]")

    path_3d = output_dir / f"episode{episode_idx:03d}_3d.png"
    fig3d.savefig(str(path_3d), dpi=150, bbox_inches="tight")
    plt.close(fig3d)
    print(f"[SAVED] 3D plot → {path_3d}")

    # ── Figure 2: XYZ 시계열 + Inference 겹쳐보기 ─────────────────
    fig2, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    axis_labels = ["X", "Y", "Z"]

    # downsampled GT 시간 축
    t_down = np.arange(T_down) * STRIDE / 30.0   # [초]
    t_full = np.arange(T_full) / 30.0            # [초]

    for dim, (ax, lbl) in enumerate(zip(axes, axis_labels)):
        # GT EE (full)
        ax.plot(t_full, ee_pos_full[:, dim],
                color="steelblue", lw=1.5, label="GT EE (current)", alpha=0.8)
        # GT command (full)
        ax.plot(t_full, cmd_pos_full[:, dim],
                color="darkorange", lw=1.2, ls="--",
                label="GT Command (desired)", alpha=0.8)

        # Inference 예측
        for i, res in enumerate(infer_results):
            t_start = res["t"] * STRIDE / 30.0
            pred_p  = res["pred_pos"]
            n_p     = len(pred_p)
            # 예측 시간 축: 시작점에서 STRIDE/30s 간격으로 n_action_steps 앞
            t_pred  = t_start + np.arange(1, n_p + 1) * STRIDE / 30.0
            ax.plot(t_pred, pred_p[:, dim],
                    color=colors[i], lw=1.2, alpha=0.8,
                    label=f"Pred t={res['t']}" if dim == 0 and i < 5 else "_")
            ax.axvline(t_start, color=colors[i], lw=0.5, alpha=0.4, ls=":")

        ax.set_ylabel(f"{lbl} [m]")
        ax.grid(True, alpha=0.3)
        if dim == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=3)

    axes[-1].set_xlabel("시각 [s]")
    fig2.suptitle(f"Episode {episode_idx}  |  XYZ 시계열  (점선=inference 시작)",
                  fontsize=12)
    plt.tight_layout()

    path_xyz = output_dir / f"episode{episode_idx:03d}_xyz.png"
    fig2.savefig(str(path_xyz), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[SAVED] XYZ plot → {path_xyz}")

    # ── Figure 3: XY 평면 투영 (overhead) ─────────────────────────
    fig_xy, ax_xy = plt.subplots(figsize=(9, 8))

    ax_xy.plot(ee_pos_full[:, 0], ee_pos_full[:, 1],
               color="steelblue", lw=2, label="GT EE (current)", alpha=0.8)
    ax_xy.plot(cmd_pos_full[:, 0], cmd_pos_full[:, 1],
               color="darkorange", lw=1.5, ls="--",
               label="GT Command (desired)", alpha=0.8)

    for i, res in enumerate(infer_results):
        obs_p  = res["obs_pos"]
        pred_p = res["pred_pos"]
        c = colors[i]
        t_sec = res["t"] * STRIDE / 30.0

        ax_xy.scatter(obs_p[0], obs_p[1], color=c, s=100,
                      zorder=5, marker="o", edgecolors="black", lw=0.5)
        traj_x = np.concatenate([[obs_p[0]], pred_p[:, 0]])
        traj_y = np.concatenate([[obs_p[1]], pred_p[:, 1]])
        ax_xy.plot(traj_x, traj_y, color=c, lw=1.5, alpha=0.85,
                   label=f"t={res['t']}({t_sec:.1f}s)" if i < 8 else "_")
        ax_xy.scatter(pred_p[-1, 0], pred_p[-1, 1],
                      color=c, s=50, marker="^", zorder=4)

    ax_xy.set_xlabel("X [m]")
    ax_xy.set_ylabel("Y [m]")
    ax_xy.set_title(f"Episode {episode_idx}  |  XY 평면 투영 (위에서 바라본 뷰)\n"
                    f"● 시작점, △ 예측끝점  /  색상: 초반→후반")
    ax_xy.legend(fontsize=8, ncol=2, loc="best")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_aspect("equal", "box")

    sm = plt.cm.ScalarMappable(cmap="coolwarm",
                               norm=plt.Normalize(0, (T_down - 1) * STRIDE / 30.0))
    sm.set_array([])
    fig_xy.colorbar(sm, ax=ax_xy, label="Inference 시작 시각 [s]", shrink=0.7)

    path_xy = output_dir / f"episode{episode_idx:03d}_xy.png"
    fig_xy.savefig(str(path_xy), dpi=150, bbox_inches="tight")
    plt.close(fig_xy)
    print(f"[SAVED] XY plot → {path_xy}")


# ══════════════════════════════════════════════
# 5. Interactive HTML (Plotly)
# ══════════════════════════════════════════════
def plot_interactive_html(ee_pos_full, cmd_pos_full, obs_dict, act_gt_pos,
                          infer_results, episode_idx, output_dir):
    """
    Plotly 기반 인터랙티브 3D HTML 생성.
    브라우저에서 마우스로 돌리기/줌/팬 가능.

    traces:
      - GT EE trajectory      : 파란 실선
      - GT Command trajectory  : 주황 점선
      - Inference 시작점       : 컬러 마커 (에피소드 시간 기준 coolwarm)
      - Inference 예측 경로    : 같은 색 실선, hover에 시각 정보
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    T_down  = len(obs_dict["robot_pose_L"])
    n_infer = len(infer_results)

    # coolwarm 색상 → Plotly rgb 문자열 변환
    cmap   = cm.get_cmap("coolwarm", n_infer)
    def to_rgb(i):
        r, g, b, _ = cmap(i / max(n_infer - 1, 1))
        return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

    traces = []

    # ── GT EE (full resolution) ──────────────────────────────────
    traces.append(go.Scatter3d(
        x=ee_pos_full[:, 0], y=ee_pos_full[:, 1], z=ee_pos_full[:, 2],
        mode="lines",
        line=dict(color="steelblue", width=4),
        name="GT EE (current)",
        hovertemplate="GT EE<br>X=%{x:.4f}<br>Y=%{y:.4f}<br>Z=%{z:.4f}<extra></extra>",
    ))

    # ── GT Command / Desired (full resolution) ───────────────────
    traces.append(go.Scatter3d(
        x=cmd_pos_full[:, 0], y=cmd_pos_full[:, 1], z=cmd_pos_full[:, 2],
        mode="lines",
        line=dict(color="darkorange", width=3, dash="dash"),
        name="GT Command (desired)",
        hovertemplate="GT Cmd<br>X=%{x:.4f}<br>Y=%{y:.4f}<br>Z=%{z:.4f}<extra></extra>",
    ))

    # ── Inference 결과 ────────────────────────────────────────────
    for i, res in enumerate(infer_results):
        t      = res["t"]
        t_sec  = t * STRIDE / 30.0
        obs_p  = res["obs_pos"]
        pred_p = res["pred_pos"]   # (n_action_steps, 3)
        color  = to_rgb(i)

        # 예측 trajectory: 현재 obs 위치 → 예측 경로
        traj_x = np.concatenate([[obs_p[0]], pred_p[:, 0]])
        traj_y = np.concatenate([[obs_p[1]], pred_p[:, 1]])
        traj_z = np.concatenate([[obs_p[2]], pred_p[:, 2]])
        n_pts  = len(traj_x)
        hover_texts = [f"step={t} ({t_sec:.1f}s) START"] + \
                      [f"step={t} +{k+1}  pred" for k in range(n_pts - 1)]

        traces.append(go.Scatter3d(
            x=traj_x, y=traj_y, z=traj_z,
            mode="lines+markers",
            line=dict(color=color, width=3),
            marker=dict(
                size=[10] + [4] * (n_pts - 2) + [7],   # 시작=크게, 중간=작게, 끝=중간
                symbol=["circle"] + ["circle"] * (n_pts - 2) + ["diamond"],
                color=color,
                line=dict(color="black", width=1),
            ),
            name=f"Infer t={t} ({t_sec:.1f}s)",
            text=hover_texts,
            hovertemplate="%{text}<br>X=%{x:.4f}<br>Y=%{y:.4f}<br>Z=%{z:.4f}<extra></extra>",
            legendgroup=f"infer_{i}",
        ))

    # ── 1:1:1 물리 스케일 계산 ──────────────────────────────────
    # 전체 데이터(GT + 예측)의 min/max로 중심 구하고,
    # 세 축 모두 동일한 half_range로 맞춤 → 0.1m = 0.1m = 0.1m
    all_pts = np.vstack([ee_pos_full, cmd_pos_full])
    for res in infer_results:
        all_pts = np.vstack([all_pts, res["pred_pos"]])

    center    = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2.0
    half_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2.0 * 1.15  # 15% 여백

    ax_range = lambda c: [float(c - half_range), float(c + half_range)]
    x_rng = ax_range(center[0])
    y_rng = ax_range(center[1])
    z_rng = ax_range(center[2])

    # ── 레이아웃 ─────────────────────────────────────────────────
    fig = go.Figure(data=traces)

    # 뷰 프리셋 정의 (eye 좌표 + up 벡터)
    # up 벡터를 각 뷰에 맞게 지정해야 gimbal lock 없이 정확히 보임
    view_presets = [
        ("🔄 ISO",       dict(x=1.5,  y=1.5,  z=0.8),  dict(x=0, y=0, z=1)),
        ("⬆ 상면(Top)",  dict(x=0,    y=0,    z=2.5),  dict(x=0, y=1, z=0)),
        ("⬇ 하면(Bot)",  dict(x=0,    y=0,   z=-2.5),  dict(x=0, y=1, z=0)),
        ("➡ 정면(+Y)",   dict(x=0,    y=2.5,  z=0),    dict(x=0, y=0, z=1)),
        ("⬅ 후면(-Y)",   dict(x=0,   y=-2.5,  z=0),    dict(x=0, y=0, z=1)),
        ("↗ 우측(+X)",   dict(x=2.5,  y=0,    z=0),    dict(x=0, y=0, z=1)),
        ("↙ 좌측(-X)",   dict(x=-2.5, y=0,    z=0),    dict(x=0, y=0, z=1)),
    ]

    buttons = []
    for label, eye, up in view_presets:
        buttons.append(dict(
            label=label,
            method="relayout",
            args=[{"scene.camera": {"eye": eye, "up": up,
                                    "projection": {"type": "perspective"}}}],
        ))

    fig.update_layout(
        title=dict(
            text=(f"Episode {episode_idx} — Inference 3D Trajectory<br>"
                  f"<sub>파란선=GT EE | 주황점선=GT Command | 컬러선=Inference 예측  "
                  f"| 🖱 좌클릭드래그=회전  스크롤=줌  우클릭드래그=팬</sub>"),
            font=dict(size=15),
        ),
        scene=dict(
            xaxis=dict(title="X [m]", range=x_rng),
            yaxis=dict(title="Y [m]", range=y_rng),
            zaxis=dict(title="Z [m]", range=z_rng),
            aspectmode="cube",         # 세 축 동일 시각 길이 + 동일 range → 완벽 1:1:1
            dragmode="orbit",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8),
                up=dict(x=0, y=0, z=1),
                projection=dict(type="perspective"),
            ),
        ),
        # 뷰 프리셋 버튼 (우상단)
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.0, y=1.08,
            xanchor="left",
            yanchor="top",
            pad=dict(r=5, t=5),
            showactive=True,
            bgcolor="rgba(240,240,240,0.9)",
            bordercolor="gray",
            font=dict(size=11),
            buttons=buttons,
        )],
        legend=dict(
            x=0.01, y=0.98,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=0, r=0, t=100, b=0),
        width=1200,
        height=850,
    )

    # 컬러바 역할을 하는 더미 trace
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode="markers",
        marker=dict(
            colorscale="RdBu_r",
            color=[0, 1],
            colorbar=dict(
                title="Infer 시작<br>시각 [s]",
                tickvals=[0, 0.5, 1],
                ticktext=["0s", f"{T_down*STRIDE/30/2:.0f}s",
                          f"{(T_down-1)*STRIDE/30:.0f}s"],
                len=0.45,
                x=1.02,
            ),
            showscale=True,
            size=0,
        ),
        showlegend=False,
    ))

    path_html = output_dir / f"episode{episode_idx:03d}_interactive.html"
    fig.write_html(str(path_html), include_plotlyjs="cdn")
    print(f"[SAVED] Interactive HTML → {path_html}")
    print(f"        조작: 좌클릭드래그=회전 / 스크롤=줌 / 우클릭드래그=팬 / 상단버튼=뷰 프리셋")


# ══════════════════════════════════════════════
# 6. 수치 요약 출력
# ══════════════════════════════════════════════
def print_summary(obs_dict, act_gt_pos, infer_results):
    print("\n" + "=" * 60)
    print("Inference 요약")
    print("=" * 60)
    print(f"{'step':>5} {'t[s]':>6} | {'GT pos':^28} | {'Pred[0] pos':^28} | {'err[mm]':>8}")
    print("-" * 85)

    for res in infer_results:
        t       = res["t"]
        t_sec   = t * STRIDE / 30.0
        obs_p   = obs_dict["robot_pose_L"][t]
        gt_cmd  = act_gt_pos[t]
        pred_p0 = res["pred_pos"][0]
        err_mm  = np.linalg.norm(gt_cmd - pred_p0) * 1000

        print(f"{t:>5} {t_sec:>6.1f} | "
              f"[{gt_cmd[0]:.3f},{gt_cmd[1]:.3f},{gt_cmd[2]:.3f}] | "
              f"[{pred_p0[0]:.3f},{pred_p0[1]:.3f},{pred_p0[2]:.3f}] | "
              f"{err_mm:>8.2f}")

    errs = [np.linalg.norm(act_gt_pos[r["t"]] - r["pred_pos"][0]) * 1000
            for r in infer_results]
    print("-" * 85)
    print(f"  평균 위치 오차 (1-step): {np.mean(errs):.2f} mm")
    print(f"  최대 위치 오차 (1-step): {np.max(errs):.2f} mm")
    print("=" * 60)


# ══════════════════════════════════════════════
# 6. Entry point
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Zarr 에피소드 → Inference 3D 시각화")
    parser.add_argument("--checkpoint", "-c",
        default="data/outputs/logistic_box_unet_abs/checkpoints/epoch=0400-train_loss=0.002.ckpt",
        help="checkpoint .ckpt 경로")
    parser.add_argument("--dataset_root", "-r",
        default="/home/rush/Desktop/Datasets/20260630_195919",
        help="zarr 에피소드 루트 디렉토리")
    parser.add_argument("--episode_idx", "-e", type=int, default=0,
        help="시각화할 에피소드 인덱스 (기본 0)")
    parser.add_argument("--infer_every", "-i", type=int, default=20,
        help="몇 step(10Hz 기준)마다 inference할지 (기본 20 = 2초마다)")
    parser.add_argument("--output_dir", "-o",
        default="./inference_results_3d",
        help="결과 저장 디렉토리")
    parser.add_argument("--device",
        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"[INFO] Checkpoint : {args.checkpoint}")
    print(f"[INFO] Dataset    : {args.dataset_root}  (episode {args.episode_idx})")
    print(f"[INFO] Infer every: {args.infer_every} steps  "
          f"({args.infer_every * STRIDE / 30.0:.1f}초마다)")
    print(f"[INFO] Device     : {args.device}")

    # 1. Policy 로드
    policy, cfg = load_policy(args.checkpoint, args.device)

    # 2. Zarr 에피소드 로드
    ee_pos_full, cmd_pos_full, obs_dict, act_gt_pos, T = load_episode(
        args.dataset_root, args.episode_idx)

    # 3. Inference
    infer_results = run_inference_at_points(
        policy, cfg, obs_dict, args.infer_every, args.device)

    # 4. 수치 요약
    print_summary(obs_dict, act_gt_pos, infer_results)

    # 5. matplotlib 정적 이미지 저장
    plot_3d(ee_pos_full, cmd_pos_full, obs_dict, act_gt_pos,
            infer_results, args.episode_idx, args.output_dir)

    # 6. Plotly 인터랙티브 HTML 저장
    plot_interactive_html(ee_pos_full, cmd_pos_full, obs_dict, act_gt_pos,
                          infer_results, args.episode_idx, args.output_dir)

    print(f"\n[완료] 결과 저장 위치: {args.output_dir}/")


if __name__ == "__main__":
    main()
