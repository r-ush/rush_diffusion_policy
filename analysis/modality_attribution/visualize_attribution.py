#!/usr/bin/env python
"""에피소드별 fine-grained modality attribution 시각화.

replay_offline.py 가 "vision vs wrench 어느 쪽이 더 영향?"(modality-level)까지 했다면,
이 스크립트는 그 안을 더 쪼갠다:

  1) Vision spatial saliency (occlusion) : 이미지 패치를 하나씩 가려(mean으로 대체) action이
     얼마나 바뀌는지 → (grid x grid) 히트맵 = "정책이 이미지의 어디를 보나".
  2) Force per-axis attribution           : 손목 wrench 6채널(Fx,Fy,Fz,Tx,Ty,Tz)을 하나씩 0으로
     바꿔 Δaction 측정 → "어떤 힘/토크 축이 action을 좌우하나".
  3) (참고) modality dominance             : vision(freeze) vs wrench(zero) Δ 요약도 같이 표기.

attribution.py 의 검증된 도구(seed 고정 예측, action_delta, normalizer)를 그대로 쓴다.
occlusion/축 ablation은 배치로 돌리되 base/ablated를 '같은 seed'로 예측해 noise를 매칭한다.

사용 예:
  python -m analysis.modality_attribution.visualize_attribution \
      -i data/outputs/260710_insert_box_hand_wrench_abs/epoch=0900-train_loss=0.001.ckpt \
      --obs data/online_runs/run_hand/actor_episodes/eval_debug/episode_000013_infer_obs.hdf5 \
      -o   data/online_runs/run_hand/actor_episodes/attribution_ep013/detail
"""

from __future__ import annotations

import os
import pathlib

import click
import numpy as np
import torch

from analysis.modality_attribution import attribution as attr
from analysis.modality_attribution.replay_offline import load_policy
from analysis.modality_attribution.record_infer_obs import load_inference_obs

WRENCH_AXIS_LABELS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]


# ---------------------------------------------------------------------------
# per-row normalized action delta (배치 벡터화)
# ---------------------------------------------------------------------------
def _per_row_total_delta(policy, action_a: torch.Tensor, action_b: torch.Tensor) -> np.ndarray:
    """(B,S,Da) 두 action의 행별 normalized L2 (step 평균) → (B,)."""
    na = policy.normalizer["action"].normalize(action_a.float())
    nb = policy.normalizer["action"].normalize(action_b.float())
    d = torch.linalg.norm(na - nb, dim=-1).mean(dim=-1)  # (B,)
    return d.detach().cpu().numpy()


def _expand_obs(obs_dict, key_override, B):
    """obs_dict(B=1)를 B로 확장. key_override 딕셔너리의 키는 그 텐서로 대체."""
    out = {}
    for k, v in obs_dict.items():
        if k in key_override:
            out[k] = key_override[k]
        else:
            out[k] = v.expand(B, *v.shape[1:]).clone()
    return out


# ---------------------------------------------------------------------------
# 1) vision occlusion saliency
# ---------------------------------------------------------------------------
def occlusion_saliency(policy, obs_dict, rgb_key, grid=8, seeds=(0,), chunk=16):
    """이미지 패치별 occlusion Δaction → (grid, grid) saliency.

    patch를 (모든 obs step/채널에 대해) 이미지 채널 평균으로 덮고, base(안 가림)와 같은 seed로
    예측해 Δ를 잰다. 큰 값 = 그 영역이 action에 크게 기여(=정책이 주목).
    """
    img = obs_dict[rgb_key]                      # (1, T, C, H, W)
    _, T, C, H, W = img.shape
    ys = np.linspace(0, H, grid + 1).astype(int)
    xs = np.linspace(0, W, grid + 1).astype(int)
    patches = [(gy, gx) for gy in range(grid) for gx in range(grid)]
    G = len(patches)
    mean_val = img.mean(dim=(0, 1, 3, 4)).view(1, C, 1, 1)  # (1,C,1,1) 채널 평균

    sal = np.zeros(G, dtype=np.float64)
    for seed in seeds:
        for start in range(0, G, chunk):
            sub = patches[start:start + chunk]
            b = len(sub)
            occ_img = img.expand(b, T, C, H, W).clone()
            for j, (gy, gx) in enumerate(sub):
                y0, y1 = ys[gy], ys[gy + 1]
                x0, x1 = xs[gx], xs[gx + 1]
                occ_img[j, :, :, y0:y1, x0:x1] = mean_val
            base_img = img.expand(b, T, C, H, W).clone()

            occ_obs = _expand_obs(obs_dict, {rgb_key: occ_img}, b)
            base_obs = _expand_obs(obs_dict, {rgb_key: base_img}, b)
            ba = attr.predict_action(policy, base_obs, seed)["action"]
            oa = attr.predict_action(policy, occ_obs, seed)["action"]
            sal[start:start + b] += _per_row_total_delta(policy, ba, oa)
    sal /= max(len(seeds), 1)
    return sal.reshape(grid, grid)


# ---------------------------------------------------------------------------
# 2) force per-axis attribution
# ---------------------------------------------------------------------------
def force_axis_attribution(policy, obs_dict, wrench_key, n_axes=6, seeds=(0, 1, 2)):
    """wrench 6채널을 하나씩 0으로 → 축별 Δaction(total). 반환 (n_axes,)."""
    deltas = np.zeros(n_axes, dtype=np.float64)
    for seed in seeds:
        base_batch = _expand_obs(obs_dict, {}, n_axes)          # 6× full wrench
        occ_wr = obs_dict[wrench_key].expand(n_axes, *obs_dict[wrench_key].shape[1:]).clone()
        for c in range(n_axes):
            occ_wr[c, ..., c, :] = 0.0                          # 행 c: 축 c만 0
        occ_batch = _expand_obs(obs_dict, {wrench_key: occ_wr}, n_axes)
        ba = attr.predict_action(policy, base_batch, seed)["action"]
        oa = attr.predict_action(policy, occ_batch, seed)["action"]
        deltas += _per_row_total_delta(policy, ba, oa)
    deltas /= max(len(seeds), 1)
    return deltas


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _rgb_last_frame(obs_np, rgb_key):
    """저장된 obs의 이미지(마지막 obs step)를 (H,W,3) uint8-ish [0,1]로."""
    img = np.asarray(obs_np[rgb_key])            # (T, C, H, W)
    frame = img[-1]                               # (C,H,W)
    frame = np.moveaxis(frame, 0, -1)             # (H,W,C)
    if frame.dtype != np.float32 and frame.max() > 1.5:
        frame = frame / 255.0
    return np.clip(frame, 0, 1)


def _upsample(sal2d, H, W):
    t = torch.from_numpy(sal2d)[None, None].float()
    up = torch.nn.functional.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return up[0, 0].numpy()


def render_vision_saliency(png_path, frames, dominant_labels):
    """frames: list of (inference_idx, rgb(H,W,3), sal(grid,grid), elapsed_s)."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    n = len(frames)
    if n == 0:
        return
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.4 * nrow), squeeze=False)
    for idx, (inf_i, rgb, sal, t_s) in enumerate(frames):
        ax = axes[idx // ncol][idx % ncol]
        H, W = rgb.shape[:2]
        ax.imshow(rgb)
        up = _upsample(sal, H, W)
        im = ax.imshow(up, cmap="jet", alpha=0.5)
        ax.set_title(f"inf {inf_i}  t={t_s:.1f}s\n{dominant_labels.get(inf_i,'')}", fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Vision occlusion saliency (brighter = occluding it changes the action more = more attended)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"Vision saliency PNG saved: {png_path}")


def render_force_axes(png_path, elapsed_s, axis_matrix, axis_labels):
    """axis_matrix: (N_inf, n_axes) Δaction(total)."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    axis_matrix = np.asarray(axis_matrix)  # (N, A)
    N, A = axis_matrix.shape
    x = np.asarray(elapsed_s, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        x = np.arange(N, dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                             gridspec_kw={"height_ratios": [2, 1]})
    # heatmap: axes x inferences
    ax = axes[0]
    im = ax.imshow(axis_matrix.T, aspect="auto", cmap="magma", origin="lower",
                   extent=[0, N, -0.5, A - 0.5])
    ax.set_yticks(range(A))
    ax.set_yticklabels(axis_labels)
    ax.set_xlabel("inference #")
    ax.set_title("Force per-axis attribution  (delta-action when each axis is zeroed; brighter = that axis matters more)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Δ action (normalized)")

    # episode-mean bar
    ax = axes[1]
    means = axis_matrix.mean(axis=0)
    colors = ["#d62728", "#d62728", "#d62728", "#1f77b4", "#1f77b4", "#1f77b4"][:A]
    ax.bar(range(A), means, color=colors)
    ax.set_xticks(range(A))
    ax.set_xticklabels(axis_labels)
    ax.set_ylabel("mean Δ action")
    ax.set_title("Episode-mean per-axis (red=force, blue=torque)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Force per-axis PNG saved: {png_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command()
@click.option("--input", "-i", required=True, help="Path to checkpoint")
@click.option("--obs", required=True, help="episode_XXXXXX_infer_obs.hdf5")
@click.option("--output", "-o", required=True, help="Directory to save outputs")
@click.option("--num_inference_steps", "-n", default=16, type=int, show_default=True)
@click.option("--seeds", default="0,1", help="Comma-separated seeds (occlusion은 비싸서 기본 2개).")
@click.option("--grid", default=8, type=int, show_default=True, help="Vision occlusion grid (grid x grid).")
@click.option("--occ_chunk", default=16, type=int, show_default=True, help="Occlusion 배치 청크(OOM 방지).")
@click.option("--vision_frames", default="6", help="saliency 그릴 프레임 수(정수) 또는 'all'.")
@click.option("--device", default="cuda", show_default=True)
def main(input, obs, output, num_inference_steps, seeds, grid, occ_chunk, vision_frames, device):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    seeds = [int(s) for s in str(seeds).split(",") if s.strip() != ""]
    out = pathlib.Path(output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading policy from {input}")
    policy, cfg = load_policy(input, num_inference_steps=num_inference_steps, device=device)
    rgb_key = policy.rgb_keys[0] if len(policy.rgb_keys) > 0 else None
    wrench_key = policy.wrench_keys[0] if len(policy.wrench_keys) > 0 else None
    print(f"  rgb_key={rgb_key}, wrench_keys={policy.wrench_keys}, low_dim_keys={policy.low_dim_keys}")

    data = load_inference_obs(obs)
    obs_by_inference = data["obs_by_inference"]
    elapsed_s = np.asarray(data["elapsed_s"], dtype=np.float64)
    inference_index = np.asarray(data["inference_index"])
    N = len(obs_by_inference)
    print(f"  {N} inferences. obs keys: {data['obs_keys']}")

    device_t = policy.device

    # ── modality dominance (vision freeze vs wrench zero) + force per-axis ──
    dominant_labels = {}
    dom_rows = []          # (Δvision_total, Δwrench_total)
    axis_matrix = []       # (N, 6)
    start_obs = attr.obs_np_to_tensor(obs_by_inference[0], device_t)
    for i, obs_np in enumerate(obs_by_inference):
        obs_dict = attr.obs_np_to_tensor(obs_np, device_t)

        dv = dw = float("nan")
        baselines = {}
        if rgb_key is not None:
            # 공정 baseline: 중립(평균) 이미지로 교체 = zero-wrench의 vision 대응.
            # (freeze-to-start는 로봇 이동량에 비례해 vision을 과대평가함 — force 분석에서 확인)
            baselines["vision"] = attr.make_blank_vision(policy)
        if wrench_key is not None:
            baselines["wrench"] = attr.make_zero_wrench(policy)
        res = attr.ablation_deltas(policy, obs_dict, baselines, seeds=seeds)
        if "vision" in res.deltas:
            dv = res.deltas["vision"].total
        if "wrench" in res.deltas:
            dw = res.deltas["wrench"].total
        dom_rows.append((dv, dw))
        if rgb_key is not None and wrench_key is not None:
            dom = "VISION" if dv >= dw else "WRENCH"
            dominant_labels[int(inference_index[i])] = f"dom={dom} (v={dv:.3f}/w={dw:.3f})"

        if wrench_key is not None:
            ax_d = force_axis_attribution(policy, obs_dict, wrench_key, seeds=seeds)
            axis_matrix.append(ax_d)
        t = elapsed_s[i] if np.isfinite(elapsed_s[i]) else i
        top_axis = (WRENCH_AXIS_LABELS[int(np.argmax(axis_matrix[-1]))]
                    if wrench_key is not None else "-")
        print(f"[{i+1}/{N}] t={t:.2f}s  dom_v={dv:.4f} dom_w={dw:.4f}  top_force_axis={top_axis}")

    # ── force per-axis figure ──
    if wrench_key is not None and len(axis_matrix) > 0:
        axis_matrix = np.asarray(axis_matrix)
        render_force_axes(out.joinpath("force_axis_attribution.png"),
                          elapsed_s, axis_matrix, WRENCH_AXIS_LABELS)
        np.savez(out.joinpath("force_axis.npz"),
                 elapsed_s=elapsed_s, inference_index=inference_index,
                 axis_matrix=axis_matrix, axis_labels=WRENCH_AXIS_LABELS)
        mean_axes = axis_matrix.mean(axis=0)
        print("\n===== Force per-axis mean =====")
        for lbl, val in zip(WRENCH_AXIS_LABELS, mean_axes):
            print(f"  {lbl}: {val:.4f}")
        print(f"  dominant axis: {WRENCH_AXIS_LABELS[int(np.argmax(mean_axes))]}")

    # ── vision occlusion saliency (선택 프레임) ──
    if rgb_key is not None:
        if str(vision_frames).lower() == "all":
            sel = list(range(N))
        else:
            k = min(int(vision_frames), N)
            sel = sorted(set(np.linspace(0, N - 1, k).astype(int).tolist()))
        print(f"\nVision occlusion saliency on {len(sel)} frames: {sel} (grid={grid})")
        frames = []
        sal_stack = []
        for i in sel:
            obs_dict = attr.obs_np_to_tensor(obs_by_inference[i], device_t)
            sal = occlusion_saliency(policy, obs_dict, rgb_key,
                                     grid=grid, seeds=seeds, chunk=occ_chunk)
            rgb = _rgb_last_frame(obs_by_inference[i], rgb_key)
            frames.append((int(inference_index[i]), rgb, sal, float(elapsed_s[i]) if np.isfinite(elapsed_s[i]) else float(i)))
            sal_stack.append(sal)
            print(f"  frame {i}: saliency max={sal.max():.4f} argmax(grid)={np.unravel_index(sal.argmax(), sal.shape)}")
        render_vision_saliency(out.joinpath("vision_saliency.png"), frames, dominant_labels)
        np.savez(out.joinpath("vision_saliency.npz"),
                 selected=np.asarray(sel), saliency=np.asarray(sal_stack), grid=grid)

    print(f"\nDone. outputs in {out}")


if __name__ == "__main__":
    main()
