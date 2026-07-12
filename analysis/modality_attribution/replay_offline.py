#!/usr/bin/env python
"""Offline modality attribution replay.

체크포인트 + inference obs 덤프(record_infer_obs.py 로 저장)를 읽어, 로봇 없이
매 inference마다 "vision vs wrench(force) 중 어느 쪽이 action을 더 좌우했는지"를
counterfactual ablation(기본)으로 계산하고 시간축 그래프/CSV로 저장한다.

사용 예:
  python -m analysis.modality_attribution.replay_offline \
      -i data/outputs/260710_insert_box_hand_wrench_abs/epoch=0900-train_loss=0.001.ckpt \
      --obs data/results/260710_insert_box_hand/eval_debug/episode_000000_infer_obs.hdf5 \
      -o data/results/260710_insert_box_hand/attribution

먼저 rollout 때 record_infer_obs.InferenceObsRecorder 로 obs를 덤프해 두어야 한다
(README 참고). 덤프 파일이 없으면 이 스크립트는 돌릴 수 없다.
"""

from __future__ import annotations

import pathlib

import click
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

from analysis.modality_attribution import attribution as attr
from analysis.modality_attribution.record_infer_obs import load_inference_obs

OmegaConf.register_new_resolver("eval", eval, replace=True)


def load_policy(ckpt_path, num_inference_steps=16, device="cuda"):
    """eval 스크립트와 동일한 방식으로 policy를 로드한다."""
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy: BaseImagePolicy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.eval().to(device)
    policy.num_inference_steps = num_inference_steps
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    return policy, cfg


def build_baselines(policy, obs_by_inference, vision_baseline, wrench_baseline):
    """vision / wrench baseline 생성기를 만든다.

    vision_baseline: 'start'(첫 inference 프레임 고정) | 'self'(자기 마지막 프레임 고정)
    wrench_baseline: 'zero'(무접촉 0)
    """
    builders_per_inference = []

    # start 프레임 기준 obs (첫 inference)
    device = policy.device
    start_obs = attr.obs_np_to_tensor(obs_by_inference[0], device)

    for obs_np in obs_by_inference:
        obs_dict = attr.obs_np_to_tensor(obs_np, device)
        baselines = {}

        if len(policy.rgb_keys) > 0:
            if vision_baseline == "start":
                baselines["vision"] = attr.make_freeze_vision(policy, start_obs)
            elif vision_baseline == "self":
                baselines["vision"] = attr.make_freeze_vision(policy, obs_dict)
            else:
                raise ValueError(f"unknown vision_baseline: {vision_baseline}")

        if len(policy.wrench_keys) > 0:
            if wrench_baseline == "zero":
                baselines["wrench"] = attr.make_zero_wrench(policy)
            else:
                raise ValueError(f"unknown wrench_baseline: {wrench_baseline}")

        builders_per_inference.append((obs_dict, baselines))

    return builders_per_inference


def write_csv(csv_path, elapsed_s, inference_index, rows, modalities):
    import csv

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["inference_index", "elapsed_s"]
        for m in modalities:
            header += [f"{m}_total", f"{m}_pos_m", f"{m}_rot"]
        if len(modalities) == 2:
            header += [f"{modalities[0]}_share"]
        writer.writerow(header)
        for i, row in enumerate(rows):
            line = [int(inference_index[i]), float(elapsed_s[i])]
            for m in modalities:
                d = row[m]
                line += [d.total, d.pos, d.rot]
            if len(modalities) == 2:
                a = row[modalities[0]].total
                b = row[modalities[1]].total
                share = a / (a + b) if (a + b) > 1e-12 else float("nan")
                line.append(share)
            writer.writerow(line)
    print(f"CSV saved: {csv_path}")


def write_timeline_png(png_path, elapsed_s, rows, modalities, grad_rows=None):
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    colors = {"vision": "#1f77b4", "wrench": "#d62728", "low_dim": "#2ca02c"}
    x = np.asarray(elapsed_s, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        x = np.arange(len(rows), dtype=np.float64)

    n_panels = 3 if len(modalities) == 2 else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.2 * n_panels), sharex=True)

    # panel 1: normalized total Δ
    ax = axes[0]
    for m in modalities:
        ax.plot(x, [r[m].total for r in rows], color=colors.get(m, None),
                marker="o", markersize=3, linewidth=1.6, label=f"Δ {m}")
    ax.set_ylabel("Δ action (normalized)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title("Modality attribution (counterfactual ablation)")

    # panel 2: position Δ (m)
    ax = axes[1]
    for m in modalities:
        ax.plot(x, [r[m].pos for r in rows], color=colors.get(m, None),
                marker="o", markersize=3, linewidth=1.6, label=f"Δ {m} pos")
    ax.set_ylabel("Δ position (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # panel 3: dominance share
    if len(modalities) == 2:
        ax = axes[2]
        a = np.array([r[modalities[0]].total for r in rows])
        b = np.array([r[modalities[1]].total for r in rows])
        denom = np.where((a + b) > 1e-12, a + b, np.nan)
        share = a / denom
        ax.plot(x, share, color="#7e22ce", marker="o", markersize=3, linewidth=1.8)
        ax.axhline(0.5, color="#333333", linewidth=0.8, alpha=0.5)
        ax.fill_between(x, 0.5, share, where=share >= 0.5, color=colors.get(modalities[0]), alpha=0.15)
        ax.fill_between(x, 0.5, share, where=share < 0.5, color=colors.get(modalities[1]), alpha=0.15)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"{modalities[0]} dominance\n(share of Δ)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("elapsed time (s)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Timeline PNG saved: {png_path}")


@click.command()
@click.option("--input", "-i", required=True, help="Path to checkpoint")
@click.option("--obs", required=True, help="Path to episode_XXXXXX_infer_obs.hdf5 (record_infer_obs로 저장)")
@click.option("--output", "-o", required=True, help="Directory to save attribution outputs")
@click.option("--num_inference_steps", "-n", default=16, type=int, show_default=True)
@click.option("--seeds", default="0,1,2", help="Comma-separated diffusion sampling seeds to average.")
@click.option("--vision_baseline", type=click.Choice(["start", "self"]), default="start", show_default=True)
@click.option("--wrench_baseline", type=click.Choice(["zero"]), default="zero", show_default=True)
@click.option("--gradient/--no_gradient", default=False, help="Also compute gradient saliency (concat mode only, slower).")
@click.option("--device", default="cuda", show_default=True)
@click.option("--limit", default=0, type=int, help="Only process first N inferences (0 = all).")
def main(input, obs, output, num_inference_steps, seeds, vision_baseline,
         wrench_baseline, gradient, device, limit):
    seeds = [int(s) for s in str(seeds).split(",") if s.strip() != ""]
    output_dir = pathlib.Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading policy from {input}")
    policy, cfg = load_policy(input, num_inference_steps=num_inference_steps, device=device)
    print(f"  fuse_mode={getattr(policy, 'fuse_mode', '?')}, "
          f"rgb_keys={policy.rgb_keys}, wrench_keys={policy.wrench_keys}, "
          f"low_dim_keys={policy.low_dim_keys}")
    print(f"  n_obs_steps={policy.n_obs_steps}, n_action_steps={policy.n_action_steps}, "
          f"num_inference_steps={policy.num_inference_steps}")

    print(f"Loading inference obs from {obs}")
    data = load_inference_obs(obs)
    obs_by_inference = data["obs_by_inference"]
    elapsed_s = data["elapsed_s"]
    inference_index = data["inference_index"]
    if limit and limit > 0:
        obs_by_inference = obs_by_inference[:limit]
        elapsed_s = elapsed_s[:limit]
        inference_index = inference_index[:limit]
    n = len(obs_by_inference)
    print(f"  {n} inferences loaded. obs keys: {data['obs_keys']}")

    builders = build_baselines(policy, obs_by_inference, vision_baseline, wrench_baseline)

    modalities = []
    if len(policy.rgb_keys) > 0:
        modalities.append("vision")
    if len(policy.wrench_keys) > 0:
        modalities.append("wrench")
    if len(modalities) == 0:
        raise click.UsageError("Policy has neither vision nor wrench modality to compare.")

    rows = []
    grad_rows = [] if gradient else None
    for i, (obs_dict, baselines) in enumerate(builders):
        result = attr.ablation_deltas(policy, obs_dict, baselines, seeds=seeds)
        rows.append(result.deltas)

        msg = f"[{i + 1}/{n}] t={float(elapsed_s[i]):.2f}s  " if np.isfinite(elapsed_s[i]) else f"[{i + 1}/{n}]  "
        msg += "  ".join(f"Δ{m}={result.deltas[m].total:.4f}(pos {result.deltas[m].pos*1000:.1f}mm)"
                         for m in modalities)
        if len(modalities) == 2:
            a, b = result.deltas[modalities[0]].total, result.deltas[modalities[1]].total
            dom = modalities[0] if a >= b else modalities[1]
            msg += f"  -> {dom}"
        print(msg)

        if gradient:
            gs = attr.gradient_saliency(policy, obs_dict, seed=seeds[0])
            grad_rows.append(gs)

    # 저장
    write_csv(output_dir.joinpath("attribution.csv"), elapsed_s, inference_index, rows, modalities)
    write_timeline_png(output_dir.joinpath("attribution_timeline.png"), elapsed_s, rows, modalities)

    if gradient and any(g is not None for g in grad_rows):
        import csv as _csv
        with open(output_dir.joinpath("gradient_saliency.csv"), "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["inference_index", "elapsed_s", "vision", "low_dim", "force", "raw_grad_norm"])
            for i, g in enumerate(grad_rows):
                if g is None:
                    continue
                w.writerow([int(inference_index[i]), float(elapsed_s[i]),
                            g.per_group.get("vision"), g.per_group.get("low_dim"),
                            g.per_group.get("force"), g.raw_grad_norm])
        print(f"Gradient saliency CSV saved: {output_dir.joinpath('gradient_saliency.csv')}")

    # 요약
    print("\n===== Episode summary (mean over inferences) =====")
    for m in modalities:
        tot = np.mean([r[m].total for r in rows])
        pos = np.mean([r[m].pos for r in rows])
        print(f"  Δ{m}: total={tot:.4f}, pos={pos*1000:.2f} mm")
    if len(modalities) == 2:
        totals = {m: np.mean([r[m].total for r in rows]) for m in modalities}
        dom = max(totals, key=totals.get)
        print(f"  overall dominant modality: {dom}")


if __name__ == "__main__":
    main()
