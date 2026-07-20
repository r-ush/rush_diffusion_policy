#!/usr/bin/env python
"""GRU vs Causal Conv (TCN) force encoder 벤치마크 — synthetic 대신 실측 wrench 사용.

배경:
  online_runs/*/accumulated.hdf5 에 저장된 wrench_wrist_R (T,6,32) 윈도우들은
  실제 로봇 250Hz 힘/토크 센서를 30~600Hz 근방으로 샘플링한 최근 32-sample
  버퍼다. 인접한 step의 윈도우는 대부분 겹치므로(data_loader.reconstruct_stream)
  거의 연속적인 실측 시계열을 복원할 수 있고, magnitude/derivative 기반
  weak-label(0=자유공간, 1=지속 접촉/sliding, 2=충돌성 스파이크)로 프레임별
  3-class 라벨을 만든다.

이 스크립트가 하는 일:
  1. 여러 demo를 로드해 학습/평가로 나누고 (B,T,6)->(B,T,3) 프레임 분류기로
     GRU / CausalConv 두 모델을 각각 학습.
  2. 평가 demo에서 실제 충돌 스파이크 onset 대비 각 모델이 "collision" 확률을
     threshold 넘기는 데 걸리는 지연(phase lag, sample 단위 + 250Hz 가정 ms)을 측정.
  3. (batch, seq_len) forward pass 시간(ms)을 seq_len 스윕으로 측정해 CausalConv의
     병렬화 이점을 정량화.
  4. 위 결과를 하나의 PNG로 시각화.

사용:
  python -m analysis.force_encoder_benchmark.benchmark \
      --data-glob "/media/rush/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7/online_runs/run_hand*/accumulated.hdf5"
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import click
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from analysis.force_encoder_benchmark.data_loader import (
    LabeledStream, load_demo_streams, load_zarr_episode_streams, make_chunks,
)
from analysis.force_encoder_benchmark.models import CausalConvEncoder, GRUEncoder

RAW_SENSOR_HZ = 250.0  # rightarm_hand_with_wrench_encoder_interpolation_controller.py 주석 기준
AXIS_COLORS = {"Fx": "tab:green", "Fy": "tab:orange", "Fz": "tab:purple"}
LABEL_BG = {0: ("#ffffff", "free"), 1: ("#fff3b0", "sliding"), 2: ("#ffb3b3", "collision")}


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def build_dataset(streams: List[LabeledStream], seq_len: int, stride: int,
                   mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
    chunks = []
    for s in streams:
        chunks.extend(make_chunks(s, seq_len, stride))
    if not chunks:
        raise click.UsageError(f"seq_len={seq_len} 에 맞는 chunk가 없습니다 (데이터가 너무 짧음).")
    X = np.stack([c[0] for c in chunks])  # (N, seq_len, 6)
    Y = np.stack([c[1] for c in chunks])  # (N, seq_len)

    if mean is None:
        mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
        std = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
    X = (X - mean) / std
    return torch.from_numpy(X.astype(np.float32)), torch.from_numpy(Y), mean, std


def train_model(model, loader, epochs, lr, device, name, class_weights=None):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    weight = class_weights.to(device) if class_weights is not None else None
    for epoch in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                                    weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.shape[0]
            n += x.shape[0]
        print(f"  [{name}] epoch {epoch+1}/{epochs}  loss={total_loss/n:.4f}")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# phase-lag evaluation
# ---------------------------------------------------------------------------

def find_true_onsets(labels: np.ndarray, target: int = 2) -> List[int]:
    onsets = []
    prev = 0
    for i, l in enumerate(labels):
        if l == target and prev != target:
            onsets.append(i)
        prev = l
    return onsets


def detect_lag(prob_collision: np.ndarray, onset: int, thr: float = 0.5,
               max_lag: int = 50) -> Optional[int]:
    for k in range(0, max_lag + 1):
        idx = onset + k
        if idx >= len(prob_collision):
            break
        if prob_collision[idx] >= thr:
            return k
    return None


@torch.no_grad()
def predict_stream(model, stream: LabeledStream, mean, std, device) -> np.ndarray:
    x = (stream.force - mean) / std
    x = torch.from_numpy(x.astype(np.float32))[None].to(device)  # (1, L, 6)
    logits = model(x)
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()  # (L, 3)
    return probs


def evaluate_phase_lag(models: dict, eval_streams: List[LabeledStream], mean, std,
                        device, thr: float = 0.5, max_lag: int = 50):
    results = {name: {"lags": [], "missed": 0, "probs": []} for name in models}
    onsets_per_stream = []
    for stream in eval_streams:
        onsets = find_true_onsets(stream.labels, target=2)
        onsets_per_stream.append(onsets)
        for name, model in models.items():
            probs = predict_stream(model, stream, mean, std, device)
            results[name]["probs"].append(probs[:, 2])
            for onset in onsets:
                lag = detect_lag(probs[:, 2], onset, thr=thr, max_lag=max_lag)
                if lag is None:
                    results[name]["missed"] += 1
                else:
                    results[name]["lags"].append(lag)
    return results, onsets_per_stream


# ---------------------------------------------------------------------------
# latency benchmark
# ---------------------------------------------------------------------------

@torch.no_grad()
def benchmark_latency(models: dict, seq_lens, batch_size, device, n_warmup=5, n_iters=30):
    table = {name: [] for name in models}
    for name, model in models.items():
        model.eval()
        for seq_len in seq_lens:
            x = torch.randn(batch_size, seq_len, 6, device=device)
            for _ in range(n_warmup):
                model(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iters):
                model(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) / n_iters * 1000
            table[name].append(elapsed_ms)
            print(f"  [{name}] seq_len={seq_len:>5d}  batch={batch_size:>3d}  "
                  f"{elapsed_ms:.3f} ms/batch")
    return table


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def _plot_report(out_path, eval_streams, onsets_per_stream, lag_results,
                  seq_lens, latency_table, thr):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    n_events_to_show = 3
    fig, axes = plt.subplots(2, n_events_to_show, figsize=(15, 7))

    # --- row 1: zoom around first few collision onsets (phase lag) ---
    stream = eval_streams[0]
    onsets = onsets_per_stream[0]
    shown = onsets[:n_events_to_show] if onsets else []
    colors = {"GRU": "#d62728", "CausalConv": "#1f77b4"}
    for col in range(n_events_to_show):
        ax = axes[0, col]
        if col >= len(shown):
            ax.axis("off")
            continue
        onset = shown[col]
        lo, hi = max(0, onset - 15), min(len(stream.mag), onset + 60)
        xs = np.arange(lo, hi)
        ax2 = ax.twinx()
        for axis_i, (axis_name, axis_color) in enumerate(AXIS_COLORS.items()):
            ax2.plot(xs, stream.force[lo:hi, axis_i], color=axis_color, lw=1.1,
                     alpha=0.8, label=axis_name)
        ax2.set_ylabel("Force (N)")
        for name in ("GRU", "CausalConv"):
            probs = lag_results[name]["probs"][0]
            ax.plot(xs, probs[lo:hi], color=colors[name], lw=1.8, label=f"P(collision) {name}")
        ax.axvline(onset, color="black", ls="--", lw=1.2, label="true onset")
        ax.axhline(thr, color="black", ls=":", lw=0.8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"event #{col+1} (onset idx={onset})")
        ax.set_xlabel("sample idx")
        if col == 0:
            ax.set_ylabel("P(collision)")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6.5)

    # --- row 2: latency vs seq_len ---
    ax = axes[1, 0]
    for name, values in latency_table.items():
        ax.plot(seq_lens, values, marker="o", label=name, color=colors.get(name))
    ax.set_xlabel("sequence length")
    ax.set_ylabel("latency (ms / batch)")
    ax.set_title("Forward-pass latency vs seq_len")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- row 2 col2: lag histogram ---
    ax = axes[1, 1]
    for name in ("GRU", "CausalConv"):
        lags = lag_results[name]["lags"]
        if lags:
            ax.hist(lags, bins=range(0, max(lags) + 2), alpha=0.6, label=name,
                     color=colors[name])
    ax.set_xlabel("detection lag (samples, thr=%.2f)" % thr)
    ax.set_ylabel("count")
    ax.set_title("Collision detection lag distribution")
    ax.legend()

    # --- row 2 col3: summary text ---
    ax = axes[1, 2]
    ax.axis("off")
    lines = ["Summary (nominal %.0fHz raw sensor)" % RAW_SENSOR_HZ, ""]
    for name in ("GRU", "CausalConv"):
        lags = lag_results[name]["lags"]
        missed = lag_results[name]["missed"]
        if lags:
            mean_lag = np.mean(lags)
            lines.append(f"{name}: mean lag = {mean_lag:.1f} samples "
                         f"(~{mean_lag/RAW_SENSOR_HZ*1000:.1f} ms), missed={missed}")
        else:
            lines.append(f"{name}: no detections within window, missed={missed}")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=9, family="monospace")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"PNG saved: {out_path}")


def _shade_labels(ax, labels: np.ndarray):
    """labels 배열의 0/1/2 구간을 배경색으로 칠한다 (free/sliding/collision)."""
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            color, _ = LABEL_BG[int(labels[start])]
            ax.axvspan(start, i, color=color, alpha=0.5, lw=0)
            start = i


def _plot_axis_breakdown(out_path, eval_streams: List[LabeledStream], max_samples: int = 1500):
    """평가 demo 전체 구간에서 Fx/Fy/Fz를 축별로, 라벨 배경과 함께 보여준다.

    |F| magnitude 하나로는 어느 축이 충돌/접촉 특징을 만드는지 안 보이므로,
    축별 시계열 + free/sliding/collision 배경색을 겹쳐서 특징을 드러낸다.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n = len(eval_streams)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.2 * n), squeeze=False)
    for row, stream in enumerate(eval_streams):
        ax = axes[row, 0]
        L = min(len(stream.mag), max_samples)
        xs = np.arange(L)
        _shade_labels(ax, stream.labels[:L])
        for axis_i, (axis_name, axis_color) in enumerate(AXIS_COLORS.items()):
            ax.plot(xs, stream.force[:L, axis_i], color=axis_color, lw=1.0, label=axis_name)
        ax.set_xlim(0, L)
        ax.set_ylabel("Force (N)")
        ax.set_title(f"{stream.source}  (first {L} samples)")
        if row == n - 1:
            ax.set_xlabel("sample idx")
        if row == 0:
            force_handles = [plt.Line2D([0], [0], color=c, lw=1.5) for c in AXIS_COLORS.values()]
            bg_handles = [mpatches.Patch(color=c, alpha=0.5, label=name)
                          for c, name in LABEL_BG.values()]
            ax.legend(force_handles + bg_handles, list(AXIS_COLORS.keys()) +
                      [n for _, n in LABEL_BG.values()], loc="upper right", fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"PNG saved: {out_path}")


def save_per_episode_pngs(out_dir: str, streams: List[LabeledStream]):
    """streams 전부(예: episode_000~099) 각각을 개별 PNG 한 장씩으로 저장.

    _plot_axis_breakdown은 eval demo 몇 개만 한 figure에 합쳐 보여주는데, 여기서는
    학습/평가 구분 없이 로드된 모든 에피소드를 하나씩 별도 파일로 뽑는다.
    """
    import os
    import re

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    ep_dir = os.path.join(out_dir, "per_episode")
    os.makedirs(ep_dir, exist_ok=True)

    force_handles = [plt.Line2D([0], [0], color=c, lw=1.5) for c in AXIS_COLORS.values()]
    bg_handles = [mpatches.Patch(color=c, alpha=0.5, label=name) for c, name in LABEL_BG.values()]
    legend_handles = force_handles + bg_handles
    legend_labels = list(AXIS_COLORS.keys()) + [name for _, name in LABEL_BG.values()]

    for stream in streams:
        m = re.search(r"(episode_\d+)", stream.source)
        name = m.group(1) if m else re.sub(r"[^\w.-]", "_", stream.source)

        L = len(stream.mag)
        xs = np.arange(L)
        fig, ax = plt.subplots(figsize=(max(10, L / 400), 3.2))
        _shade_labels(ax, stream.labels)
        for axis_i, (axis_name, axis_color) in enumerate(AXIS_COLORS.items()):
            ax.plot(xs, stream.force[:, axis_i], color=axis_color, lw=0.9, label=axis_name)
        ax.set_xlim(0, L)
        ax.set_ylabel("Force (N)")
        ax.set_xlabel("sample idx")
        ax.set_title(stream.source)
        ax.legend(legend_handles, legend_labels, loc="upper right", fontsize=7, ncol=2)

        fig.tight_layout()
        out_path = os.path.join(ep_dir, f"{name}.png")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)

    print(f"Saved {len(streams)} per-episode PNGs to {ep_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_GLOB = "/media/rush/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7/online_runs/run_hand*/accumulated.hdf5"


@click.command()
@click.option("--data-glob", default=DEFAULT_GLOB, show_default=True,
              help="accumulated.hdf5 glob 패턴 (--zarr-dir 미지정시 사용)")
@click.option("--zarr-dir", default=None,
              help="UMIFT raw 수집 데이터셋 폴더 (예: .../20260630_195919). "
                   "지정하면 episode_XXX/ft/<wrench-name>.zarr 를 직접 사용 (겹침 복원 불필요).")
@click.option("--wrench-name", default="wrench_raw", show_default=True,
              help="--zarr-dir 사용 시 어떤 wrench를 쓸지 (wrench_raw가 실제 학습 파이프라인 입력)")
@click.option("--limit-episodes", default=0, type=int,
              help="--zarr-dir 사용 시 앞 N개 episode만 사용 (0=전체)")
@click.option("--no-filter", is_flag=True, default=False,
              help="offset-subtract + EMA 전처리를 끄고 raw 신호 그대로 사용 (비교용)")
@click.option("--wrench-offset-samples", default=10, type=int, show_default=True,
              help="실제 파이프라인과 동일: 초기 N샘플 평균을 offset으로 뺀다")
@click.option("--wrench-ema-alpha", default=0.03, type=float, show_default=True,
              help="실제 파이프라인과 동일한 EMA low-pass 계수")
@click.option("--seq-len", default=128, type=int, show_default=True)
@click.option("--stride", default=32, type=int, show_default=True)
@click.option("--epochs", default=15, type=int, show_default=True)
@click.option("--batch-size", default=32, type=int, show_default=True)
@click.option("--lr", default=1e-3, type=float, show_default=True)
@click.option("--eval-holdout", default=2, type=int, show_default=True,
              help="맨 뒤 N개 demo를 학습에서 빼고 평가용으로 사용")
@click.option("--device", default="cuda" if torch.cuda.is_available() else "cpu",
              show_default=True)
@click.option("--seed", default=0, type=int, show_default=True)
@click.option("--out-dir", default=None, help="출력 폴더 (기본: 이 파일 옆 outputs/)")
def main(data_glob, zarr_dir, wrench_name, limit_episodes, no_filter,
         wrench_offset_samples, wrench_ema_alpha, seq_len, stride, epochs,
         batch_size, lr, eval_holdout, device, seed, out_dir):
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = out_dir or os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    if zarr_dir:
        apply_filter = not no_filter
        print(f"Loading real wrench streams from zarr dataset: {zarr_dir} "
              f"(wrench={wrench_name}, limit_episodes={limit_episodes or 'all'}, "
              f"filter={'EMA(alpha=%.3f) after offset-%d' % (wrench_ema_alpha, wrench_offset_samples) if apply_filter else 'OFF (raw)'})")
        streams = load_zarr_episode_streams(zarr_dir, wrench_name=wrench_name,
                                             limit_episodes=limit_episodes,
                                             apply_filter=apply_filter,
                                             offset_samples=wrench_offset_samples,
                                             ema_alpha=wrench_ema_alpha)
    else:
        print(f"Loading real wrench streams from: {data_glob}")
        streams = load_demo_streams(data_glob)
    if len(streams) <= eval_holdout:
        raise click.UsageError(
            f"demo가 {len(streams)}개뿐인데 eval_holdout={eval_holdout} 개를 빼면 학습 데이터가 없습니다.")
    print(f"  loaded {len(streams)} demos, lengths="
          f"{[len(s.mag) for s in streams]}")
    for s in streams:
        n0, n1, n2 = (s.labels == 0).sum(), (s.labels == 1).sum(), (s.labels == 2).sum()
        print(f"    {s.source}: L={len(s.mag)} free={n0} sliding={n1} collision={n2}")

    train_streams, eval_streams = streams[:-eval_holdout], streams[-eval_holdout:]

    X, Y, mean, std = build_dataset(train_streams, seq_len, stride)
    loader = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)
    print(f"Train chunks: {X.shape[0]}  (seq_len={seq_len}, stride={stride})")

    counts = torch.bincount(Y.reshape(-1), minlength=3).float()
    class_weights = (counts.sum() / (3 * counts.clamp(min=1)))
    print(f"Class counts (free/sliding/collision): {counts.tolist()}  "
          f"-> weights: {class_weights.tolist()}")

    models = {
        "GRU": GRUEncoder(in_dim=6, hidden=64, num_layers=2, num_classes=3),
        "CausalConv": CausalConvEncoder(in_dim=6, channels=(32, 64, 64), kernel_size=3,
                                         dilations=(1, 2, 4), num_classes=3),
    }
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nTraining {name} ({n_params:,} params)...")
        train_model(model, loader, epochs=epochs, lr=lr, device=device, name=name,
                     class_weights=class_weights)

    print("\nEvaluating phase lag on held-out demos...")
    lag_results, onsets_per_stream = evaluate_phase_lag(models, eval_streams, mean, std, device)
    for name in models:
        lags = lag_results[name]["lags"]
        missed = lag_results[name]["missed"]
        if lags:
            print(f"  {name}: n_events={len(lags)+missed}  mean_lag={np.mean(lags):.2f} samples "
                  f"(~{np.mean(lags)/RAW_SENSOR_HZ*1000:.1f} ms)  missed={missed}")
        else:
            print(f"  {name}: no detections, missed={missed}")

    print("\nBenchmarking forward-pass latency...")
    seq_lens = [32, 64, 128, 256, 512, 1024]
    latency_table = benchmark_latency(models, seq_lens, batch_size=32, device=device)

    out_path = os.path.join(out_dir, "gru_vs_causalconv_report.png")
    _plot_report(out_path, eval_streams, onsets_per_stream, lag_results,
                 seq_lens, latency_table, thr=0.5)

    axis_out_path = os.path.join(out_dir, "force_axis_breakdown.png")
    _plot_axis_breakdown(axis_out_path, eval_streams)

    save_per_episode_pngs(out_dir, streams)


if __name__ == "__main__":
    main()
