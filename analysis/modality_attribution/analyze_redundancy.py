#!/usr/bin/env python
"""Joint / conditional ablation 분석: wrench가 '진짜 안 쓰임'인지 '중복에 가려짐'인지 판정.

배경:
  단일 ablation Δwrench 는 '한 개만' 지웠을 때의 한계(marginal) 기여도다. vision과 wrench가
  중복(redundant) 정보를 담고 있으면, wrench만 지워도 vision이 메워서 Δwrench가 작게 나온다.
  그럼 낮은 Δwrench 는 두 경우로 애매하다:
    (a) 중복인데 정책이 두 경로를 다 학습 → vision이 빠지면 wrench가 대응 = 보완 가능(robust)
    (b) 정책이 vision 경로만 학습 → vision이 빠져도 wrench 반응 없음 = 진짜 미사용(fragile)

핵심 판별식 = conditional ablation:
    Δwrench|no_vision = d( action(vision 제거) , action(vision+wrench 제거) )
  - 이 값이 크면 → (a) 중복에 가려짐: vision을 빼면 wrench가 실제로 action을 좌우함.
  - 이 값이 ~0 이면 → (b) 진짜 미사용: vision을 빼도 wrench는 여전히 무시됨.

사용:
  python -m analysis.modality_attribution.analyze_redundancy \
      -i <ckpt> --obs <episode_XXXXXX_infer_obs.hdf5> -o <outdir> --seeds 0,1
"""

from __future__ import annotations

import pathlib

import click
import numpy as np

from analysis.modality_attribution import attribution as attr
from analysis.modality_attribution.replay_offline import load_policy
from analysis.modality_attribution.record_infer_obs import load_inference_obs


def _verdict(mean_wrench, mean_wrench_cond, scale):
    """conditional Δwrench 를 vision 규모(scale)에 견줘 문장으로 판정."""
    if scale <= 1e-9:
        return "판정 불가 (vision Δ가 0에 가까움 — 정규화 기준 없음)"
    cond_ratio = mean_wrench_cond / scale
    gain = mean_wrench_cond / max(mean_wrench, 1e-9)
    if cond_ratio < 0.15:
        return (f"진짜 미사용: vision을 빼도 wrench 반응이 vision 규모의 {cond_ratio*100:.0f}%뿐 "
                f"→ 현재 정책은 wrench로 vision을 보완하지 못함 (fragile)")
    if cond_ratio > 0.5 and gain > 1.8:
        return (f"중복에 가려짐: vision 제거 시 Δwrench가 {gain:.1f}배로 커지고 vision 규모의 "
                f"{cond_ratio*100:.0f}% → wrench가 대응 가능 (redundant-but-usable)")
    return (f"부분적: conditional Δwrench = vision 규모의 {cond_ratio*100:.0f}%, "
            f"vision 제거 시 {gain:.1f}배 증가 — 약한 보완력")


def _write_png(png_path, elapsed_s, rows):
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    x = np.asarray(elapsed_s, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        x = np.arange(len(rows), dtype=np.float64)

    dv = [r.vision for r in rows]
    dw = [r.wrench for r in rows]
    dw_cond = [r.wrench_given_no_vision for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax = axes[0]
    ax.plot(x, dv, color="#1f77b4", marker="o", ms=3, lw=1.6, label="Δvision (base 대비)")
    ax.plot(x, dw, color="#d62728", marker="o", ms=3, lw=1.6, label="Δwrench (base 대비, 단일)")
    ax.plot(x, dw_cond, color="#7e22ce", marker="o", ms=3, lw=2.0,
            label="Δwrench | vision 제거 (conditional) ★")
    ax.set_ylabel("Δaction (normalized)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title("Conditional ablation: vision을 빼면 wrench가 살아나는가?")

    ax = axes[1]
    both = [r.both for r in rows]
    summ = [r.vision + r.wrench for r in rows]
    ax.plot(x, both, color="#333333", marker="o", ms=3, lw=1.6, label="Δ(vision+wrench 동시 제거)")
    ax.plot(x, summ, color="#999999", ls="--", lw=1.4, label="Δvision + Δwrench (합)")
    ax.fill_between(x, both, summ, where=np.array(summ) >= np.array(both),
                    color="#2ca02c", alpha=0.15, label="중복(redundancy) 영역")
    ax.set_ylabel("Δaction (normalized)")
    ax.set_xlabel("elapsed time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"PNG saved: {png_path}")


@click.command()
@click.option("--input", "-i", required=True, help="Path to checkpoint")
@click.option("--obs", required=True, help="episode_XXXXXX_infer_obs.hdf5")
@click.option("--output", "-o", default=None, help="출력 폴더 (기본: obs 옆 redundancy/)")
@click.option("--seeds", default="0,1", help="Comma-separated diffusion seeds.")
@click.option("--num_inference_steps", "-n", default=16, type=int, show_default=True)
@click.option("--limit", default=0, type=int, help="앞 N개 inference만 (0=전체).")
@click.option("--device", default="cuda", show_default=True)
def main(input, obs, output, seeds, num_inference_steps, limit, device):
    seeds = [int(s) for s in str(seeds).split(",") if s.strip() != ""]
    out_dir = pathlib.Path(output) if output else pathlib.Path(obs).parent.joinpath("redundancy")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading policy from {input}")
    policy, cfg = load_policy(input, num_inference_steps=num_inference_steps, device=device)
    if len(policy.rgb_keys) == 0 or len(policy.wrench_keys) == 0:
        raise click.UsageError("이 분석은 vision + wrench 를 모두 쓰는 정책에서만 의미가 있습니다.")

    data = load_inference_obs(obs)
    frames = data["obs_by_inference"]
    elapsed_s = data["elapsed_s"]
    if limit and limit > 0:
        frames = frames[:limit]
        elapsed_s = elapsed_s[:limit]
    n = len(frames)
    print(f"  {n} inferences. seeds={seeds}")

    vision_baseline = attr.make_blank_vision(policy)     # 공정: 중립 이미지
    wrench_baseline = attr.make_zero_wrench(policy)      # 공정: 무접촉 0

    rows = []
    for i, obs_np in enumerate(frames):
        obs_dict = attr.obs_np_to_tensor(obs_np, policy.device)
        idl = attr.interaction_deltas(policy, obs_dict, vision_baseline, wrench_baseline, seeds=seeds)
        rows.append(idl)
        t = f"t={float(elapsed_s[i]):.2f}s" if np.isfinite(elapsed_s[i]) else f"#{i}"
        print(f"[{i+1}/{n}] {t}  Δvision={idl.vision:.4f}  Δwrench={idl.wrench:.4f}  "
              f"Δwrench|noV={idl.wrench_given_no_vision:.4f}  redund={idl.redundancy:+.4f}")

    # 집계
    mean_v = float(np.mean([r.vision for r in rows]))
    mean_w = float(np.mean([r.wrench for r in rows]))
    mean_w_cond = float(np.mean([r.wrench_given_no_vision for r in rows]))
    mean_both = float(np.mean([r.both for r in rows]))
    mean_redund = float(np.mean([r.redundancy for r in rows]))

    # CSV
    import csv
    with open(out_dir.joinpath("redundancy.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["inference", "elapsed_s", "vision", "wrench", "both",
                    "wrench_given_no_vision", "vision_given_no_wrench", "redundancy"])
        for i, r in enumerate(rows):
            w.writerow([i, float(elapsed_s[i]), r.vision, r.wrench, r.both,
                        r.wrench_given_no_vision, r.vision_given_no_wrench, r.redundancy])
    _write_png(out_dir.joinpath("redundancy_timeline.png"), elapsed_s, rows)

    print("\n" + "=" * 64)
    print("EPISODE 요약 (mean over inferences)")
    print("=" * 64)
    print(f"  Δvision                = {mean_v:.4f}   (vision 제거 시 action 변화; 규모 기준)")
    print(f"  Δwrench (단일)         = {mean_w:.4f}   (wrench만 제거 — 낮게 나오는 그 값)")
    print(f"  Δwrench | vision 제거  = {mean_w_cond:.4f}   ★ 핵심 판별값")
    print(f"  Δ(vision+wrench 동시)  = {mean_both:.4f}")
    print(f"  redundancy = Δv+Δw−Δboth = {mean_redund:+.4f}   (>0 = 정보 중복)")
    print("-" * 64)
    print("  판정:", _verdict(mean_w, mean_w_cond, scale=mean_v))
    print("=" * 64)


if __name__ == "__main__":
    main()
