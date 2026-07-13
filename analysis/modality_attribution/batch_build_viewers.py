#!/usr/bin/env python
"""새 에피소드 attribution 뷰어를 일괄/증분 생성.

<run_dir>/eval_debug/episode_XXXXXX_infer_obs.hdf5 들을 스캔해서, 아직 뷰어가 없는
에피소드만 build_attribution_viewer 로 생성한다. (이미 있으면 스킵 → "새 에피소드만" 처리)

각 에피소드는 별도 프로세스로 build_attribution_viewer 를 호출한다(체크포인트를 매번 로드해서
느리지만 단순·견고). --loop 를 주면 주기적으로 폴링해 새 덤프가 생길 때마다 자동 생성.

사용 예:
  # 한 번: 없는 것만 생성
  python -m analysis.modality_attribution.batch_build_viewers \
      -i data/outputs/260710_insert_box_hand_wrench_abs/epoch=0900-train_loss=0.001.ckpt

  # 감시 모드: 30초마다 새 에피소드 뷰어 자동 생성 (Ctrl+C로 종료)
  python -m analysis.modality_attribution.batch_build_viewers -i <ckpt> --loop 30
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
import pathlib

import click


def _episode_num(path: pathlib.Path):
    m = re.search(r"episode_(\d+)_infer_obs", path.name)
    return m.group(1) if m else None


def _scan_and_build(ckpt, run_dir, grid, seeds, frames, vision_baseline, force, device):
    run = pathlib.Path(run_dir)
    eval_debug = run / "eval_debug"
    obs_files = sorted(eval_debug.glob("episode_*_infer_obs.hdf5"))
    if not obs_files:
        print(f"[batch] infer_obs 덤프 없음: {eval_debug}")
        return [], []

    built, skipped = [], []
    for obs in obs_files:
        ep = _episode_num(obs)
        if ep is None:
            continue
        out_html = run / f"attribution_ep{ep}" / "viewer.html"
        if out_html.exists() and not force:
            skipped.append(ep)
            continue
        print(f"\n[batch] ===== episode {ep} 뷰어 생성 =====")
        cmd = [
            sys.executable, "-m", "analysis.modality_attribution.build_attribution_viewer",
            "-i", str(ckpt), "--obs", str(obs), "-o", str(out_html),
            "--grid", str(grid), "--seeds", seeds, "--frames", frames,
            "--vision_baseline", vision_baseline, "--device", device,
        ]
        r = subprocess.run(cmd)
        (built if r.returncode == 0 else skipped).append(ep + ("" if r.returncode == 0 else "(실패)"))
    return built, skipped


@click.command()
@click.option("--input", "-i", required=True, help="Path to checkpoint")
@click.option("--run_dir", default="data/online_runs/run_hand/actor_episodes", show_default=True,
              help="actor_episodes 폴더 (eval_debug/ 아래 infer_obs 스캔, attribution_epXXXXXX/ 에 뷰어 저장)")
@click.option("--grid", default=8, type=int, show_default=True)
@click.option("--seeds", default="0,1", show_default=True)
@click.option("--frames", default="all", show_default=True)
@click.option("--vision_baseline", type=click.Choice(["mean", "self", "start"]), default="mean",
              show_default=True)
@click.option("--force", is_flag=True, help="이미 뷰어가 있어도 다시 생성.")
@click.option("--loop", default=0, type=int,
              help="N초마다 폴링해 새 에피소드 자동 생성(0=한 번만).")
@click.option("--device", default="cuda", show_default=True)
def main(input, run_dir, grid, seeds, frames, vision_baseline, force, loop, device):
    while True:
        built, skipped = _scan_and_build(input, run_dir, grid, seeds, frames,
                                         vision_baseline, force, device)
        print(f"\n[batch] 생성: {built or '-'}   스킵(이미 있음): {skipped or '-'}")
        if loop <= 0:
            break
        print(f"[batch] {loop}s 후 다시 스캔... (Ctrl+C 종료)")
        try:
            time.sleep(loop)
        except KeyboardInterrupt:
            print("\n[batch] 종료.")
            break


if __name__ == "__main__":
    main()
