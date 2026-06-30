import argparse
import json
from pathlib import Path

import dill
import torch


def parse_last_log(log_path: Path):
    if not log_path.exists():
        return None, None, None
    last_epoch = None
    last_step = None
    last_loss = None
    with log_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            last_epoch = obj.get("epoch", last_epoch)
            last_step = obj.get("global_step", last_step)
            last_loss = obj.get("train_loss", last_loss)
    return last_epoch, last_step, last_loss


def parse_latest_ckpt(ckpt_path: Path):
    if not ckpt_path.exists():
        return None, None
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
    pickles = payload.get("pickles", {})
    epoch = dill.loads(pickles["epoch"]) if "epoch" in pickles else None
    step = dill.loads(pickles["global_step"]) if "global_step" in pickles else None
    return epoch, step


def score(epoch, step):
    return (epoch if epoch is not None else -1, step if step is not None else -1)


def status_tag(log_epoch, log_step, ckpt_epoch, ckpt_step, has_ckpt):
    if log_epoch is None and not has_ckpt:
        return "EMPTY"
    if not has_ckpt:
        return "LOG_ONLY"
    if (ckpt_epoch, ckpt_step) == (log_epoch, log_step):
        return "RESUMABLE"
    return "STALE_CKPT"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", help="e.g. outputs/2026.04.19")
    parser.add_argument("--sort", choices=["name", "progress"], default="progress")
    args = parser.parse_args()

    root = Path(args.run_root)
    run_dirs = [p for p in root.iterdir() if p.is_dir()]

    rows = []
    for run_dir in run_dirs:
        latest_ckpt = run_dir / "checkpoints" / "latest.ckpt"
        log_epoch, log_step, log_loss = parse_last_log(run_dir / "logs.json.txt")
        ck_epoch, ck_step = parse_latest_ckpt(latest_ckpt)
        has_ckpt = latest_ckpt.exists()
        rows.append(
            {
                "run": run_dir.name,
                "log_epoch": log_epoch,
                "log_step": log_step,
                "log_loss": log_loss,
                "ckpt_epoch": ck_epoch,
                "ckpt_step": ck_step,
                "has_ckpt": has_ckpt,
                "status": status_tag(log_epoch, log_step, ck_epoch, ck_step, has_ckpt),
                "latest_ckpt": str(latest_ckpt) if has_ckpt else None,
            }
        )

    if args.sort == "progress":
        rows.sort(key=lambda r: score(r["log_epoch"], r["log_step"]), reverse=True)
    else:
        rows.sort(key=lambda r: r["run"])

    print("status | run | log_epoch | log_step | log_loss | ckpt_epoch | ckpt_step | has_latest_ckpt")
    for r in rows:
        print(
            f"{r['status']} | {r['run']} | {r['log_epoch']} | {r['log_step']} | {r['log_loss']} | "
            f"{r['ckpt_epoch']} | {r['ckpt_step']} | {r['has_ckpt']}"
        )

    if rows:
        best_log = rows[0]
        resumable_rows = [r for r in rows if r["has_ckpt"]]
        best_resume = None
        if resumable_rows:
            best_resume = max(
                resumable_rows,
                key=lambda r: score(r["ckpt_epoch"], r["ckpt_step"])
            )

        print("\nBest-by-log-progress:")
        print(best_log["run"])
        if best_resume is not None:
            print("\nBest-resume-checkpoint:")
            print(best_resume["latest_ckpt"])


if __name__ == "__main__":
    main()
