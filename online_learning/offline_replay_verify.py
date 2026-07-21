#!/usr/bin/env python
"""
로봇 없이, 옛 교정 데이터로 residual-online DAgger 전체를 한 번에 돌려 #4 검증까지.

  train 소스(run_hand) 에피소드를 하나씩 → slow_pred 재계산 → residual learner 에 넣고
  warm-continue 학습(가중치 발행) → 다 돌면 held-out 소스(run_hand2)로 head 예측 vs
  GT 교정(residual) 비교 → report(JSON) + plots(PNG).

replay_feed_episodes.py(feeder) + residual_teleop_learner.py + verify_residual_on_data.py 의
로직을 in-process 로 묶은 오케스트레이터(별도 터미널 3개 대신 한 번에).

실행(백그라운드 권장):
  RESIDUAL_SLOW_CKPT=<abs ckpt> RESIDUAL_ONLINE_WORKDIR=<wd> \
  RESIDUAL_CONFIG_NAME=residual_policy/hand_online_abs_mlp \
  <py> online_learning/offline_replay_verify.py \
     --train_source <SSD>/online_runs/run_hand/transitions \
     --heldout_source <SSD>/online_runs/run_hand2/transitions \
     --out data/residual_verify_abs
"""
import os
import sys
import glob
import json
import time
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply
from online_learning import config_residual_online as C
from online_learning.replay_feed_episodes import iter_source_episodes, compute_slow_pred_target_abs
from online_learning.residual_relabel_utils import write_residual_episode_hdf5, RAW_OBS_KEYS
from online_learning.verify_residual_on_data import metrics


def _episodes(source):
    files = sorted(glob.glob(os.path.join(source, "*.hdf5"))) if os.path.isdir(source) else [source]
    for fp in files:
        for tag, obs in iter_source_episodes(fp):
            yield tag, obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_source", required=True)
    ap.add_argument("--heldout_source", required=True)
    ap.add_argument("--out", default="data/residual_verify_abs")
    ap.add_argument("--num_inference_steps", type=int, default=8)
    ap.add_argument("--max_train_episodes", type=int, default=0)
    ap.add_argument("--first_epochs", type=int, default=100)
    ap.add_argument("--epochs_per_round", type=int, default=30)
    ap.add_argument("--max_samples_per_epoch", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    C.FIRST_EPOCHS = args.first_epochs
    C.EPOCHS_PER_ROUND = args.epochs_per_round
    C.MAX_SAMPLES_PER_EPOCH = args.max_samples_per_epoch
    C.MIN_EPISODES_BEFORE_TRAIN = 2

    from online_learning.residual_teleop_learner import ResidualOnlineLearner
    learner = ResidualOnlineLearner()
    slow = learner.policy.slow_policy
    slow.num_inference_steps = args.num_inference_steps
    slow.n_action_steps = int(slow.horizon) - int(slow.n_obs_steps) + 1
    apr = getattr(slow, "action_pose_repr", "abs")
    dev = learner.device
    t0 = time.time()

    # ── train: 에피소드 하나씩 warm-continue ──
    n_fed = 0
    for tag, obs in _episodes(args.train_source):
        if any(k not in obs for k in RAW_OBS_KEYS) or len(obs["robot_pose_R"]) < 2:
            print(f"[skip] {tag}", flush=True); continue
        sp = compute_slow_pred_target_abs(slow, obs, dev, action_pose_repr=apr, batch_size=24)
        ep = {k: obs[k] for k in RAW_OBS_KEYS}; ep["slow_pred_target_abs"] = sp
        p = write_residual_episode_hdf5(os.path.join(C.ONLINE_WORKDIR, "feed.hdf5"), ep, "demo_0")
        learner.add_episode(p)
        n_fed += 1
        if learner.num_demos >= C.MIN_EPISODES_BEFORE_TRAIN:
            learner.train_round()
        print(f"[train] fed={n_fed} demos={learner.num_demos} elapsed={time.time()-t0:.0f}s", flush=True)
        if args.max_train_episodes and n_fed >= args.max_train_episodes:
            break
    print(f">>> 학습 완료: {n_fed} 에피소드, head v{learner.version-1}, {time.time()-t0:.0f}s", flush=True)

    # ── verify: held-out ──
    from diffusion_policy.residual_policy.step_dataset import FastResidualContextStepDataset
    ds_cfg = OmegaConf.to_container(learner.cfg.task.dataset, resolve=True); ds_cfg.pop("_target_")
    ds_cfg["val_ratio"] = 0.0
    learner.policy.eval()
    gts, preds, epids = [], [], []
    ei = 0
    for tag, obs in _episodes(args.heldout_source):
        if any(k not in obs for k in RAW_OBS_KEYS) or len(obs["robot_pose_R"]) < 2:
            continue
        sp = compute_slow_pred_target_abs(slow, obs, dev, action_pose_repr=apr, batch_size=24)
        ep = {k: obs[k] for k in RAW_OBS_KEYS}; ep["slow_pred_target_abs"] = sp
        tmp = os.path.join(args.out, "_held.hdf5")
        write_residual_episode_hdf5(tmp, ep, "demo_0")
        ds_cfg["dataset_path"] = tmp
        ds = FastResidualContextStepDataset(**ds_cfg)
        with torch.no_grad():
            for b in torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False):
                b = dict_apply(b, lambda x: x.to(dev))
                preds.append(learner.policy.predict_action(b["obs"])["action"][:, 0].cpu().numpy())
                gts.append(b["action"][:, -1].cpu().numpy())
                epids.append(np.full(len(gts[-1]), ei))
        print(f"[verify] {tag}: {len(ds)} samples", flush=True)
        ei += 1
    gt = np.concatenate(gts); pred = np.concatenate(preds); ep = np.concatenate(epids)

    gn = np.linalg.norm(gt[:, :3], axis=1)
    hard = gn >= np.quantile(gn, 0.75)
    report = {"train_episodes": n_fed, "heldout_samples": int(len(gt)),
              "overall": metrics(gt, pred), "hard_top25pct": metrics(gt, pred, hard)}
    print("\n===== #4 검증 =====\n" + json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    with open(os.path.join(args.out, "verify_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, c in zip(range(3), "xyz"):
        ax[0].scatter(gt[:, i], pred[:, i], s=4, alpha=0.3, label=c)
    lim = float(np.abs(gt[:, :3]).max()) * 1.05
    ax[0].plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax[0].set_xlabel("GT residual trans (m)"); ax[0].set_ylabel("pred (m)")
    ax[0].set_title("pred vs GT (대각선=완벽)"); ax[0].legend(); ax[0].axis("equal")
    cap = (gn - np.linalg.norm(gt[:, :3] - pred[:, :3], axis=1)) / (gn + 1e-9)
    ax[1].hist(np.clip(cap, -1, 1), bins=40); ax[1].axvline(0, color="r", ls="--"); ax[1].axvline(1, color="g", ls="--")
    ax[1].set_xlabel("교정 포착률"); ax[1].set_title("전체 프레임")
    m0 = ep == 0
    ax[2].plot(gn[m0], label="필요 교정 ‖GT‖"); ax[2].plot(np.linalg.norm(pred[m0][:, :3], axis=1), label="head ‖pred‖")
    ax[2].set_xlabel("frame"); ax[2].set_ylabel("trans (m)"); ax[2].legend(); ax[2].set_title("첫 held-out 에피소드")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "verify_plots.png"), dpi=120)
    print(f"[done] {os.path.join(args.out,'verify_report.json')} / verify_plots.png", flush=True)


if __name__ == "__main__":
    main()
