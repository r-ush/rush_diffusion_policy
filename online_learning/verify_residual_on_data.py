#!/usr/bin/env python
"""
학습된 residual head 를 held-out 교정 데이터에 대해 검증 (사용자 요청 #4):
"inference 가 잘못된 상황으로 다시 갔을 때, residual policy 가 교정 dataset 쪽으로 보내주는가."

핵심 아이디어(정렬 문제 없음):
  각 프레임의 GT residual = residual_delta6_slow_pred_to_virtual
    = (사람이 실제로 간 곳 achieved) − (slow base 예측)  = 그 상황에서 필요했던 '교정'.
  head 예측 residual 이 이 GT 를 재현하면, "잘못된 상황(=GT 큰 곳)에서 교정 쪽으로 당긴다"는 뜻.

지표:
  * correction 크기 ‖GT_trans‖(m): 클수록 base 가 많이 틀린 '잘못된 상황'.
  * 교정 포착률 = (‖GT‖ − ‖GT − pred‖)/‖GT‖  (1=완벽 재현, 0=아무것도 안 함, <0=반대로).
  * 방향 cosine(pred_trans, GT_trans): 교정 '방향'이 맞는가.
  * 위 지표를 전체 + 'correction 큰 상위 25% 프레임'에서 각각.
그림: pred-vs-GT 산점도, 교정 포착률 히스토그램, 첫 에피소드 per-frame GT/pred 타임라인.

실행(learner 가 학습해 head 를 mailbox 에 발행한 상태에서):
  RESIDUAL_ONLINE_WORKDIR=... <py> online_learning/verify_residual_on_data.py \
    --slow_ckpt <abs ckpt> --config_name residual_policy/hand_online_abs_mlp \
    --heldout /media/.../run_hand/transitions/ep_00030.hdf5 --out data/residual_verify
"""
import os
import sys
import glob
import json
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import h5py
import numpy as np
import torch
import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.residual_policy.create_slow_pred_fast_dataset import load_policy_from_ckpt
from online_learning import config_residual_online as C
from online_learning.mailbox import FileMailbox
from online_learning.residual_relabel_utils import write_residual_episode_hdf5, RAW_OBS_KEYS
from online_learning.replay_feed_episodes import iter_source_episodes, compute_slow_pred_target_abs

OmegaConf.register_new_resolver("eval", eval, replace=True)


def load_fast_policy(config_name, slow_ckpt, device):
    config_dir = os.path.join(ROOT, "diffusion_policy", "config")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
    cfg.task.slow_ckpt_path = slow_ckpt
    OmegaConf.resolve(cfg)
    policy = hydra.utils.instantiate(cfg.policy).to(device).eval()
    return cfg, policy


def load_head_from_mailbox(mailbox, policy, device, version=None):
    v = version if version is not None else mailbox.get_latest_weight_version()
    if v is None:
        raise RuntimeError("mailbox 에 발행된 head 가중치가 없습니다 (learner 를 먼저 학습시키세요).")
    payload = mailbox.load_weights(v, map_location=device)
    policy.head.load_state_dict({k: x.to(device) for k, x in payload["head_state"].items()})
    policy.normalizer.load_state_dict(
        {k: x.to(device) for k, x in payload["normalizer_state"].items()}, strict=False)
    if payload.get("force_encoder_state") is not None and policy.force_encoder is not None:
        policy.force_encoder.load_state_dict({k: x.to(device) for k, x in payload["force_encoder_state"].items()})
    policy.normalizer.to(device)
    print(f"[Verify] head v{v} 로드 (learner demo={payload.get('num_demos','?')})")
    return v


@torch.no_grad()
def collect_gt_pred(fast_policy, slow_policy, heldout_files, cfg, device, tmp_hdf5, batch_size=64):
    """held-out 에피소드 → slow_pred 재계산 → residual HDF5 → dataset → (GT, pred) residual 수집."""
    from diffusion_policy.residual_policy.step_dataset import FastResidualContextStepDataset
    from online_learning.residual_relabel_utils import build_residual_demo
    action_pose_repr = getattr(slow_policy, "action_pose_repr", "abs")

    gts, preds, ep_ids = [], [], []
    ds_cfg = OmegaConf.to_container(cfg.task.dataset, resolve=True)
    ds_cfg.pop("_target_"); ds_cfg["dataset_path"] = tmp_hdf5; ds_cfg["val_ratio"] = 0.0

    src_list = []
    if os.path.isdir(heldout_files):
        src_list = sorted(glob.glob(os.path.join(heldout_files, "*.hdf5")))
    else:
        src_list = [heldout_files]

    for ei, fp in enumerate(src_list):
        for tag, obs in iter_source_episodes(fp):
            if any(k not in obs for k in RAW_OBS_KEYS) or len(obs["robot_pose_R"]) < 2:
                continue
            slow_pred = compute_slow_pred_target_abs(slow_policy, obs, device,
                                                     action_pose_repr=action_pose_repr, batch_size=16)
            episode = {k: obs[k] for k in RAW_OBS_KEYS}
            episode["slow_pred_target_abs"] = slow_pred
            write_residual_episode_hdf5(tmp_hdf5, episode, demo_name="demo_0")

            dataset = FastResidualContextStepDataset(**ds_cfg)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            for batch in loader:
                batch = dict_apply(batch, lambda x: x.to(device))
                pred = fast_policy.predict_action(batch["obs"])["action"][:, 0].cpu().numpy()  # (B,6)
                gt = batch["action"][:, -1].cpu().numpy()                                       # (B,6)
                gts.append(gt); preds.append(pred); ep_ids.append(np.full(len(gt), ei))
            print(f"[Verify] {tag}: {len(dataset)} samples")
    return np.concatenate(gts), np.concatenate(preds), np.concatenate(ep_ids)


def metrics(gt, pred, mask=None):
    if mask is not None:
        gt, pred = gt[mask], pred[mask]
    gt_t, pred_t = gt[:, :3], pred[:, :3]
    gt_norm = np.linalg.norm(gt_t, axis=1) + 1e-9
    capture = (gt_norm - np.linalg.norm(gt_t - pred_t, axis=1)) / gt_norm       # 교정 포착률
    cos = np.sum(gt_t * pred_t, axis=1) / (gt_norm * (np.linalg.norm(pred_t, axis=1) + 1e-9))
    return {
        "n": int(len(gt)),
        "gt_trans_mean_m": float(gt_norm.mean()),
        "capture_ratio_mean": float(capture.mean()),
        "capture_ratio_median": float(np.median(capture)),
        "direction_cosine_mean": float(cos.mean()),
        "frac_capture_pos": float((capture > 0).mean()),
    }


def main():
    ap = argparse.ArgumentParser(description="held-out 교정 데이터로 residual head 검증(#4).")
    ap.add_argument("--slow_ckpt", default=None)
    ap.add_argument("--config_name", default="residual_policy/hand_online_abs_mlp")
    ap.add_argument("--heldout", required=True, help="held-out 에피소드 파일 또는 디렉토리.")
    ap.add_argument("--weight_version", type=int, default=None, help="특정 head 버전(기본 최신).")
    ap.add_argument("--out", default="data/residual_verify")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    slow_ckpt = args.slow_ckpt or C.SLOW_CKPT
    device = torch.device(args.device or (C.DEVICE if torch.cuda.is_available() else "cpu"))

    cfg, fast_policy = load_fast_policy(args.config_name, slow_ckpt, device)
    slow_policy = fast_policy.slow_policy
    slow_policy.num_inference_steps = 12
    slow_policy.n_action_steps = int(slow_policy.horizon) - int(slow_policy.n_obs_steps) + 1
    mailbox = FileMailbox(C.ONLINE_WORKDIR)
    load_head_from_mailbox(mailbox, fast_policy, device, version=args.weight_version)

    tmp = os.path.join(args.out, "_heldout_residual.hdf5")
    gt, pred, ep = collect_gt_pred(fast_policy, slow_policy, args.heldout, cfg, device, tmp)

    gt_norm = np.linalg.norm(gt[:, :3], axis=1)
    hard = gt_norm >= np.quantile(gt_norm, 0.75)   # '잘못된 상황' = 교정 큰 상위 25%
    report = {
        "overall": metrics(gt, pred),
        "hard_top25pct": metrics(gt, pred, hard),
        "slow_only_baseline_note": "capture_ratio=0 이면 residual 이 교정을 전혀 못 함, 1 이면 완벽 재현",
    }
    print("\n===== 검증 결과 =====")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    with open(os.path.join(args.out, "verify_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── 그림 ──
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].scatter(gt[:, 0], pred[:, 0], s=4, alpha=0.3, label="x")
    ax[0].scatter(gt[:, 1], pred[:, 1], s=4, alpha=0.3, label="y")
    ax[0].scatter(gt[:, 2], pred[:, 2], s=4, alpha=0.3, label="z")
    lim = np.abs(gt[:, :3]).max() * 1.05
    ax[0].plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax[0].set_xlabel("GT residual trans (m)"); ax[0].set_ylabel("pred residual trans (m)")
    ax[0].set_title("pred vs GT (대각선=완벽)"); ax[0].legend(); ax[0].axis("equal")

    cap = (gt_norm - np.linalg.norm(gt[:, :3] - pred[:, :3], axis=1)) / (gt_norm + 1e-9)
    ax[1].hist(np.clip(cap, -1, 1), bins=40, color="tab:blue", alpha=0.8)
    ax[1].axvline(0, color="r", ls="--"); ax[1].axvline(1, color="g", ls="--")
    ax[1].set_xlabel("교정 포착률 (1=완벽, 0=무시)"); ax[1].set_title("전체 프레임")

    ep0 = ep == ep.min()
    ax[2].plot(gt_norm[ep0], label="필요한 교정 ‖GT‖", lw=1.5)
    ax[2].plot(np.linalg.norm(pred[ep0][:, :3], axis=1), label="head 예측 ‖pred‖", lw=1.2)
    ax[2].set_xlabel("frame"); ax[2].set_ylabel("trans (m)"); ax[2].set_title("첫 에피소드 타임라인"); ax[2].legend()

    fig.tight_layout()
    png = os.path.join(args.out, "verify_plots.png")
    fig.savefig(png, dpi=120)
    print(f"\n[Verify] 그림 저장: {png}")
    print(f"[Verify] 리포트: {os.path.join(args.out, 'verify_report.json')}")


if __name__ == "__main__":
    main()
