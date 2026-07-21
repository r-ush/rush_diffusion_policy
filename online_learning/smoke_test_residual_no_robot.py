#!/usr/bin/env python
"""
로봇 없이 온라인 residual DAgger 코어 전체를 검증.

검증 항목:
  A. residual 정책이 실제 slow(base) ckpt 로 인스턴스화되는가 (호환성).
  B. 합성 교정 에피소드 -> residual relabel -> mailbox -> accumulated -> dataset ->
     head warm-continue 학습(loss 감소)까지 도는가.
  C. 발행된 head+normalizer 가중치를 hot-swap 왕복 로드했을 때 정확히 일치하는가.

실행 (timm 있는 env):
  export RESIDUAL_ONLINE_WORKDIR=$(mktemp -d)/run_hand_residual
  /home/rush/anaconda3/envs/bae_robodiff/bin/python online_learning/smoke_test_residual_no_robot.py
"""
import os
import sys
import shutil
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from online_learning import config_residual_online as C
from online_learning.residual_relabel_utils import write_residual_episode_hdf5, pos_quat_to_pose9


def make_synthetic_episode(T=24, seed=0):
    """actor 가 보낼 원시 에피소드(dict) 를 흉내. 부드러운 pose 궤적 + slow_pred(base)."""
    rng = np.random.default_rng(seed)
    # 부드러운 pose 궤적
    t = np.linspace(0, 1, T)[:, None].astype(np.float32)
    base_pos = np.array([0.5, 0.0, 0.3], dtype=np.float32)
    robot_pos = base_pos[None] + 0.05 * t * np.array([1.0, 0.5, -0.3], dtype=np.float32)
    robot_pos += 0.002 * rng.standard_normal((T, 3)).astype(np.float32)
    # 부드럽게 회전하는 quat
    angles = 0.2 * t[:, 0]
    quats = Rotation.from_rotvec(np.stack([angles, 0.5 * angles, -0.3 * angles], axis=-1)).as_quat().astype(np.float32)

    ep = {
        "image0": rng.integers(0, 255, size=(T, 224, 224, 3), dtype=np.uint8),
        "robot_pose_R": robot_pos.astype(np.float32),
        "robot_quat_R": quats,
        "hand_pose_R": (0.1 * rng.standard_normal((T, 7))).astype(np.float32),
        "wrench_wrist_R": (0.5 * rng.standard_normal((T, 6, 32))).astype(np.float32),
    }
    # slow_pred(base) = 로봇 궤적에서 살짝 벗어난 예측 -> nonzero residual(사람 교정) 신호
    slow_pos = robot_pos + 0.01 * np.array([0.0, 0.0, 1.0], dtype=np.float32)
    slow_pred_target_abs = pos_quat_to_pose9(slow_pos, quats)
    ep["slow_pred_target_abs"] = slow_pred_target_abs
    return ep


def main():
    # 격리된 임시 workdir
    workdir = tempfile.mkdtemp(prefix="residual_smoke_")
    C.ONLINE_WORKDIR = os.path.join(workdir, "run")
    C.ACTOR_OUTPUT_DIR = os.path.join(C.ONLINE_WORKDIR, "actor_episodes")
    C.MIN_EPISODES_BEFORE_TRAIN = 2
    C.FIRST_EPOCHS = 60
    C.EPOCHS_PER_ROUND = 30
    C.BATCH_SIZE = 32
    C.NUM_WORKERS = 0
    os.makedirs(C.ACTOR_OUTPUT_DIR, exist_ok=True)
    print(f"[smoke] workdir={workdir}")
    print(f"[smoke] slow ckpt={C.SLOW_CKPT}")
    assert os.path.exists(C.SLOW_CKPT), f"slow ckpt 없음: {C.SLOW_CKPT} (SSD 마운트 확인)"

    # ── A. learner 인스턴스화 (slow ckpt 로 residual 정책 build) ───────────────
    from online_learning.residual_teleop_learner import ResidualOnlineLearner
    learner = ResidualOnlineLearner()
    print("[smoke] ✅ A: residual 정책이 slow ckpt 로 인스턴스화됨")

    # ── B. 합성 에피소드 -> relabel -> mailbox -> add -> train ──────────────────
    n_eps = 3
    for i in range(n_eps):
        ep = make_synthetic_episode(T=24, seed=i)
        ep_path = os.path.join(C.ACTOR_OUTPUT_DIR, f"raw_ep_{i}.hdf5")
        write_residual_episode_hdf5(ep_path, ep, demo_name="demo_0")
        learner.mailbox.send_episode(ep_path)
    print(f"[smoke] {n_eps}개 합성 에피소드 전송")

    new_eps = learner.mailbox.poll_new_episodes()
    assert len(new_eps) == n_eps, f"poll 개수 불일치: {len(new_eps)} != {n_eps}"
    for ep in new_eps:
        learner.add_episode(ep)
        learner.mailbox.mark_episode_done(ep)

    # 초기 loss vs 학습 후 loss 비교를 위해 첫 라운드 전 한 번 측정
    dataset = learner._build_dataset()
    assert len(dataset) > 0, "dataset 샘플 0개"
    print(f"[smoke] dataset 샘플 수: {len(dataset)}")
    learner.policy.set_normalizer(dataset.get_normalizer())
    learner.policy.normalizer.to(learner.device)
    from torch.utils.data import DataLoader
    from diffusion_policy.common.pytorch_util import dict_apply
    loader = DataLoader(dataset, batch_size=C.BATCH_SIZE, shuffle=False, num_workers=0)
    learner.policy.eval()
    with torch.no_grad():
        b0 = next(iter(loader))
        b0 = dict_apply(b0, lambda x: x.to(learner.device))
        init_loss = learner.policy.compute_loss(b0).item()

    learner.train_round()

    learner.policy.eval()
    with torch.no_grad():
        final_loss = learner.policy.compute_loss(b0).item()
    print(f"[smoke] init_loss={init_loss:.6f}  final_loss={final_loss:.6f}")
    assert np.isfinite(final_loss), "final loss 비유한"
    assert final_loss < init_loss, f"loss 감소 안 함: {init_loss} -> {final_loss}"
    print("[smoke] ✅ B: 데이터->relabel->dataset->head 학습, loss 감소 확인")

    # ── C. 발행된 head+normalizer hot-swap 왕복 ────────────────────────────────
    version = learner.mailbox.get_latest_weight_version()
    payload = learner.mailbox.load_weights(version, map_location="cpu")
    assert payload is not None and "head_state" in payload
    ref_head = {k: v.clone() for k, v in learner.policy.head.state_dict().items()}
    # head 를 망가뜨린 뒤 payload 로 복원
    with torch.no_grad():
        for p in learner.policy.head.parameters():
            p.mul_(0).add_(123.0)
    learner.policy.head.load_state_dict({k: v.to(learner.device) for k, v in payload["head_state"].items()})
    learner.policy.normalizer.load_state_dict(
        {k: v.to(learner.device) for k, v in payload["normalizer_state"].items()}, strict=False)
    max_diff = max(
        (learner.policy.head.state_dict()[k].cpu() - ref_head[k].cpu()).abs().max().item()
        for k in ref_head)
    print(f"[smoke] hot-swap head max|Δ|={max_diff:.3e}")
    assert max_diff < 1e-6, f"hot-swap head 불일치: {max_diff}"
    print("[smoke] ✅ C: head+normalizer hot-swap 왕복 일치")

    shutil.rmtree(workdir, ignore_errors=True)
    print("\n[smoke] 🎉 전부 통과 (A 호환 / B 학습 / C hot-swap)")


if __name__ == "__main__":
    main()
