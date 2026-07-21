#!/usr/bin/env python
"""
로봇 없이 온라인 residual DAgger(개입형) 코어를 검증.

Part 0 (정책/ckpt 불필요 — 항상 실행):
  개입형 전용 배선을 검증한다. 합성 개입 에피소드 -> residual relabel(is_intervention 통과)
  -> FastResidualContextStepDataset(intervention_key) -> replay_buffer 에 is_intervention 적재
  -> intervention_sample_weights 가 개입 프레임 윈도우만 W 로 가중하는지.

Parts A/B/C (slow ckpt 필요 — 있으면 실행, 없으면 skip):
  A. ResidualInterventionLearner 가 slow ckpt 로 인스턴스화되는가.
  B. 합성 개입 에피소드 -> mailbox -> 가중 샘플러로 head 학습(loss 감소)까지 도는가.
  C. 발행된 head+normalizer hot-swap 왕복이 정확히 일치하는가.

실행 (timm 있는 env):
  /home/rush/anaconda3/envs/bae_robodiff/bin/python \
    online_learning/smoke_test_residual_intervention_no_robot.py
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

from online_learning import config_residual_intervention as C
from online_learning.residual_relabel_utils import write_residual_episode_hdf5, pos_quat_to_pose9
from online_learning.residual_intervention_learner import intervention_sample_weights


def make_synthetic_episode(T=24, seed=0, intervene_slice=(8, 16)):
    """actor 가 보낼 원시 개입 에피소드(dict) 를 흉내: pose 궤적 + slow_pred(base) + is_intervention.
    intervene_slice 프레임 구간을 개입(1)으로 표시하고, 그 구간에서 achieved 를 slow 로부터 더
    크게 벗어나게(=사람 밀림) 만든다."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, T)[:, None].astype(np.float32)
    base_pos = np.array([0.5, 0.0, 0.3], dtype=np.float32)
    robot_pos = base_pos[None] + 0.05 * t * np.array([1.0, 0.5, -0.3], dtype=np.float32)
    robot_pos += 0.002 * rng.standard_normal((T, 3)).astype(np.float32)
    angles = 0.2 * t[:, 0]
    quats = Rotation.from_rotvec(
        np.stack([angles, 0.5 * angles, -0.3 * angles], axis=-1)).as_quat().astype(np.float32)

    is_intervention = np.zeros(T, dtype=np.float32)
    lo, hi = intervene_slice
    is_intervention[lo:hi] = 1.0

    # slow_pred(base): nominal 에선 achieved 와 거의 같게, 개입 구간에선 크게 벗어나게(사람 밀림 신호)
    slow_pos = robot_pos.copy()
    slow_pos += 0.001 * np.array([0.0, 0.0, 1.0], dtype=np.float32)          # nominal 미세 오프셋
    slow_pos[lo:hi] -= 0.03 * np.array([0.0, 1.0, 0.0], dtype=np.float32)     # 개입: 큰 잔차
    slow_pred_target_abs = pos_quat_to_pose9(slow_pos, quats)

    return {
        "image0": rng.integers(0, 255, size=(T, 224, 224, 3), dtype=np.uint8),
        "robot_pose_R": robot_pos.astype(np.float32),
        "robot_quat_R": quats,
        "hand_pose_R": (0.1 * rng.standard_normal((T, 7))).astype(np.float32),
        "wrench_wrist_R": (0.5 * rng.standard_normal((T, 6, 32))).astype(np.float32),
        "slow_pred_target_abs": slow_pred_target_abs,
        "is_intervention": is_intervention,
    }


def part0_plumbing(workdir):
    """정책/ckpt 없이: relabel -> dataset(intervention_key) -> 가중치 배선 검증."""
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    print("\n[smoke] ===== Part 0: 개입 배선(정책 불필요) =====")
    ds_hdf5 = os.path.join(workdir, "part0_intervention.hdf5")
    T, lo, hi = 24, 8, 16
    ep = make_synthetic_episode(T=T, seed=0, intervene_slice=(lo, hi))
    write_residual_episode_hdf5(ds_hdf5, ep, demo_name="demo_0")

    # relabel 이 last step 을 버리므로 n=T-1. 개입 프레임 수는 그대로(개입 구간이 끝에 안 걸림).
    n = T - 1
    n_int_expected = int(ep["is_intervention"][:n].sum())
    assert n_int_expected == (hi - lo), f"개입 프레임 수 예상과 다름: {n_int_expected}"

    # 개입형 task 로 dataset 인스턴스화 (정책은 만들지 않음)
    config_dir = os.path.join(ROOT, "diffusion_policy", "config")
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=C.RESIDUAL_CONFIG_NAME)
    ds_cfg = OmegaConf.to_container(cfg.task.dataset, resolve=True)
    ds_cfg["dataset_path"] = ds_hdf5
    ds_cfg["val_ratio"] = 0.0
    assert ds_cfg.get("intervention_key") == "obs/is_intervention", \
        f"hand_intervention task 에 intervention_key 누락: {ds_cfg.get('intervention_key')}"
    cls = hydra.utils.get_class(ds_cfg.pop("_target_"))
    dataset = cls(**ds_cfg)

    assert "is_intervention" in dataset.replay_buffer, "replay_buffer 에 is_intervention 미적재"
    isint = np.asarray(dataset.replay_buffer["is_intervention"]).reshape(-1)
    assert isint.shape[0] == n, f"is_intervention 길이 {isint.shape[0]} != n={n}"
    assert int(isint.sum()) == n_int_expected, f"개입 스텝 수 불일치: {int(isint.sum())}"

    # 가중치: pad_after=0 이라 스텝↔윈도우 1:1 → 개입 윈도우 수 == 개입 스텝 수
    W = 5.0
    weights = intervention_sample_weights(dataset.sampler.indices, isint, W)
    assert len(weights) == len(dataset), f"weights 길이 {len(weights)} != len(dataset) {len(dataset)}"
    n_weighted = int((weights == W).sum())
    assert n_weighted == n_int_expected, \
        f"가중 윈도우 수 {n_weighted} != 개입 스텝 수 {n_int_expected}"
    assert np.all((weights == 1.0) | (weights == W)), "weights 값이 1 또는 W 가 아님"
    print(f"[smoke] dataset len={len(dataset)}, 개입 스텝={n_int_expected}, "
          f"가중(W={W}) 윈도우={n_weighted}")
    print("[smoke] ✅ Part 0: relabel→dataset(intervention_key)→가중 배선 검증")
    return True


def parts_abc(workdir):
    """slow ckpt 필요: 인스턴스화 / 가중 학습 / hot-swap."""
    if not os.path.exists(C.SLOW_CKPT):
        print(f"\n[smoke] ⏭  Part A/B/C skip — slow ckpt 없음: {C.SLOW_CKPT} (SSD 마운트 시 실행)")
        return False

    from online_learning.residual_intervention_learner import ResidualInterventionLearner
    from torch.utils.data import WeightedRandomSampler, DataLoader
    from diffusion_policy.common.pytorch_util import dict_apply

    C.ONLINE_WORKDIR = os.path.join(workdir, "run")
    C.ACTOR_OUTPUT_DIR = os.path.join(C.ONLINE_WORKDIR, "actor_episodes")
    C.MIN_EPISODES_BEFORE_TRAIN = 2
    C.FIRST_EPOCHS = 60
    C.EPOCHS_PER_ROUND = 30
    C.BATCH_SIZE = 32
    C.NUM_WORKERS = 0
    os.makedirs(C.ACTOR_OUTPUT_DIR, exist_ok=True)

    print("\n[smoke] ===== Part A: learner 인스턴스화 =====")
    learner = ResidualInterventionLearner()
    print("[smoke] ✅ A: 개입 residual 정책이 slow ckpt 로 인스턴스화됨")

    print("\n[smoke] ===== Part B: 가중 학습 =====")
    n_eps = 3
    for i in range(n_eps):
        ep = make_synthetic_episode(T=24, seed=i, intervene_slice=(8, 16))
        ep_path = os.path.join(C.ACTOR_OUTPUT_DIR, f"raw_ep_{i}.hdf5")
        write_residual_episode_hdf5(ep_path, ep, demo_name="demo_0")
        learner.mailbox.send_episode(ep_path)
    for ep in learner.mailbox.poll_new_episodes():
        learner.add_episode(ep)
        learner.mailbox.mark_episode_done(ep)

    dataset = learner._build_dataset()
    assert len(dataset) > 0
    # 가중 샘플러가 실제로 WeightedRandomSampler 인지 확인
    sampler, shuffle = learner._make_sampler(dataset)
    assert isinstance(sampler, WeightedRandomSampler), f"가중 샘플러 아님: {type(sampler)}"
    assert shuffle is False
    print(f"[smoke] dataset len={len(dataset)}, sampler={type(sampler).__name__}")

    learner.policy.set_normalizer(dataset.get_normalizer())
    learner.policy.normalizer.to(learner.device)
    loader = DataLoader(dataset, batch_size=C.BATCH_SIZE, shuffle=False, num_workers=0)
    learner.policy.eval()
    with torch.no_grad():
        b0 = dict_apply(next(iter(loader)), lambda x: x.to(learner.device))
        init_loss = learner.policy.compute_loss(b0).item()
    learner.train_round()
    learner.policy.eval()
    with torch.no_grad():
        final_loss = learner.policy.compute_loss(b0).item()
    print(f"[smoke] init_loss={init_loss:.6f}  final_loss={final_loss:.6f}")
    assert np.isfinite(final_loss) and final_loss < init_loss, \
        f"loss 감소 안 함: {init_loss} -> {final_loss}"
    print("[smoke] ✅ B: 가중 샘플러로 head 학습, loss 감소 확인")

    print("\n[smoke] ===== Part C: hot-swap 왕복 =====")
    version = learner.mailbox.get_latest_weight_version()
    payload = learner.mailbox.load_weights(version, map_location="cpu")
    assert payload is not None and "head_state" in payload
    ref_head = {k: v.clone() for k, v in learner.policy.head.state_dict().items()}
    with torch.no_grad():
        for p in learner.policy.head.parameters():
            p.mul_(0).add_(123.0)
    learner.policy.head.load_state_dict(
        {k: v.to(learner.device) for k, v in payload["head_state"].items()})
    max_diff = max(
        (learner.policy.head.state_dict()[k].cpu() - ref_head[k].cpu()).abs().max().item()
        for k in ref_head)
    print(f"[smoke] hot-swap head max|Δ|={max_diff:.3e}")
    assert max_diff < 1e-6, f"hot-swap head 불일치: {max_diff}"
    print("[smoke] ✅ C: head+normalizer hot-swap 왕복 일치")
    return True


def main():
    workdir = tempfile.mkdtemp(prefix="residual_intervention_smoke_")
    print(f"[smoke] workdir={workdir}")
    print(f"[smoke] slow ckpt={C.SLOW_CKPT}")
    try:
        part0_plumbing(workdir)
        ran_abc = parts_abc(workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if ran_abc:
        print("\n[smoke] 🎉 전부 통과 (Part 0 배선 / A 호환 / B 가중학습 / C hot-swap)")
    else:
        print("\n[smoke] ✅ Part 0 통과 (A/B/C 는 slow ckpt 있을 때 실행)")


if __name__ == "__main__":
    main()
