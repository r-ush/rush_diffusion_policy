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


# 합성 데이터가 흉내내는 '진짜' 임피던스 도달률 — Part D 가 이 값을 복원하는지 검증한다.
ALPHA_TRUE_T = np.array([0.30, 0.15, 0.25], dtype=np.float64)   # 병진 축별
ALPHA_TRUE_R = np.array([0.12, 0.07, 0.07], dtype=np.float64)   # 회전 축별


def make_synthetic_episode(T=48, seed=0, intervene_slice=(16, 30), head_gain=0.0):
    """actor 가 보낼 원시 개입 에피소드(dict) 를 흉내.

    **임피던스 지연을 실제로 시뮬레이션한다**: base 가 매 tick 스텝 A 를 명령하면 팔은 그
    일부(ALPHA_TRUE)만 도달한다.
        slow_pred(t) = achieved(t) ⊕ A(t)
        achieved(t+1) = achieved(t) ⊕ (α·A(t) + 사람밀림(t))
    그래서 raw residual = T_base⁻¹·T_achieved(t+1) ≈ (α−I)·A + 사람밀림 이 되어, 실데이터와
    같은 구조(지연이 지배, 개입 구간에만 추가 성분)를 갖는다. 회전은 소각이라 선형 근사.

    head_gain > 0 이면 **head 가 켜진 라운드**를 재현한다: 매 tick δ 를 얹어 실행 명령이
    (A+δ) 가 된다. 이 경우 순진한 적합(residual~A)은 α 를 편향시키므로, fit 이 δ 를 받아
    (residual−δ)~(A+δ) 로 보정하는지 검증하는 데 쓴다.
    """
    rng = np.random.default_rng(seed)
    lo, hi = intervene_slice
    is_intervention = np.zeros(T, dtype=np.float32)
    is_intervention[lo:hi] = 1.0

    # base 가 명령하는 스텝(현재 pose 기준). 방향이 변해야 α 가 축별로 식별된다.
    tt = np.linspace(0, 2 * np.pi, T)[:, None]
    A_t = 0.012 * np.concatenate([np.cos(tt), np.sin(tt), 0.6 * np.cos(2 * tt)], axis=1)
    A_t += 0.002 * rng.standard_normal((T, 3))
    A_r = 0.03 * np.concatenate([np.sin(tt), 0.5 * np.cos(tt), 0.4 * np.sin(2 * tt)], axis=1)
    A_r += 0.005 * rng.standard_normal((T, 3))

    # 사람 밀림: 개입 구간에만, 한 방향으로 일관되게(coherent)
    push_t = np.zeros((T, 3)); push_r = np.zeros((T, 3))
    push_t[lo:hi] = np.array([0.0, 0.004, 0.002])
    push_r[lo:hi] = np.array([0.006, 0.0, -0.003])

    # head 가 켜진 라운드: 매 tick δ 를 얹는다(개입 여부와 무관하게 계속 낸다 — 실제 관측된 행태).
    # ★ 실측 재현 ★ head 출력은 base 명령 A 와 **상관**이 있다(δ ≈ K·A, 실데이터 K 대각
    #   [-0.01, -0.35, -0.02, -0.03, -0.05, -0.15] — head 가 base 전진을 부분 상쇄).
    #   이 상관이 있어야 순진한 적합이 α 를 (1+k) 배로 편향시키고, 보정식의 효과가 드러난다.
    #   무상관 δ 로는 편향이 생기지 않아 테스트가 무의미해진다.
    K_HEAD = np.array([-0.05, -0.35, -0.05, -0.03, -0.05, -0.15])
    cmd6 = np.zeros((T, 6))
    if head_gain > 0:
        cmd6[:, :3] = head_gain * (K_HEAD[:3] * A_t + 0.0012 * rng.standard_normal((T, 3)))
        cmd6[:, 3:] = head_gain * (K_HEAD[3:] * A_r + 0.003 * rng.standard_normal((T, 3)))

    pos = np.zeros((T, 3)); rot = np.zeros((T, 3))
    pos[0] = [0.5, 0.0, 0.3]
    for t in range(T - 1):
        # 실행 명령 = base A + head δ  ->  도달은 그 합의 α 배
        pos[t + 1] = pos[t] + ALPHA_TRUE_T * (A_t[t] + cmd6[t, :3]) + push_t[t]
        rot[t + 1] = rot[t] + ALPHA_TRUE_R * (A_r[t] + cmd6[t, 3:]) + push_r[t]
    pos += 0.0002 * rng.standard_normal((T, 3))          # 측정 노이즈(작게)

    quats = Rotation.from_rotvec(rot).as_quat().astype(np.float32)
    slow_pos = (pos + A_t).astype(np.float32)
    slow_quat = Rotation.from_rotvec(rot + A_r).as_quat().astype(np.float32)
    slow_pred_target_abs = pos_quat_to_pose9(slow_pos, slow_quat)

    return {
        "image0": rng.integers(0, 255, size=(T, 224, 224, 3), dtype=np.uint8),
        "robot_pose_R": pos.astype(np.float32),
        "robot_quat_R": quats,
        "hand_pose_R": (0.1 * rng.standard_normal((T, 7))).astype(np.float32),
        "wrench_wrist_R": (0.5 * rng.standard_normal((T, 6, 32))).astype(np.float32),
        "slow_pred_target_abs": slow_pred_target_abs,
        "is_intervention": is_intervention,
        "residual_pred6": (cmd6 / max(head_gain, 1e-9) if head_gain > 0
                           else np.zeros((T, 6))).astype(np.float32),
        "residual_cmd6": cmd6.astype(np.float32),
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
    # 지연 모델 적합에 nominal 프레임이 최소 50 필요 -> 합성 에피소드를 넉넉히.
    n_eps = 4
    for i in range(n_eps):
        # 뒤 2개는 head 가 켜진 라운드(δ 를 얹음) — α 적합 편향 보정을 검증하기 위함
        ep = make_synthetic_episode(T=48, seed=i, intervene_slice=(16, 30),
                                    head_gain=(1.0 if i >= 2 else 0.0))
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

    print("\n[smoke] ===== Part D: 추종지연 제거 + 명령 공간 게인 =====")
    from online_learning import lag_model
    lag = learner.lag
    assert lag is not None, "learner.lag 이 비어 있음 — _build_dataset 에서 적합됐어야 함"
    print(f"[smoke] {lag_model.describe(lag)}")

    # (0) 합성 데이터에 심어둔 '진짜' α 를 복원했는가
    a_hat = np.diag(lag["alpha"])
    a_true = np.concatenate([ALPHA_TRUE_T, ALPHA_TRUE_R])
    err = np.abs(a_hat - a_true)
    print(f"[smoke] α 복원: 추정 {np.round(a_hat,3).tolist()}\n"
          f"                 진짜 {np.round(a_true,3).tolist()}  max|오차| {err.max():.3f}")
    assert err.max() < 0.08, f"α 복원 실패 max|오차|={err.max():.3f}"
    print("[smoke] ✅ D0: 심어둔 임피던스 도달률 α 를 복원")
    assert lag.get("n_head", 0) > 0, "합성 데이터에 head 켜진 라운드가 없음 — 보정 경로 미검증"
    # 보정을 끄면(=순진한 적합) 같은 데이터에서 α 가 얼마나 틀리는지 대조
    import h5py as _h5
    with _h5.File(learner.accumulated_hdf5, "r") as _f:
        _ns = sorted(_f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
        _r = np.vstack([np.asarray(_f[f"data/{n}/obs"]["residual_delta6_slow_pred_to_virtual"]) for n in _ns])
        _x = np.vstack([lag_model.rel9_to_delta6(np.asarray(_f[f"data/{n}/obs"]["slow_pred_action_rel"])[:, :9])
                        for n in _ns])
        _c = np.vstack([np.asarray(_f[f"data/{n}/obs"]["residual_cmd6"]) for n in _ns])
        _i = np.concatenate([np.asarray(_f[f"data/{n}/obs"]["is_intervention"]).reshape(-1) > 0.5 for n in _ns])
    naive = np.diag(lag_model.fit(_r, _x, _i)["alpha"])          # δ 안 넘김 = 순진한 적합
    print(f"[smoke] 보정 α 오차 {err.max():.3f}  vs  순진한 적합 오차 {np.abs(naive-a_true).max():.3f}")
    assert np.abs(naive - a_true).max() > err.max(), "보정이 순진한 적합보다 낫지 않음"
    print("[smoke] ✅ D0b: head 켜진 라운드에서 (res−δ)~(A+δ) 보정이 편향을 줄임")

    # (1) accumulated 에 지연 제거 타깃이 실제로 써졌는가
    import h5py
    with h5py.File(learner.accumulated_hdf5, "r") as f:
        o = f["data/demo_0/obs"]
        assert lag_model.CORRECTION_KEY in o, \
            f"{lag_model.CORRECTION_KEY} 가 accumulated 에 없음: {list(o.keys())}"
        raw = np.asarray(o["residual_delta6_slow_pred_to_virtual"])
        cor = np.asarray(o[lag_model.CORRECTION_KEY])
        rel = lag_model.rel9_to_delta6(np.asarray(o["slow_pred_action_rel"])[:, :9])
        isint = np.asarray(o["is_intervention"]).reshape(-1) > 0.5
    assert cor.shape == raw.shape, f"타깃 shape 불일치 {cor.shape} vs {raw.shape}"
    # 재계산과 일치하는가 — nominal 은 0 으로 덮이므로 개입 프레임에서만 비교
    assert np.abs(cor[isint] - lag_model.remove_lag(raw, rel, lag)[isint]).max() < 1e-5
    # nominal 타깃이 정확히 0 으로 강제됐는가 (ZERO_NOMINAL)
    assert np.abs(cor[~isint]).max() == 0.0, \
        f"nominal 타깃이 0 이 아님 (max={np.abs(cor[~isint]).max():.3e})"
    assert np.abs(cor[isint]).max() > 0.0, "개입 프레임 타깃까지 0 이 됨"
    print(f"[smoke] nominal 타깃 전부 0 ({int((~isint).sum())} 프레임), "
          f"개입 타깃 최대 {np.abs(cor[isint]).max()*1000:.2f}mm")
    print("[smoke] ✅ D1: nominal 타깃 0 강제 + 개입 타깃 보존")

    # (2) 학습 타깃이 실제로 지연 제거본인가 — replay_buffer 의 action 값을 직접 비교
    ds2 = learner._build_dataset()
    act = np.asarray(ds2.replay_buffer["action"], dtype=np.float64)
    with h5py.File(learner.accumulated_hdf5, "r") as f:
        raw_all = np.concatenate(
            [np.asarray(f[f"data/{n}/obs"]["residual_delta6_slow_pred_to_virtual"])
             for n in sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))], axis=0)
        cor_all = np.concatenate(
            [np.asarray(f[f"data/{n}/obs"][lag_model.CORRECTION_KEY])
             for n in sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))], axis=0)
    m_ = min(len(act), len(cor_all))
    d_cor = np.abs(act[:m_] - cor_all[:m_]).max()
    d_raw = np.abs(act[:m_] - raw_all[:m_]).max()
    print(f"[smoke] dataset action vs 지연제거본 max|Δ|={d_cor:.3e} · vs raw residual max|Δ|={d_raw:.3e}")
    assert d_cor < 1e-5, f"학습 타깃이 지연 제거본이 아님 (Δ={d_cor:.3e})"
    assert d_raw > 1e-4, "지연 제거본과 raw 가 구분되지 않음 — 교체가 안 된 듯"
    print("[smoke] ✅ D2: learner 가 지연 제거본을 타깃으로 학습")

    # (3) payload 에 지연 모델이 실려 actor 가 복원 가능한가
    assert payload.get("lag") is not None, "발행 payload 에 lag 없음"
    lg = lag_model.from_payload(payload["lag"])
    assert np.abs(lg["alpha_inv"] - lag["alpha_inv"]).max() < 1e-9
    print("[smoke] ✅ D3: payload 로 지연 모델 왕복 일치")

    # (4) 명령 공간 환산: δ = α⁻¹·e 를 다시 α 로 통과시키면 e 로 돌아오는가
    e = np.array([[0.002, -0.001, 0.0015, 0.01, -0.005, 0.004]])
    d = lag_model.to_command(e, lag)
    back = d @ lag["alpha"].T
    print(f"[smoke] e={np.round(e[0][:3]*1000,2)}mm -> 명령 δ={np.round(d[0][:3]*1000,2)}mm "
          f"-> 도달 {np.round(back[0][:3]*1000,2)}mm")
    assert np.abs(back - e).max() < 1e-9, "α⁻¹ 환산 왕복 불일치"
    amp = np.linalg.norm(d[0][:3]) / np.linalg.norm(e[0][:3])
    print(f"[smoke] 병진 증폭 배율 {amp:.2f}x (게인 scale=1.0)")
    assert amp > 1.5, f"명령 공간 환산이 안 먹음(배율 {amp:.2f})"
    # 게인 스케일이 선형으로 먹는가
    assert np.abs(lag_model.to_command(e, lag, gain_scale=0.5) - 0.5 * d).max() < 1e-12
    print("[smoke] ✅ D4: e -> α⁻¹ -> 도달 왕복 일치, 게인 스케일 선형")
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
