"""임피던스 추종지연(lag) 모델 — 학습 타깃에서 '컨트롤러 지연'을 걷어내기 위한 공용 모듈.

## 왜 필요한가

actor 가 기록하는 raw residual 은
    residual[t] = T_base(t)⁻¹ · T_achieved(t+1)
인데, 임피던스 팔은 한 tick(0.1s)에 명령 스텝의 일부(α)만 도달한다. 그래서

    T_achieved(t+1) ≈ T_current ⊕ α·A       (A = base_action_rel = base 가 명령한 스텝)
    residual        ≈ (α − I)·A

즉 라벨의 대부분이 '사람 교정'이 아니라 '컨트롤러 지연'이다. 실측(개입형 5 에피소드,
nominal 1252 프레임): R² 0.974(병진) / 0.995(회전), α 대각 = 병진 [0.29, 0.14, 0.26] /
회전 [0.11, 0.06, 0.07]. 명령 13.1mm 를 주면 그 tick 에 3.3mm 만 간다.

게다가 A(base_action_rel)는 residual head 의 **입력**이다. 그대로 학습시키면 head 는
residual ≈ (α−I)·A 라는 자명한 선형 사상을 배우고, 추론 시
    최종 명령 = base ⊕ (α−I)·A  →  실제 명령 스텝이 α 배로 줄어 팔이 느려진다.

## 무엇을 하는가

    e[t] = residual[t] − (W_A·A[t] + b)      # 지연 제거 → '사람이 추가로 만든 변위'
                                             # (achieved 공간. nominal 에선 ≈ 0)

e 가 학습 타깃이다. 추론 시에는 actor 가 명령 공간으로 환산해서 얹는다:

    δ = α⁻¹ · e_pred ,   α = W_A + I         # 명령 공간
    최종 명령 = T_base ⊕ δ

α⁻¹ 을 **라벨에 굽지 않고 actor 게인으로 두는** 이유:
  * α 는 stiffness/주파수/페이로드에 딸린 값이라 세션마다 변한다. 라벨에 구우면 stiffness 를
    바꿀 때마다 전량 재라벨해야 한다.
  * 게인이면 재학습 없이 온라인으로 낮췄다 올릴 수 있다(초기 라운드 안전).
  * α⁻¹ 은 회전에서 15배까지 증폭이라, 라벨에 구우면 그 증폭이 학습 손실에 직접 들어간다.

## 주의

  * α 는 **수집 세션마다 nominal 프레임으로 다시 적합**해야 한다(이 모듈이 학습 라운드마다
    accumulated 에서 다시 적합한다).
  * nominal 판별에는 per-step is_intervention 이 필요하다 → 개입형(intervention) 경로 전용.
    teleop 판은 이 플래그가 없어 nominal 을 못 가르므로 적용하지 않는다.
  * 6D 벡터 뺄셈은 SE(3) 합성의 1차 근사다. 이 스케일(≈10mm, ≈2°)에선 오차가 무시할 수준.
"""
import numpy as np

# 지연 제거된 학습 타깃이 저장되는 obs 키 (learner 가 accumulated 에 써 넣는다)
CORRECTION_KEY = "correction_delta6"


def rel9_to_delta6(rel9):
    """base_action_rel(pose9 = pos3 + rot6d) -> 6D twist [병진3, rotvec3]."""
    from scipy.spatial.transform import Rotation
    from diffusion_policy.residual_policy.pose_util import pose9_to_mat
    M = np.asarray(pose9_to_mat(np.asarray(rel9, dtype=np.float64)))
    return np.concatenate(
        [M[:, :3, 3], Rotation.from_matrix(M[:, :3, :3]).as_rotvec()], axis=1)


def fit(residual6, rel6, is_intervention, cmd6=None, ridge=1e-4):
    """nominal 프레임으로 도달률 α 를 적합.

    ★ head 가 켜진 라운드 보정 ★
    head 가 δ 를 얹으면 실행 명령은 A 가 아니라 (A+δ) 다. 그래서 순진하게 residual 을 A 에만
    회귀하면 α 가 편향된다(실측: y축 α 0.219 → 0.084, 2.6배 과소추정 → 게인 2.6배 과대).
    정확한 식은 선형으로 정리된다:

        residual = α·(A+δ) − A ,  α = W_A + I
                 = W_A·(A+δ) + δ
        ⇒  (residual − δ) = W_A·(A+δ) + b        ← (A+δ) 에 회귀

    δ=0(head 없는 라운드)이면 기존 식 residual = W_A·A + b 와 동일하다.

    residual6        : (N,6) raw residual (base 프레임)
    rel6             : (N,6) base 가 명령한 스텝 A (같은 프레임 근사)
    cmd6             : (N,6) 그 tick 에 head 가 실제로 얹은 δ. None/0 이면 head 없는 라운드.
    is_intervention  : (N,) bool — True 인 프레임은 사람 교정이 섞여 있어 적합에서 제외
    ridge            : 정규화 세기(스케일 상대값). 어떤 축이 거의 안 움직여 여기(excitation)가
                       없으면 그 방향의 α 를 추정할 수 없다. ridge 는 그때 W_A→0, 즉 α→I 로
                       수축시켜 **게인 1(=보정 안 함)** 쪽으로 안전하게 떨어지게 한다.
                       (반대로 정규화가 없으면 α 가 특이해져 α⁻¹ 이 폭주한다.)
    반환: dict(W_A(6,6), b(6,), alpha(6,6), alpha_inv(6,6), r2_t, r2_r, n, cond)
    """
    residual6 = np.asarray(residual6, dtype=np.float64)
    rel6 = np.asarray(rel6, dtype=np.float64)
    cmd6 = (np.zeros_like(residual6) if cmd6 is None
            else np.asarray(cmd6, dtype=np.float64))
    nom = ~np.asarray(is_intervention, dtype=bool)
    if nom.sum() < 50:
        raise ValueError(f"지연 모델 적합에 nominal 프레임이 부족합니다 ({int(nom.sum())} < 50)")

    # (residual − δ) 를 (A + δ) 에 회귀 — head 가 켜진 라운드도 편향 없이 들어간다.
    X = np.hstack([(rel6 + cmd6)[nom], np.ones((int(nom.sum()), 1))])
    Y = (residual6 - cmd6)[nom]
    # ridge: 절편은 규제하지 않는다(오프셋은 그대로 빼야 nominal 편향이 0 이 된다).
    G = X.T @ X
    lam = float(ridge) * (np.trace(G[:6, :6]) / 6.0 + 1e-30)
    Rg = np.eye(7) * lam
    Rg[6, 6] = 0.0
    W = np.linalg.solve(G + Rg, X.T @ Y)               # (7,6)
    W_A, b = W[:6].T, W[6]                             # residual ≈ W_A·rel + b

    pred = (rel6 + cmd6)[nom] @ W_A.T + b
    ss = ((Y - pred) ** 2).sum(axis=0)
    tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss / np.maximum(tot, 1e-12)

    alpha = W_A + np.eye(6)                            # 한 tick 도달률 행렬
    cond = float(np.linalg.cond(alpha))
    if not np.isfinite(cond) or cond > 1e3:
        raise ValueError(f"도달률 행렬 α 가 특이합니다(cond={cond:.1f}). 데이터를 확인하세요.")
    return dict(W_A=W_A, b=b, alpha=alpha, alpha_inv=np.linalg.inv(alpha),
                r2_t=float(np.mean(r2[:3])), r2_r=float(np.mean(r2[3:])),
                n=int(nom.sum()), cond=cond,
                n_head=int((np.abs(cmd6[nom]).max(axis=1) > 1e-9).sum()))


def remove_lag(residual6, rel6, lag):
    """raw residual -> 지연 제거된 교정 성분 e (achieved 공간). 학습 타깃.

    기준은 **base 명령만으로 갔을 곳**이다(δ 를 빼지 않는다). head 가 켜진 라운드에서는
    e = (head 기여 α·δ) + (사람 추가) 가 되는데, 이게 DAgger 누적으로 맞다 — 다음 head 는
    "base 위에 얹어야 할 총 교정량"을 배워야 하지 사람이 추가한 증분만 배우면 안 된다.
    (사람 증분만 보려면 e − α·δ. 리포트의 head/사람 분해가 그 식을 쓴다.)"""
    residual6 = np.asarray(residual6, dtype=np.float64)
    rel6 = np.asarray(rel6, dtype=np.float64)
    return residual6 - (rel6 @ lag["W_A"].T + lag["b"])


def to_command(e6, lag, gain_scale=1.0):
    """교정 성분 e(achieved 공간) -> 명령 공간 δ = gain_scale · α⁻¹·e. actor 가 쓴다."""
    e6 = np.asarray(e6, dtype=np.float64)
    return float(gain_scale) * (e6 @ np.asarray(lag["alpha_inv"]).T)


def describe(lag):
    a = np.diag(np.asarray(lag["alpha"]))
    extra = ""
    if lag.get("n_head"):
        extra += f" · head-on {lag['n_head']} 프레임 보정 적합"
    if lag.get("n_zeroed"):
        extra += f" · nominal 타깃 0 강제 {lag['n_zeroed']} 프레임"
    return (f"α 대각 병진 {np.round(a[:3], 3).tolist()} 회전 {np.round(a[3:], 3).tolist()} · "
            f"R² 병진 {lag['r2_t']:.3f} 회전 {lag['r2_r']:.3f} · "
            f"nominal {lag['n']} 프레임 · cond {lag['cond']:.1f}{extra}")


# ── 직렬화 (learner -> actor payload / 디스크) ────────────────────────────────
def to_payload(lag):
    return {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
            for k, v in lag.items()}


def from_payload(d):
    if d is None:
        return None
    out = dict(d)
    for k in ("W_A", "b", "alpha", "alpha_inv"):
        if k in out:
            out[k] = np.asarray(out[k], dtype=np.float64)
    return out


# ── HDF5 (accumulated.hdf5) 에 지연 제거 타깃 써 넣기 ─────────────────────────
def fit_and_write_hdf5(hdf5_path,
                       residual_key="residual_delta6_slow_pred_to_virtual",
                       rel_key="slow_pred_action_rel",
                       intervention_key="is_intervention",
                       cmd_key="residual_cmd6",
                       out_key=CORRECTION_KEY,
                       zero_nominal=True):
    """accumulated.hdf5 전체 demo 로 지연 모델을 적합하고, 각 demo 에 obs/<out_key> 를 쓴다.

    zero_nominal: 개입하지 않은 프레임의 타깃을 **정확히 0** 으로 강제한다(기본 켜짐).
        CR-DAgger 전제 — "사람이 개입 안 한 구간 = base 가 옳다 = 교정 불필요". 끄면 지연
        제거 후 남은 잔차(프레임당 ~2mm, 방향 랜덤)가 그대로 라벨이 되는데, head 가 그
        노이즈를 맞추려 들어 **안 밀어도 되는 구간에서 계속 명령을 낸다**(v2 실측: nominal
        예측 1.71mm → 게인 적용 시 6.7mm, base 스텝의 55%). 0 으로 두면 그 구간을
        "건드리지 마라" 는 명시적 negative 로 배운다.

    반환: lag dict. is_intervention 이 없는 demo 가 하나라도 있으면 None 을 반환하고
    아무것도 쓰지 않는다(teleop 판 호환 — 그 경로는 지연 제거를 쓰지 않는다).
    """
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        if not names:
            return None
        res, rel, isint, cmd = [], [], [], []
        for nm in names:
            o = f[f"data/{nm}/obs"]
            if intervention_key not in o or residual_key not in o or rel_key not in o:
                return None
            r = np.asarray(o[residual_key], dtype=np.float64)
            res.append(r)
            rel.append(rel9_to_delta6(np.asarray(o[rel_key])[:, :9])[:len(r)])
            isint.append(np.asarray(o[intervention_key]).reshape(-1)[:len(r)] > 0.5)
            # head 가 켜진 라운드면 그 tick 에 얹은 δ. 없으면 0(구버전 데이터 호환).
            cmd.append(np.asarray(o[cmd_key], dtype=np.float64)[:len(r)]
                       if cmd_key in o else np.zeros_like(r))

    lag = fit(np.vstack(res), np.vstack(rel),
              np.concatenate(isint), cmd6=np.vstack(cmd))

    n_zero = 0
    with h5py.File(hdf5_path, "a") as f:
        for nm, r, x, mi in zip(names, res, rel, isint):
            o = f[f"data/{nm}/obs"]
            tgt = remove_lag(r, x, lag)
            if zero_nominal:
                tgt[~mi] = 0.0
                n_zero += int((~mi).sum())
            if out_key in o:
                del o[out_key]
            o.create_dataset(out_key, data=tgt.astype(np.float32))
    lag["n_zeroed"] = n_zero
    return lag
