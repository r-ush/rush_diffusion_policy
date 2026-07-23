#!/usr/bin/env python
"""개입(intervention) 수집 품질 리포트 HTML.

"사람이 얼마나 / 어떻게 개입했나" 를 에피소드별로 본다. export_traj3d_playback_html.py 가
프레임별 3D 궤적 재생이라면, 이쪽은 **개입량·개입품질 대시보드**다.

측정 지표 (왜 이걸 보는가):
  * 개입 비율/시간/span 수   — 얼마나 개입했나(양).
  * ‖평균벡터‖ 와 coherence  — 얼마나 "일관된 방향"으로 교정했나(질).
      residual 크기(‖·‖의 평균)만 보면 안 된다. nominal 의 임피던스 추종지연은 크기는
      크지만 방향이 랜덤이라 평균벡터가 작고, 사람 교정은 크기가 비슷해도 방향이 일정해
      평균벡터가 크다. residual head 가 학습하는 것은 조건부 평균이므로, 학습 가능한
      신호 = 평균벡터. coherence = ‖mean(v)‖ / mean(‖v‖) ∈ [0,1].
  * 평균벡터 비(개입 / nominal 위상0) — nominal 대비 교정이 얼마나 두드러지나.
      nominal 은 chunk 위상(stale chunk drift)에 따라 residual 이 커지므로, 개입(매 tick
      fresh replan = 위상0)과 공정하게 비교하려면 nominal 도 위상0 프레임만 쓴다.

합격 기준(기본): 평균벡터 비 > 2.0 AND 개입 coherence > 0.8.

실행:
  python online_learning/export_intervention_report_html.py \
    --episodes data/online_runs/run_hand_intervention/transitions \
    --out data/verify_intervention
  (순수 numpy/h5py — 정책·torch 불필요)
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
from scipy.spatial.transform import Rotation

from diffusion_policy.residual_policy.pose_util import pose9_to_mat

from online_learning import lag_model


# ── 데이터 ────────────────────────────────────────────────────────────────────
def _coh(v):
    """(n,3) -> (‖평균벡터‖, 평균 크기, coherence). 학습 가능한 신호는 평균벡터."""
    if len(v) == 0:
        return 0.0, 0.0, 0.0
    mv = float(np.linalg.norm(v.mean(axis=0)))
    mn = float(np.linalg.norm(v, axis=1).mean())
    return mv, mn, mv / max(mn, 1e-9)


def _spans(mask):
    """bool 마스크 -> [(start, end)] 연속 구간."""
    out, cur = [], None
    for t, v in enumerate(mask):
        if v and cur is None:
            cur = t
        elif not v and cur is not None:
            out.append((cur, t)); cur = None
    if cur is not None:
        out.append((cur, len(mask)))
    return out


def load_raw(fp):
    """에피소드 원시 배열 로드(지연 모델 적합 전 단계)."""
    with h5py.File(fp, "r") as f:
        if "data" not in f:
            return None
        demo = sorted(f["data"].keys())[0]
        o = f[f"data/{demo}/obs"]
        if "is_intervention" not in o:
            return None
        d = dict(
            name=os.path.basename(fp).replace(".hdf5", ""),
            isint=np.asarray(o["is_intervention"]).reshape(-1) > 0.5,
            res=np.asarray(o["residual_delta6_slow_pred_to_virtual"]).astype(np.float64),
            rel9=np.asarray(o["slow_pred_action_rel"])[:, :9].astype(np.float64),
            pose=np.asarray(o["robot_pose_R"])[:, :3].astype(np.float64),
            slow9=np.asarray(o["slow_pred_target_abs"])[:, :9].astype(np.float64),
            quat=np.asarray(o["robot_quat_R"]).astype(np.float64),
            wr=np.asarray(o["wrench_wrist_R"]).astype(np.float64),
            # 라운드 2 이후에만 존재. 없으면 0(=head 미적용 라운드).
            pred6=(np.asarray(o["residual_pred6"]).astype(np.float64)
                   if "residual_pred6" in o else None),
            cmd6=(np.asarray(o["residual_cmd6"]).astype(np.float64)
                  if "residual_cmd6" in o else None),
        )
    T = len(d["res"])
    d["T"] = T
    d["isint"] = d["isint"][:T]
    for k in ("pred6", "cmd6"):
        d[k] = np.zeros((T, 6)) if d[k] is None else d[k][:T]
    d["has_head"] = bool(np.abs(d["cmd6"]).max() > 1e-9)
    # base 가 그 tick 에 명령한 스텝(현재 pose 기준 상대) 6D = [병진3, 회전 rotvec3]
    M = np.asarray(pose9_to_mat(d["rel9"][:T]))
    d["rel6"] = np.concatenate(
        [M[:, :3, 3], Rotation.from_matrix(M[:, :3, :3]).as_rotvec()], axis=1)
    return d


def fit_lag(raws):
    """추종지연 모델 적합 — 학습 경로(online_learning/lag_model.py)와 **같은 구현**을 쓴다.

    리포트가 보여주는 값과 learner 가 실제로 학습하는 타깃이 어긋나면 안 되므로, 여기서
    따로 계산하지 않고 그대로 위임한다(축별 α 행렬 + ridge 포함).
    """
    return lag_model.fit(
        np.vstack([r["res"] for r in raws]),
        np.vstack([r["rel6"] for r in raws]),
        np.concatenate([r["isint"] for r in raws]),
        cmd6=np.vstack([r["cmd6"] for r in raws]))   # head 켜진 라운드 편향 보정


def analyze(raw, lag, spi, dt, pass_mag, pass_rmag, pass_coh):
    T, m = raw["T"], raw["isint"]
    res = raw["res"]
    # 지연 제거 = 학습에 실제로 쓰여야 할 타깃(사람 교정 성분)
    cor = lag_model.remove_lag(res, raw["rel6"], lag)

    # ── 두 공간을 모두 쓴다 ──
    #  e   = 지연 제거된 교정 성분 (achieved 공간). **head 가 학습하는 타깃 그 자체**.
    #        판정(교정량·coherence)은 이 공간에서 한다 — "학습 가능한 신호가 있는가" 가 질문이라
    #        head 가 회귀하는 값의 일관성을 재는 게 맞다.
    #  cmd = α⁻¹·e (명령 공간). actor 가 base 예측 위에 실제로 얹을 delta.
    #        크기 감각(base 스텝 대비 / 캡 대비)은 이쪽이라 표시는 이 값으로 한다.
    #  주의: α 는 행렬이라 스칼라와 달리 방향을 바꾼다 → coherence 는 두 공간에서 값이 다르다.
    #        그래서 판정은 타깃 공간(e)으로 고정한다.
    cmd = lag_model.to_command(cor, lag)
    et = cor[:, :3]                            # 학습 타깃 · 병진 (m)
    er = np.degrees(cor[:, 3:6])               # 학습 타깃 · 회전 (deg)
    ct = cmd[:, :3]                            # 교정 명령 · 병진 (m)
    cr = np.degrees(cmd[:, 3:6])               # 교정 명령 · 회전 (deg)
    rt = res[:, :3]                            # raw residual · 병진 (지연 포함)
    rrt = np.degrees(res[:, 3:6])
    base_step = np.linalg.norm(raw["rel6"][:, :3], axis=1)   # base 가 명령한 스텝 크기

    # chunk 위상(참고용): 개입 토글마다 slow chunk 이 무효화되므로 세그먼트 시작에서 리셋.
    bounds = [0] + [int(i) + 1 for i in np.where(np.diff(m.astype(int)) != 0)[0]] + [T]
    ph = np.zeros(T, dtype=int)
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = np.arange(a, min(b, T))
        ph[idx] = (idx - a) % spi

    move = np.zeros(T)
    move[:-1] = np.linalg.norm(np.diff(raw["pose"][:T], axis=0), axis=1)
    force = np.linalg.norm(raw["wr"][:T, :3, -1], axis=1)

    # 핵심 지표: 개입 프레임의 ‖평균벡터‖(=학습 가능한 신호량) 와 방향 일관성,
    # 그리고 nominal 프레임에 남은 '누설'(지연 모델이 못 걷어낸 잔여 편향).
    # ── head 기여 vs 사람 추가 분해 (라운드 2 이후) ──
    #   e_total = 지연 제거 교정(= base 명령만으로 갔을 곳 대비 실제 변위)
    #           = head 가 만든 변위 + 사람이 그 위에 얹은 변위
    #   head 가 명령한 δ(=residual_cmd6, 명령 공간)는 한 tick 에 α 배만 도달하므로
    #   achieved 공간 기여분은 α·δ 다.
    e_head = raw["cmd6"] @ np.asarray(lag["alpha"]).T
    e_human = cor - e_head
    ht, hh = e_head[:, :3], e_human[:, :3]

    # 판정용(타깃 공간 e)
    mag_i, _, coh_i = _coh(et[m]);   mag_n, _, coh_n = _coh(et[~m])
    rmag_i, _, rcoh_i = _coh(er[m]); rmag_n, _, rcoh_n = _coh(er[~m])
    # 표시용(명령 공간)
    cmag_i, _, _ = _coh(ct[m]);  crmag_i, _, _ = _coh(cr[m])
    raw_i, _, _ = _coh(rt[m]);   raw_n, _, _ = _coh(rt[~m])
    # head 성적표(개입 프레임 기준)
    hm_i, _, _ = _coh(ht[m]); hm_n, _, _ = _coh(ht[~m])
    hu_i, _, _ = _coh(hh[m])
    pm_i, _, _ = _coh(raw["pred6"][m][:, :3]) if m.any() else (0.0, 0.0, 0.0)
    _mt, _mp = et[m].mean(axis=0), raw["pred6"][m][:, :3].mean(axis=0) if m.any() else (0, 0)
    head_cos = float(np.dot(_mt, _mp) / max(np.linalg.norm(_mt) * np.linalg.norm(_mp), 1e-12)) \
        if m.any() else 0.0

    bstep_i = float(base_step[m].mean()) if m.any() else 0.0
    bstep_n = float(base_step[~m].mean()) if (~m).any() else 0.0

    mm = lambda x: round(x * 1000, 2)
    ok_t = bool(mm(mag_i) >= pass_mag and coh_i >= pass_coh and mag_i > mag_n)
    ok_r = bool(round(rmag_i, 3) >= pass_rmag and rcoh_i >= pass_coh and rmag_i > rmag_n)
    ok = ok_t or ok_r

    spans = []
    for si, (a, b) in enumerate(_spans(m)):
        sub = []
        if (b - a) >= 15:
            for k, idx in enumerate(np.array_split(np.arange(a, b), 3)):
                lo, hi = idx[0], idx[-1] + 1
                smv, _, sc = _coh(et[lo:hi])          # coherence 는 타깃 공간
                srmv, _, src = _coh(er[lo:hi])
                smv, srmv = _coh(ct[lo:hi])[0], _coh(cr[lo:hi])[0]   # 크기는 명령 공간
                sub.append(dict(k=f"{k+1}/3", coh=round(sc, 3), mag=mm(smv),
                                rcoh=round(src, 3), rmag=round(srmv, 2),
                                vec=[mm(float(x)) for x in ct[lo:hi].mean(axis=0)],
                                rvec=[round(float(x), 2) for x in cr[lo:hi].mean(axis=0)]))
        emv, _, sc = _coh(et[a:b])            # 판정: 타깃 공간
        ermv, _, src = _coh(er[a:b])
        smv = _coh(ct[a:b])[0]                # 표시: 명령 공간
        srmv = _coh(cr[a:b])[0]
        smn = _coh(et[a:b])[1]
        s_ok_t = bool(mm(emv) >= pass_mag and sc >= pass_coh)
        s_ok_r = bool(round(ermv, 3) >= pass_rmag and src >= pass_coh)
        spans.append(dict(
            ok=bool(s_ok_t or s_ok_r), ok_t=s_ok_t, ok_r=s_ok_r,
            sub=sub, idx=si + 1, start=int(a), end=int(b), n=int(b - a),
            dur=round((b - a) * dt, 1),
            mag=mm(smv), meanmag=mm(smn), coh=round(sc, 3),
            vec=[mm(float(x)) for x in ct[a:b].mean(axis=0)],
            rmag=round(srmv, 2), rcoh=round(src, 3),
            rvec=[round(float(x), 2) for x in cr[a:b].mean(axis=0)],
            rawmag=mm(_coh(rt[a:b])[0]),
            move=round(float(move[a:b].mean()) * 1000, 2),
            force=round(float(force[a:b].mean()), 2),
        ))

    # 3D 패널용: 에피소드 첫 achieved 를 원점으로 한 mm 좌표.
    org = raw["pose"][0]
    p3 = [[round(float(x), 1) for x in ((raw["pose"][i] - org) * 1000)] for i in range(T)]
    s3 = [[round(float(x), 1) for x in ((raw["slow9"][i, :3] - org) * 1000)] for i in range(T)]
    Rb = np.asarray(pose9_to_mat(raw["slow9"][:T]))[:, :3, :3]
    Ra = Rotation.from_quat(raw["quat"][:T]).as_matrix()
    # 교정 '명령' delta(ct)는 base pose 의 로컬 프레임 값이다. 3D 에 그리려면 로깅 좌표계로
    # 돌려서, **base 예측 지점에 붙는 벡터**로 내보낸다(현재 pose 가 아니라 base 목표가 기점).
    dcmd = np.einsum('tij,tj->ti', Rb, ct) * 1000.0
    dc3 = [[round(float(x), 2) for x in dcmd[i]] for i in range(T)]
    rb = [[round(float(x), 3) for x in Rb[i].T.reshape(-1)] for i in range(T)]
    ra = [[round(float(x), 3) for x in Ra[i].T.reshape(-1)] for i in range(T)]

    return dict(
        name=raw["name"], T=int(T), dt=dt, pose3=p3, slow3=s3, rotb=rb, rota=ra,
        dcmd3=dc3,
        has_head=bool(raw["has_head"]),
        etot_mm=[round(float(x), 3) for x in (np.linalg.norm(et, axis=1) * 1000)],
        ehead_mm=[round(float(x), 3) for x in (np.linalg.norm(ht, axis=1) * 1000)],
        ehuman_mm=[round(float(x), 3) for x in (np.linalg.norm(hh, axis=1) * 1000)],
        cor_mm=[round(float(x), 3) for x in (np.linalg.norm(ct, axis=1) * 1000)],
        base_mm=[round(float(x), 3) for x in (base_step * 1000)],
        res_mm=[round(float(x), 3) for x in (np.linalg.norm(rt, axis=1) * 1000)],
        rcor_deg=[round(float(x), 3) for x in np.linalg.norm(cr, axis=1)],
        rres_deg=[round(float(x), 3) for x in np.linalg.norm(rrt, axis=1)],
        move_mm=[round(float(x), 3) for x in (move * 1000)],
        force_n=[round(float(x), 3) for x in force],
        isint=[int(x) for x in m], spans=spans,
        summary=dict(
            n_int=int(m.sum()), pct=round(100 * float(m.mean()), 1),
            dur=round(float(m.sum()) * dt, 1), total_dur=round(T * dt, 1),
            n_spans=len(spans),
            mag_int=mm(mag_i), leak=mm(mag_n), coh_int=round(coh_i, 3), coh_nom=round(coh_n, 3),
            rmag_int=round(rmag_i, 3), rleak=round(rmag_n, 3),
            rcoh_int=round(rcoh_i, 3), rcoh_nom=round(rcoh_n, 3),
            raw_int=mm(raw_i), raw_nom=mm(raw_n),
            cmd_int=mm(cmag_i), rcmd_int=round(crmag_i, 2),     # 명령 공간(표시용)
            base_step_int=mm(bstep_i), base_step_nom=mm(bstep_n),
            frac_of_base=round(100 * cmag_i / max(bstep_i, 1e-9), 1),  # 명령 공간 / base 스텝 %
            has_head=bool(raw["has_head"]),
            head_mag=mm(hm_i), human_mag=mm(hu_i), head_nom=mm(hm_n),
            pred_mag=mm(pm_i), head_cos=round(head_cos, 3),
            head_frac=round(100 * hm_i / max(mag_i, 1e-12), 1),
            ok=ok, ok_t=ok_t, ok_r=ok_r,
            n_ok_spans=int(sum(1 for sp in spans if sp["ok"])),
        ),
    )


# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>개입 수집 리포트 — residual intervention</title>
<style>
:root{
  color-scheme:light;
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --neg:#e34948; --good:#0ca30c; --crit:#d03b3b;
  --wash:rgba(235,104,52,.13);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  color-scheme:dark;
  --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --neg:#e66767; --good:#0ca30c; --crit:#d03b3b;
  --wash:rgba(217,89,38,.20);
}}
:root[data-theme="dark"]{
  color-scheme:dark;
  --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --neg:#e66767; --good:#0ca30c; --crit:#d03b3b;
  --wash:rgba(217,89,38,.20);
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.5}
.wrap{max-width:1120px;margin:0 auto;padding:32px 20px 64px}
header{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap;margin-bottom:8px}
h1{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--ink2);font-size:13px;margin:0}
button.theme{background:var(--surface);color:var(--ink2);border:1px solid var(--border);
  border-radius:8px;padding:7px 12px;font-size:13px;cursor:pointer;font-family:inherit}
button.theme:hover{color:var(--ink)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;
  padding:22px;margin-top:20px}
.ephead{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:18px}
.ephead h2{font-size:17px;margin:0;letter-spacing:-.01em}
.ephead .meta{color:var(--muted);font-size:13px}
.pill{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;
  padding:3px 10px;border-radius:999px;border:1px solid var(--border)}
.pill.ok{color:var(--good)} .pill.no{color:var(--crit)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:1px;
  background:var(--border);border:1px solid var(--border);border-radius:10px;overflow:hidden;margin-bottom:24px}
.tile{background:var(--surface);padding:14px 16px}
.tile .k{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;
  margin-bottom:6px;white-space:nowrap}
.tile .v{font-size:26px;font-weight:600;letter-spacing:-.02em;line-height:1.1}
.tile .v small{font-size:13px;font-weight:500;color:var(--ink2);margin-left:3px}
.tile .n{font-size:11px;color:var(--muted);margin-top:3px}
.chart{margin-bottom:26px}
.scene{position:relative;border:1px solid var(--border);border-radius:10px;overflow:hidden;
  background:var(--surface)}
.scene canvas{display:block;width:100%;height:420px;cursor:grab;touch-action:none}
.scene canvas:active{cursor:grabbing}
.ctrlwrap{border-top:1px solid var(--border)}
.ctrls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:10px 12px;
  font-size:12px;color:var(--ink2)}
.ctrls.row2{padding-top:0}
/* 재생 버튼은 라벨이 바뀌어도(▶ 재생 ↔ ❚❚ 일시정지) 폭이 고정 — 슬라이더가 안 밀린다 */
.ctrls button.play{min-width:104px}
.ctrls button{background:transparent;color:var(--ink2);border:1px solid var(--border);
  border-radius:7px;padding:5px 11px;font-size:12px;cursor:pointer;font-family:inherit}
.ctrls button:hover{color:var(--ink)}
.ctrls button[aria-pressed="true"]{color:var(--ink);border-color:var(--axis)}
.ctrls input[type=range]{flex:1;min-width:140px;accent-color:var(--s1)}
.ctrls .fr{font-variant-numeric:tabular-nums;min-width:236px;text-align:left;color:var(--muted)}
.hint{font-size:11.5px;color:var(--muted);padding:0 12px 10px}
.mat{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;color:var(--ink2);
  padding:0 12px 10px;white-space:pre;overflow-x:auto;font-variant-numeric:tabular-nums}
.mat b{color:var(--ink);font-weight:600}
.mat .warn{color:var(--crit);font-weight:600}
.ctitle{font-size:13px;font-weight:600;margin:0 0 2px}
.cnote{font-size:12px;color:var(--muted);margin:0 0 10px}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--ink2);margin-bottom:8px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:12px;height:12px;border-radius:3px;display:inline-block}
.sw.line{height:3px;border-radius:2px;width:16px}
svg{display:block;width:100%;height:auto;overflow:visible}
.tick{fill:var(--muted);font-size:10px;font-variant-numeric:tabular-nums}
.alab{fill:var(--muted);font-size:11px}
.dlab{fill:var(--ink2);font-size:11px;font-variant-numeric:tabular-nums}
.tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);
  border:1px solid var(--border);border-radius:8px;padding:8px 11px;font-size:12px;
  box-shadow:0 4px 16px rgba(0,0,0,.16);opacity:0;transition:opacity .1s;z-index:20;
  font-variant-numeric:tabular-nums;white-space:nowrap}
.tip b{font-weight:600} .tip .r{color:var(--ink2)}
details{margin-top:6px} summary{cursor:pointer;font-size:13px;color:var(--ink2);
  padding:6px 0;list-style:none} summary::-webkit-details-marker{display:none}
summary::before{content:"▸ ";color:var(--muted)} details[open] summary::before{content:"▾ "}
.tblwrap{overflow-x:auto;margin-top:8px}
table{border-collapse:collapse;font-size:12.5px;min-width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:7px 12px;border-bottom:1px solid var(--border);white-space:nowrap}
th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
th:first-child,td:first-child{text-align:left}
tbody tr:last-child td{border-bottom:none}
.note{font-size:12.5px;color:var(--ink2);background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:14px 16px;margin-top:20px}
.note b{color:var(--ink)}
</style></head><body>
<div class="wrap">
<header>
  <div><h1>개입 수집 리포트</h1>
  <p class="sub">residual intervention — 사람이 얼마나, 어떻게 개입했는가</p></div>
  <button class="theme" onclick="tog()">테마 전환</button>
</header>
<div class="note" id="lagnote"></div>
<div id="root"></div>
<div class="note">
  <b>읽는 법.</b> 두 단계로 걸러서 본다. (1) <b>지연 제거</b> — raw residual 의 대부분은 명령 스텝에
  비례하는 컨트롤러 지연이라 먼저 뺀다(위 배너). (2) <b>방향 일관성</b> — 남은 성분도 크기만으로는
  판단할 수 없다. 잔여 노이즈는 방향이 랜덤이라 평균이 상쇄되고(coherence 낮음), 사람 교정은 한 방향으로
  일정하다(coherence 높음). residual head 는 조건부 평균을 학습하므로 <b>‖평균벡터‖ 가 곧 학습 가능한
  신호량</b>이다. 그래서 합격 기준은 "교정량(‖평균벡터‖) ≥ 기준 AND coherence ≥ 기준 AND 교정량 &gt; 누설"
  이고, 병진·회전 중 하나만 통과해도 합격으로 본다(자세만 비튼 교정을 병진 지표로 버리지 않기 위해).
</div>
</div>
<div class="tip" id="tip"></div>
<script>
const DATA = __DATA__, CFG = __CFG__;
const tip = document.getElementById('tip');
function tog(){const r=document.documentElement;
  const d=r.getAttribute('data-theme')||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');
  r.setAttribute('data-theme', d==='dark'?'light':'dark');}
const NS='http://www.w3.org/2000/svg';
const el=(t,a={})=>{const e=document.createElementNS(NS,t);
  for(const k in a) e.setAttribute(k,a[k]); return e;};
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const nice=(v,n=2)=>v.toFixed(n);

/* ── 시계열 (단일 계열 + 개입 구간 wash) ── */
function lineChart(host, {series, isint, dt, unit, fmt=2}){
  const vals=series[0].vals;
  const W=1000,H=182,L=52,R=14,T=24,B=26, pw=W-L-R, phh=H-T-B;
  const n=vals.length, mx=Math.max(...series.flatMap(s=>s.vals),1e-9);
  const step=Math.pow(10,Math.floor(Math.log10(mx)));
  const top=Math.ceil(mx/(step/2))*(step/2);
  const X=i=>L+(n<=1?0:i/(n-1)*pw), Y=v=>T+phh-(v/top)*phh;
  const s=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img'});

  // 개입 구간 wash (secondary encoding: 상단 3px 러그도 함께)
  let run=null;
  for(let i=0;i<=n;i++){
    if(i<n && isint[i] && run===null) run=i;
    else if((i===n||!isint[i]) && run!==null){
      s.appendChild(el('rect',{x:X(run),y:T,width:Math.max(X(i-1)-X(run),1.5),height:phh,
        fill:'var(--wash)'}));
      s.appendChild(el('rect',{x:X(run),y:T,width:Math.max(X(i-1)-X(run),1.5),height:3,
        fill:'var(--s2)'}));
      run=null;
    }
  }
  // 그리드 + y축
  for(let g=0;g<=2;g++){const v=top*g/2, y=Y(v);
    s.appendChild(el('line',{x1:L,x2:W-R,y1:y,y2:y,stroke:g?'var(--grid)':'var(--axis)','stroke-width':1}));
    const t=el('text',{x:L-8,y:y+3.5,'text-anchor':'end',class:'tick'});
    t.textContent=nice(v,fmt); s.appendChild(t);}
  const yl=el('text',{x:L-8,y:T-10,'text-anchor':'end',class:'alab'}); yl.textContent=unit;
  s.appendChild(yl);
  // x축 (초)
  for(let g=0;g<=4;g++){const i=Math.round((n-1)*g/4);
    const t=el('text',{x:X(i),y:H-8,'text-anchor':g===0?'start':(g===4?'end':'middle'),class:'tick'});
    t.textContent=nice(i*dt,1)+'s'; s.appendChild(t);}
  // 선 — 뒤쪽 계열(참고용 raw)을 먼저 깔고 주 계열을 위에
  for(let k=series.length-1;k>=0;k--){
    let d='';
    for(let i=0;i<n;i++) d+=(i?'L':'M')+X(i).toFixed(1)+' '+Y(series[k].vals[i]).toFixed(1);
    s.appendChild(el('path',{d,fill:'none',stroke:series[k].color,
      'stroke-width':k?1.4:2,opacity:k?0.65:1,
      'stroke-linejoin':'round','stroke-linecap':'round'}));
  }
  // hover
  const cur=el('line',{x1:0,x2:0,y1:T,y2:T+phh,stroke:'var(--axis)','stroke-width':1,opacity:0});
  const dot=el('circle',{r:4.5,fill:'var(--s1)',stroke:'var(--surface)','stroke-width':2,opacity:0});
  s.appendChild(cur); s.appendChild(dot);
  const hit=el('rect',{x:L,y:T,width:pw,height:phh,fill:'transparent'});
  s.appendChild(hit);
  s.addEventListener('mousemove',e=>{
    const b=s.getBoundingClientRect(), sx=(e.clientX-b.left)/b.width*W;
    if(sx<L-4||sx>W-R+4){cur.setAttribute('opacity',0);dot.setAttribute('opacity',0);tip.style.opacity=0;return;}
    const i=Math.max(0,Math.min(n-1,Math.round((sx-L)/pw*(n-1))));
    cur.setAttribute('x1',X(i));cur.setAttribute('x2',X(i));cur.setAttribute('opacity',1);
    dot.setAttribute('cx',X(i));dot.setAttribute('cy',Y(vals[i]));dot.setAttribute('opacity',1);
    tip.innerHTML=series.map((sr,k)=>
        `${k?'<span class="r">':'<b>'}${esc(sr.name)} ${nice(sr.vals[i],fmt)} ${esc(unit)}${k?'</span>':'</b>'}`)
        .join('<br>')+
      `<br><span class="r">frame ${i} · ${nice(i*dt,1)}s · `+
      (isint[i]?'<b style="color:var(--s2)">개입</b>':'nominal')+'</span>';
    tip.style.opacity=1;
    tip.style.left=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8)+'px';
    tip.style.top=(e.clientY-tip.offsetHeight-12)+'px';
  });
  s.addEventListener('mouseleave',()=>{cur.setAttribute('opacity',0);dot.setAttribute('opacity',0);tip.style.opacity=0;});
  host.appendChild(s);
}

/* ── span 별 평균 교정 벡터 (diverging: + 파랑 / − 빨강) ── */
function vecChart(host, spans, key='vec', unit='mm'){
  const rot = key==='rvec';
  const ck = rot?'rcoh':'coh', mk2 = rot?'rmag':'mag';
  const rows=[];
  spans.forEach(sp=>{
    const parts = sp.sub.length ? sp.sub : [Object.assign({k:'전체'}, sp)];
    parts.forEach(p=>['X','Y','Z'].forEach((ax,k)=>
      rows.push({lab:`구간 ${sp.idx} · ${p.k} · ${rot?'r':''}${ax}`, v:p[key][k],
                 sp, p, ck, mk2, unit})));
  });
  const W=1000, rh=26, T=22, B=8, L=132, R=64, H=T+B+rows.length*rh, pw=W-L-R;
  const mx=Math.max(...rows.map(r=>Math.abs(r.v)),1e-9);
  // 25% 헤드룸: 최대 막대가 축 끝에 닿으면 값 라벨이 행 라벨 위로 넘어간다.
  const top=Math.max(2,Math.ceil(mx*1.25/2)*2), C=L+pw/2, X=v=>C+(v/top)*(pw/2);
  const s=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img'});
  [-top,-top/2,0,top/2,top].forEach(v=>{
    s.appendChild(el('line',{x1:X(v),x2:X(v),y1:T-6,y2:H-B,
      stroke:v===0?'var(--axis)':'var(--grid)','stroke-width':1}));
    const t=el('text',{x:X(v),y:T-11,'text-anchor':'middle',class:'tick'});
    t.textContent=nice(v,0); s.appendChild(t);});
  const u=el('text',{x:W-R+8,y:T-11,'text-anchor':'start',class:'alab'});
  u.textContent=unit; s.appendChild(u);
  rows.forEach((r,i)=>{
    const y=T+i*rh+rh/2, h=11;
    const x0=Math.min(C,X(r.v)), w=Math.max(Math.abs(X(r.v)-C),1.5);
    const rect=el('rect',{x:x0,y:y-h/2,width:w,height:h,rx:4,
      fill:r.v>=0?'var(--s1)':'var(--neg)'});
    s.appendChild(rect);
    const lb=el('text',{x:L-12,y:y+4,'text-anchor':'end',class:'dlab'});
    lb.textContent=r.lab; s.appendChild(lb);
    const vl=el('text',{x:r.v>=0?X(r.v)+8:X(r.v)-8,y:y+4,
      'text-anchor':r.v>=0?'start':'end',class:'dlab'});
    vl.textContent=(r.v>0?'+':'')+nice(r.v,1); s.appendChild(vl);
    rect.addEventListener('mousemove',e=>{
      tip.innerHTML=`<b>${esc(r.lab)}: ${(r.v>0?'+':'')+nice(r.v,2)} ${esc(r.unit)}</b><br>`+
        `<span class="r">이 조각 coherence ${r.p[r.ck]} · ‖평균벡터‖ ${r.p[r.mk2]}${esc(r.unit)}`+
        ` · 구간 전체 ${r.sp.dur}s</span>`;
      tip.style.opacity=1;
      tip.style.left=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8)+'px';
      tip.style.top=(e.clientY-tip.offsetHeight-12)+'px';});
    rect.addEventListener('mouseleave',()=>tip.style.opacity=0);
  });
  host.appendChild(s);
}

/* ── 3D 궤적 (드래그 회전 / 휠 줌 / 프레임 스크럽) ── */
function scene3d(host, ep){
  const N=ep.T, P=ep.pose3, S=ep.slow3;
  const D3=ep.dcmd3||P.map(()=>[0,0,0]);
  // frm: 'world' = 로봇 base 가 X축으로 기울어진 걸 보정한 월드 기준(기본),
  //      'base'  = 로깅 그대로(로봇 base 프레임). export_traj3d_playback_html 와 동일 규약.
  // 기본 시점 = residual_playback.html(Plotly) 의 기본 카메라(eye 1.25,1.25,1.25, up=+z)와
  // 동일하게 맞춘 값. 두 파일을 나란히 볼 때 같은 방향으로 보이게 하기 위함.
  // (검증: 이 az/el 에서 +X/+Y/+Z 축의 화면 방향이 Plotly 기본 카메라와 소수점 4자리까지 일치)
  // 기본값은 CLI(--view_az/--view_el/--view_flip_x/--view_flip_y)로 바꿀 수 있다.
  // 기본은 residual_playback.html(Plotly 기본 카메라)과 정합된 135° / -35.26°.
  const V0=CFG.view||{};
  const AZ0=(V0.az!==undefined?V0.az:135.0)*Math.PI/180;
  const EL0=(V0.el!==undefined?V0.el:-35.26)*Math.PI/180;
  const FX0=V0.fx?-1:1, FY0=V0.fy?-1:1;
  let az=AZ0, el=EL0, zoom=1, cur=0, play=false, showBase=true, showVec=true,
      showRot=true, frm=(CFG.world_R?'world':'base'), timer=null;
  // 화면축 반전(거울). 다른 도구의 렌더와 좌우/상하를 맞춰볼 때 쓴다.
  // 주의: 반전은 회전이 아니라 **반사**라 det<0 인 좌우 뒤집힌 시점이 된다(아래 행렬에 표시).
  let flipX = FX0, flipY = FY0;

  const box=document.createElement('div'); box.className='scene';
  const cv=document.createElement('canvas'); box.appendChild(cv);
  const cc=document.createElement('div'); cc.className='ctrlwrap';
  cc.innerHTML=`<div class="ctrls">
      <button data-a="play" class="play">▶ 재생</button>
      <input type="range" min="0" max="${N-1}" value="0" aria-label="프레임">
      <span class="fr"></span>
    </div>
    <div class="ctrls row2">
      <button data-a="base" aria-pressed="true">base 궤적</button>
      <button data-a="vec" aria-pressed="true">교정 명령 δ</button>
      <button data-a="rot" aria-pressed="true">자세(회전)</button>
      <button data-a="frame" aria-pressed="true"></button>
      <button data-a="flipx" aria-pressed="${FX0<0}">↔ 좌우반전</button>
      <button data-a="flipy" aria-pressed="${FY0<0}">↕ 상하반전</button>
      <button data-a="reset">시점 초기화</button>
    </div>`;
  box.appendChild(cc);
  const hint=document.createElement('div'); hint.className='hint';
  hint.textContent='드래그 = 자유 회전(상하 제한 없음) · 휠 = 확대/축소 · 슬라이더 = 프레임 이동. 좌표는 에피소드 '+
    '시작점 기준 mm. 자세: 짧은 틱 = tool 접근축(Z), 현재 프레임은 X·Y·Z 축 전체(Z가 굵음). '+
    'world 기준 = 로봇 base 의 기울기를 보정한 시점(평행이동은 생략 — 상대 기하 동일). '+
    '교정 명령 δ 는 base 예측 지점 기점(현재 pose 기점 아님). '+
    '기본 시점은 residual_playback.html 과 동일(Plotly 기본 카메라 정합).';
  box.appendChild(hint);
  const matEl=document.createElement('div'); matEl.className='mat';
  box.appendChild(matEl);
  host.appendChild(box);
  const rng=cc.querySelector('input'), lab=cc.querySelector('.fr'), ctx=cv.getContext('2d');

  // 시점 회전 -> [화면x, 화면y, 깊이]
  function rot(p){
    const ca=Math.cos(az), sa=Math.sin(az), ce=Math.cos(el), se=Math.sin(el);
    const x1= p[0]*ca + p[1]*sa, y1= -p[0]*sa + p[1]*ca, z1=p[2];
    // [화면x, 화면y(아래로 +), 깊이].  flip 은 화면축 부호만 뒤집는다(깊이는 그대로).
    return [flipX*x1, flipY*(-(-y1*se + z1*ce)), y1*ce + z1*se];
  }
  // 현재 시점을 3x3 행렬로: [화면x; 화면y; 깊이] = M · (표시 좌표계 점)
  function viewMatrix(){
    const ca=Math.cos(az), sa=Math.sin(az), ce=Math.cos(el), se=Math.sin(el);
    return [[ flipX*ca,        flipX*sa,        0        ],
            [ flipY*(-sa*se),  flipY*( ca*se),  flipY*(-ce)],
            [ -sa*ce,          ca*ce,           se        ]];
  }
  // base -> world 변환(로봇이 X축으로 world_rot_x_deg 만큼 기울어져 있음).
  // 순수 회전이라 위치·자세축(방향벡터)에 똑같이 적용하면 된다(평행이동은 상대 기하에 무관).
  const RW=CFG.world_R;
  const mv=(M,p)=>[M[0][0]*p[0]+M[0][1]*p[1]+M[0][2]*p[2],
                   M[1][0]*p[0]+M[1][1]*p[1]+M[1][2]*p[2],
                   M[2][0]*p[0]+M[2][1]*p[1]+M[2][2]*p[2]];
  const xf=p=>(frm==='world'&&RW)?mv(RW,p):p;

  // 전체 점의 바운딩으로 스케일/중심 결정(시점을 돌려도 안 튀게 3D 중심 고정).
  // 좌표계를 바꾸면 바운딩도 달라지므로 다시 계산한다.
  let ctr=[0,0,0], span=1;
  function bounds(){
    const all=P.concat(S).map(xf);
    ctr=[0,1,2].map(k=>(Math.min(...all.map(p=>p[k]))+Math.max(...all.map(p=>p[k])))/2);
    span=Math.max(...[0,1,2].map(k=>
      Math.max(...all.map(p=>p[k]))-Math.min(...all.map(p=>p[k]))),1)||1;
  }
  bounds();
  let W=0,H=0,sc=1;
  const css=k=>getComputedStyle(document.documentElement).getPropertyValue(k).trim();

  function fit(){
    const r=cv.getBoundingClientRect(), d=devicePixelRatio||1;
    W=r.width; H=r.height; cv.width=W*d; cv.height=H*d;
    ctx.setTransform(d,0,0,d,0,0); sc=Math.min(W,H)*0.82/span;
  }
  // to2w: 이미 표시 좌표계인 점 / to2: 로깅(base) 좌표계 점
  const to2w=w=>{const q=rot([w[0]-ctr[0],w[1]-ctr[1],w[2]-ctr[2]]);
    return [W/2+q[0]*sc*zoom, H/2+q[1]*sc*zoom, q[2]];};
  const to2=p=>to2w(xf(p));

  function draw(){
    const C={s1:css('--s1'),s2:css('--s2'),aqua:css('--s3'),grid:css('--grid'),
             muted:css('--muted'),surf:css('--surface')};
    ctx.clearRect(0,0,W,H);
    const p2=P.map(to2), s2=S.map(to2);

    // 바닥 그리드(현재 시점의 XY 평면) — 깊이감
    const zmin=Math.min(...P.concat(S).map(p=>xf(p)[2]))-ctr[2];
    ctx.strokeStyle=C.grid; ctx.lineWidth=1;
    const g=span/2, st=span/4;
    for(let i=-2;i<=2;i++){
      for(const seg of [[[i*st,-g,zmin],[i*st,g,zmin]],[[-g,i*st,zmin],[g,i*st,zmin]]]){
        const a=to2w([seg[0][0]+ctr[0],seg[0][1]+ctr[1],seg[0][2]+ctr[2]]);
        const b=to2w([seg[1][0]+ctr[0],seg[1][1]+ctr[1],seg[1][2]+ctr[2]]);
        ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke();
      }
    }
    // 그릴 것들을 깊이 순으로(먼 것 먼저) 정렬 = painter's algorithm
    const items=[];
    for(let i=0;i<N-1;i++){
      if(showBase) items.push({d:(s2[i][2]+s2[i+1][2])/2,a:s2[i],b:s2[i+1],
        c:C.aqua,w:1.5,al:.55});
      items.push({d:(p2[i][2]+p2[i+1][2])/2,a:p2[i],b:p2[i+1],
        c:ep.isint[i]?C.s2:C.s1,w:ep.isint[i]?3:2,al:1});
      // 교정 명령 delta: **base 예측 지점에 붙인다**(현재 pose 기점이 아니라).
      //   base 목표 → base 목표 ⊕ δ  = head 가 냈을 때 명령이 가리킬 지점.
      //   (base→achieved 를 그리면 추종지연이 섞인 raw gap 이라 교정량이 아니다.)
      if(showVec && ep.isint[i] && i%2===0){
        const t2=to2([S[i][0]+D3[i][0], S[i][1]+D3[i][1], S[i][2]+D3[i][2]]);
        items.push({d:(t2[2]+s2[i][2])/2,a:s2[i],b:t2,c:C.s2,w:1.4,al:.6});
      }
    }
    items.sort((x,y)=>y.d-x.d);
    for(const it of items){
      ctx.globalAlpha=it.al; ctx.strokeStyle=it.c; ctx.lineWidth=it.w;
      ctx.lineCap='round'; ctx.beginPath();
      ctx.moveTo(it.a[0],it.a[1]); ctx.lineTo(it.b[0],it.b[1]); ctx.stroke();
    }
    ctx.globalAlpha=1;

    // ── 자세(회전) ──
    // 궤적 위: tool 접근축(회전행렬 3번째 축) 틱을 일정 간격으로 -> 자세 변화 추이.
    // 현재 프레임: base/achieved 각각 X·Y·Z 축 triad 를 크게 -> 회전 교정량이 보인다.
    const AX=span*0.062, AXC=span*0.15;
    const axpt=(p,rot,k,len)=>[p[0]+rot[k*3]*len, p[1]+rot[k*3+1]*len, p[2]+rot[k*3+2]*len];
    if(showRot){
      const K=Math.max(1,Math.round(N/20));
      ctx.lineWidth=1.5;
      for(let i=0;i<N;i+=K){
        for(const [src,rot,col,al] of [[P[i],ep.rota[i],ep.isint[i]?C.s2:C.s1,.85],
                                       [S[i],ep.rotb[i],C.aqua,.5]]){
          if(src===S[i] && !showBase) continue;
          const a0=to2(src), a1=to2(axpt(src,rot,2,AX));
          ctx.globalAlpha=al; ctx.strokeStyle=col;
          ctx.beginPath(); ctx.moveTo(a0[0],a0[1]); ctx.lineTo(a1[0],a1[1]); ctx.stroke();
        }
      }
      ctx.globalAlpha=1;
      // 현재 프레임 triad (축 식별은 굵기+끝 라벨로 — 색은 엔티티 정체성 유지)
      ctx.font='600 10px system-ui,-apple-system,sans-serif';
      for(const [src,rot,col] of [[S[cur],ep.rotb[cur],C.aqua],
                                  [P[cur],ep.rota[cur],ep.isint[cur]?C.s2:C.s1]]){
        if(src===S[cur] && !showBase) continue;
        const o0=to2(src);
        ['X','Y','Z'].forEach((nm,k)=>{
          const e1=to2(axpt(src,rot,k,AXC));
          ctx.strokeStyle=col; ctx.lineWidth=k===2?3:1.75; ctx.globalAlpha=k===2?1:.75;
          ctx.beginPath(); ctx.moveTo(o0[0],o0[1]); ctx.lineTo(e1[0],e1[1]); ctx.stroke();
          ctx.globalAlpha=1; ctx.fillStyle=C.muted;
          ctx.fillText(nm, e1[0]+3, e1[1]-3);
        });
      }
    }

    // 현재 프레임
    //   흐린 회색선 = base 목표 → achieved (raw gap, 대부분 추종지연 — 참고용)
    //   주황 굵은선 = base 목표 → base 목표 ⊕ 교정명령 δ (head 가 내야 할 값)
    const a=s2[cur], b=p2[cur];
    const dEnd=to2([S[cur][0]+D3[cur][0], S[cur][1]+D3[cur][1], S[cur][2]+D3[cur][2]]);
    ctx.strokeStyle=C.grid; ctx.lineWidth=1.5; ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke();
    ctx.strokeStyle=C.s2; ctx.lineWidth=3;
    ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(dEnd[0],dEnd[1]); ctx.stroke();
    ctx.beginPath(); ctx.arc(dEnd[0],dEnd[1],5,0,7); ctx.fillStyle=C.s2; ctx.fill();
    ctx.lineWidth=2; ctx.strokeStyle=C.surf; ctx.stroke();
    for(const [pt,col,r] of [[a,C.aqua,4],[b,ep.isint[cur]?C.s2:C.s1,6]]){
      ctx.beginPath(); ctx.arc(pt[0],pt[1],r,0,7); ctx.fillStyle=col; ctx.fill();
      ctx.lineWidth=2; ctx.strokeStyle=C.surf; ctx.stroke();
    }
    updateMat();
    lab.textContent=`${cur} / ${N-1} · ${(cur*ep.dt).toFixed(1)}s · `+
      `${ep.res_mm[cur].toFixed(1)}mm · ${ep.rres_deg[cur].toFixed(1)}°`+
      (ep.isint[cur]?' · 개입':'');
  }
  const mul3=(A,B)=>A.map(r=>[0,1,2].map(j=>r[0]*B[0][j]+r[1]*B[1][j]+r[2]*B[2][j]));
  const det3=M=>M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
               -M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
               +M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]);
  const f3=v=>(v>=0?' ':'-')+Math.abs(v).toFixed(3);
  function updateMat(){
    const V=viewMatrix();
    // 로깅(base) 좌표 -> 화면. world 표시 중이면 world_R 을 먼저 태운다.
    const M=(frm==='world'&&RW)?mul3(V,RW):V;
    const d=det3(M);
    const rows=['화면x','화면y','깊이 '];
    const body=M.map((r,i)=>`  ${rows[i]}  [${r.map(f3).join('  ')}]`).join('\n');
    matEl.innerHTML =
      `<b>시점</b>  az ${(az*180/Math.PI).toFixed(1)}°  el ${(el*180/Math.PI).toFixed(1)}°  `+
      `zoom ${zoom.toFixed(2)}×  ·  좌표계 ${frm}${(frm==='world'&&RW)?` (Rx ${CFG.world_rot_x_deg}° 포함)`:''}  ·  `+
      `반전 ${flipX<0?'좌우':''}${flipX<0&&flipY<0?'+':''}${flipY<0?'상하':''}${flipX>0&&flipY>0?'없음':''}\n`+
      `<b>M</b> (${frm} 좌표 → 화면, 화면y 는 아래가 +)\n${body}\n`+
      `  det ${f3(d)} ` + (d<0 ? '<span class="warn">← 반사(거울) 시점 — 손대칭이 뒤집혀 보임</span>'
                              : '(정상 회전)');
  }
  const redraw=()=>{fit();draw();};

  // 상호작용
  let drag=null;
  cv.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];
    try{cv.setPointerCapture(e.pointerId);}catch(_){}});
  cv.addEventListener('pointermove',e=>{
    if(!drag) return;
    // 제한 없는 자유 회전(트랙볼). 위/아래로 90°를 넘어가면 뒤집힌 시점이 되는 게 정상.
    // 각도는 ±π 로 감아 부동소수 누적만 막는다.
    const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));
    az=wrap(az+(e.clientX-drag[0])*0.01); el=wrap(el+(e.clientY-drag[1])*0.01);
    drag=[e.clientX,e.clientY]; draw();});
  cv.addEventListener('pointerup',()=>drag=null);
  cv.addEventListener('pointercancel',()=>drag=null);
  cv.addEventListener('wheel',e=>{e.preventDefault();
    zoom=Math.max(.4,Math.min(6,zoom*(e.deltaY<0?1.12:1/1.12))); draw();},{passive:false});
  rng.addEventListener('input',()=>{cur=+rng.value; draw();});
  cc.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b) return;
    const a=b.dataset.a;
    if(a==='base'){showBase=!showBase; b.setAttribute('aria-pressed',showBase);}
    if(a==='vec'){showVec=!showVec; b.setAttribute('aria-pressed',showVec);}
    if(a==='rot'){showRot=!showRot; b.setAttribute('aria-pressed',showRot);}
    if(a==='frame'){frm=(frm==='world')?'base':'world'; bounds(); syncFrameBtn();}
    if(a==='flipx'){flipX=-flipX; b.setAttribute('aria-pressed',flipX<0);}
    if(a==='flipy'){flipY=-flipY; b.setAttribute('aria-pressed',flipY<0);}
    if(a==='reset'){az=AZ0; el=EL0; zoom=1; flipX=FX0; flipY=FY0;
      cc.querySelector('[data-a=\"flipx\"]').setAttribute('aria-pressed',FX0<0);
      cc.querySelector('[data-a=\"flipy\"]').setAttribute('aria-pressed',FY0<0);}
    if(a==='play'){
      play=!play; b.textContent=play?'❚❚ 일시정지':'▶ 재생';
      clearInterval(timer);
      if(play) timer=setInterval(()=>{cur=(cur+1)%N; rng.value=cur; draw();},
        Math.max(20,ep.dt*1000/2));
    }
    draw();});
  const fbtn=cc.querySelector('[data-a="frame"]');
  function syncFrameBtn(){
    const w=(frm==='world');
    fbtn.textContent = w ? `world 기준 (X ${CFG.world_rot_x_deg}°)` : 'base 기준 (로깅 그대로)';
    fbtn.setAttribute('aria-pressed', w);
    fbtn.title = RW ? `world = Rx(${CFG.world_rot_x_deg}°)·base — 로봇 base 가 기울어진 만큼 보정`
                    : 'world 회전 미설정';
  }
  if(!RW) fbtn.style.display='none'; else syncFrameBtn();
  updateMat();
  addEventListener('resize',redraw);
  new MutationObserver(draw).observe(document.documentElement,{attributes:true});
  matchMedia('(prefers-color-scheme:dark)').addEventListener('change',draw);
  requestAnimationFrame(redraw);
}

function legend(items){
  const d=document.createElement('div'); d.className='legend';
  d.innerHTML=items.map(([c,t,line])=>
    `<span><i class="sw${line?' line':''}" style="background:${c}"></i>${esc(t)}</span>`).join('');
  return d;
}
function chartBlock(parent,title,note,leg,draw){
  const w=document.createElement('div'); w.className='chart';
  w.innerHTML=`<p class="ctitle">${esc(title)}</p><p class="cnote">${esc(note)}</p>`;
  if(leg) w.appendChild(legend(leg));
  parent.appendChild(w); draw(w);
}

const _pct = a => a.map(x=>(100*x).toFixed(0)+'%').join(' / ');
document.getElementById('lagnote').innerHTML =
  `<b>추종지연을 뺀 값으로 잽니다.</b> 임피던스 팔은 한 tick 에 명령 스텝의 `+
  `<b>${_pct(CFG.lag.alpha.slice(0,3))}</b>(축별 x/y/z, 회전 ${_pct(CFG.lag.alpha.slice(3))})만 도달해서, `+
  `raw residual = T_base⁻¹·T_achieved(t+1) 의 <b>${(100*CFG.lag.r2_t).toFixed(0)}%</b>가 `+
  `사람 교정이 아니라 명령 스텝에 비례하는 지연항이다(nominal ${CFG.lag.n} 프레임으로 적합, `+
  `R² 병진 ${CFG.lag.r2_t} / 회전 ${CFG.lag.r2_r}). 명령 스텝(base_action_rel)은 head 의 입력이라 `+
  `그대로 학습시키면 head 는 base 의 전진을 상쇄하는 자명한 사상을 배우고 추론 시 팔이 느려진다. `+
  `그래서 이 리포트의 <b>교정량·coherence 는 모두 지연 제거 후</b> 값이다. `+
  `<b>판정은 학습 타깃 공간(e)</b>에서 한다 — head 가 회귀하는 값의 크기·일관성이 곧 학습 가능성이라서다. `+
  `<br><br><b>그리고 명령 공간으로 환산해서 보여준다.</b> 사람이 실제로 밀어낸 변위 e 는 head 가 낼 값이 `+
  `아니다 — 명령한 만큼의 α 만 도달하므로, 같은 변위를 만들려면 <b>δ = α⁻¹·e</b> 를 명령해야 한다 `+
  `(α 는 축마다 달라 스칼라가 아닌 <b>행렬</b>로 환산; α⁻¹ 대각 = ${CFG.lag.alpha_inv.join(', ')}). `+
  `즉 표시되는 "교정 명령"은 <b>base 예측 위에 실제로 얹힐 delta</b>이고, actor 의 residual 캡`+
  `(병진 50mm / 회전 23°)과 같은 공간이다. <b>누설</b> = nominal 프레임에 남은 잔여 편향`+
  `(0 에 가까워야 정상).`+
  (DATA.some(e=>e.has_head)
    ? `<br><br><b>head 가 켜진 라운드.</b> 실행 명령이 base ⊕ δ 라 relabel 타깃 e 에는 `+
      `<b>head 기여분과 사람 추가분이 함께</b> 들어간다(DAgger 누적으로는 정상). `+
      `기록된 residual_cmd6 로 둘을 분리해 보여준다 — head 기여 = α·δ, 사람 추가 = e − α·δ. `+
      `게인이 1 보다 작으면 head 가 낸 예측의 일부만 실제로 얹히므로 사람 추가분이 계속 남는다.`
    : '');

const root=document.getElementById('root');
DATA.forEach(ep=>{
  const S=ep.summary;
  const card=document.createElement('div'); card.className='card';
  card.innerHTML=`<div class="ephead"><h2>${esc(ep.name)}</h2>
    <span class="meta">${ep.T} frames · ${S.total_dur}s · ${CFG.hz}Hz</span>
    <span class="pill ${S.ok?'ok':'no'}">${S.ok
      ? '✓ 교정 신호 있음 — '+[S.ok_t?'병진':null,S.ok_r?'회전':null].filter(Boolean).join('+')
      : '✗ 교정 신호 부족'}</span></div>`;

  const tiles=document.createElement('div'); tiles.className='tiles';
  const mk=(k,v,u,n)=>`<div class="tile"><div class="k">${esc(k)}</div>
    <div class="v">${esc(v)}${u?`<small>${esc(u)}</small>`:''}</div>
    <div class="n">${esc(n)}</div></div>`;
  tiles.innerHTML=
    mk('개입 비율',S.pct,'%',`${S.n_int} / ${ep.T} frames`)+
    mk('개입 시간',S.dur,'s',`전체 ${S.total_dur}s 중`)+
    mk('쓸 만한 구간',`${S.n_ok_spans}/${S.n_spans}`,'','span 단위 판정')+
    mk('교정 명령 · 병진',S.cmd_int,'mm',
       `base 스텝 ${S.base_step_int}mm 의 ${S.frac_of_base}% · 캡 50mm`)+
    mk('교정량(타깃) · 병진',S.mag_int,'mm',
       `기준 ≥${CFG.pass_mag} · 누설 ${S.leak} · coh ${S.coh_int}(기준 ≥${CFG.pass_coh})`)+
    mk('교정 명령 · 회전',S.rcmd_int,'°',
       `타깃 ${S.rmag_int}° (기준 ≥${CFG.pass_rmag}) · coh ${S.rcoh_int}`)+
    (S.has_head
      ? mk('head 기여율',S.head_frac,'%',`head ${S.head_mag}mm / 필요 ${S.mag_int}mm`)+
        mk('head 방향일치',S.head_cos,'',`cos(head 예측, 필요 교정) · 1 이면 완전 일치`)+
        mk('nominal 오출력',S.head_nom,'mm',`안 밀 때 head 가 낸 양 · 0 에 가까워야 정상`)
      : '');
  card.appendChild(tiles);

  chartBlock(card,'3D 궤적',
    '주황 벡터는 base(slow) 예측 지점에서 시작해 <교정 명령 δ>만큼 뻗는다 — head 가 그 tick 에 '+
    '내야 할 값이고, 끝점이 교정된 명령이 가리킬 곳이다. 회색 실선(현재 프레임)은 base→achieved '+
    'raw gap 으로, 대부분 추종지연이라 교정량이 아니다(참고용).',
    [['var(--s1)','achieved · nominal',1],['var(--s2)','교정 명령 δ / achieved·개입',1],
     ['var(--s3)','base(slow) 목표',1]],
    h=>scene3d(h,ep));

  if(ep.has_head) chartBlock(card,'head 기여 vs 사람 추가',
    '필요했던 교정(파랑) 중 head 가 실제로 만든 변위(청록). 둘의 간격이 사람이 더 얹은 몫이다. '+
    'head 가 잘 배웠다면 개입 구간에서 청록이 파랑을 따라 올라가고, nominal 에서는 둘 다 0 에 붙는다.',
    [['var(--s1)','필요 교정 ‖e‖',1],['var(--s3)','head 가 만든 ‖α·δ‖',1],['var(--s2)','개입 구간(a→b)',0]],
    h=>lineChart(h,{series:[{vals:ep.etot_mm,color:'var(--s1)',name:'필요 교정'},
                            {vals:ep.ehead_mm,color:'var(--s3)',name:'head 기여'}],
                    isint:ep.isint,dt:ep.dt,unit:'mm',fmt:2}));

  chartBlock(card,'교정 명령 타임라인 · 병진',
    'head 가 base 예측 위에 얹어야 할 delta(파랑)를, base 가 그 tick 에 명령한 스텝(청록)과 같은 '+
    '축에서 비교한다 — 둘 다 명령 공간이라 "base 가 이만큼 가라 할 때 이만큼 더 밀어야 한다"로 읽힌다.',
    [['var(--s1)','교정 명령(head 출력)',1],['var(--s3)','base 명령 스텝',1],['var(--s2)','개입 구간(a→b)',0]],
    h=>lineChart(h,{series:[{vals:ep.cor_mm,color:'var(--s1)',name:'교정 명령'},
                            {vals:ep.base_mm,color:'var(--s3)',name:'base 스텝'}],
                    isint:ep.isint,dt:ep.dt,unit:'mm',fmt:1}));

  chartBlock(card,'교정 명령 타임라인 · 회전',
    'head 가 내야 할 자세 delta(명령 공간). 회전은 α 가 더 작아(한 tick 8%) 같은 실측 회전이라도 '+
    '명령은 더 크게 나가야 한다.',
    [['var(--s1)','교정 명령(head 출력)',1],['var(--s2)','개입 구간(a→b)',0]],
    h=>lineChart(h,{series:[{vals:ep.rcor_deg,color:'var(--s1)',name:'교정 명령'}],
                    isint:ep.isint,dt:ep.dt,unit:'°',fmt:1}));

  chartBlock(card,'실제 팔 이동량','tick 당 achieved 이동 — 밀었는데 안 움직이면 임피던스 stiffness 가 너무 높다.',
    [['var(--s1)','이동량',1],['var(--s2)','개입 구간(a→b)',0]],
    h=>lineChart(h,{series:[{vals:ep.move_mm,color:'var(--s1)',name:'이동량'}],
                    isint:ep.isint,dt:ep.dt,unit:'mm/tick',fmt:1}));

  chartBlock(card,'손목 힘','‖F‖ — 개입 중 힘이 올라갔는지(손목보다 위쪽 링크를 밀면 안 잡힐 수 있음).',
    [['var(--s1)','‖F‖',1],['var(--s2)','개입 구간(a→b)',0]],
    h=>lineChart(h,{series:[{vals:ep.force_n,color:'var(--s1)',name:'‖F‖'}],
                    isint:ep.isint,dt:ep.dt,unit:'N',fmt:1}));

  if(ep.spans.length) chartBlock(card,'구간별 평균 교정 명령 · 병진',
    '지연 제거 후 값. 한 구간 안에서 축 부호가 뒤집히면(밀었다 당겼다) 평균이 상쇄돼 학습 신호가 '+
    '사라진다. 구간이 여러 개면 에피소드 평균보다 구간별 판정(표)이 실질적이다.',
    [['var(--s1)','+ 방향',0],['var(--neg)','− 방향',0]], h=>vecChart(h,ep.spans,'vec','mm'));
  if(ep.spans.length) chartBlock(card,'구간별 평균 교정 명령 · 회전',
    'base 자세 대비 사람이 돌린 양(rotvec, 도). 병진 없이 회전만 교정한 구간도 여기서 드러난다.',
    [['var(--s1)','+ 방향',0],['var(--neg)','− 방향',0]], h=>vecChart(h,ep.spans,'rvec','°'));

  const det=document.createElement('details');
  det.innerHTML=`<summary>구간별 수치 (표)</summary><div class="tblwrap"><table>
    <thead><tr><th>구간</th><th>판정</th><th>frame</th><th>길이</th><th>병진 명령 ‖평균‖</th><th>병진 coh</th>
    <th>X</th><th>Y</th><th>Z</th><th>회전 ‖평균벡터‖</th><th>회전 coh</th>
    <th>rX</th><th>rY</th><th>rZ</th><th>raw(지연포함)</th><th>이동량</th><th>‖F‖</th></tr></thead>
    <tbody>${ep.spans.map(s=>`<tr><td>구간 ${s.idx}</td>
      <td style="color:${s.ok?'var(--good)':'var(--crit)'}">${s.ok
        ? '✓ '+[s.ok_t?'병진':null,s.ok_r?'회전':null].filter(Boolean).join('+') : '✗'}</td>
      <td>${s.start}–${s.end}</td>
      <td>${s.dur}s</td><td>${s.mag} mm</td><td>${s.coh}</td>
      <td>${s.vec[0]}</td><td>${s.vec[1]}</td><td>${s.vec[2]}</td>
      <td>${s.rmag}°</td><td>${s.rcoh}</td>
      <td>${s.rvec[0]}</td><td>${s.rvec[1]}</td><td>${s.rvec[2]}</td>
      <td>${s.rawmag} mm</td>
      <td>${s.move} mm/tick</td><td>${s.force} N</td></tr>`).join('')}</tbody></table></div>`;
  card.appendChild(det);
  root.appendChild(card);
});
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description="개입 수집 품질 리포트 HTML.")
    ap.add_argument("--episodes", nargs="+",
                    default=["data/online_runs/run_hand_intervention/transitions"],
                    help="transitions 디렉토리 또는 *.hdf5 파일들(여러 개 가능).")
    ap.add_argument("--out", default="data/verify_intervention")
    ap.add_argument("--steps_per_inference", type=int, default=6,
                    help="actor 의 --steps_per_inference (nominal chunk 위상 계산용).")
    ap.add_argument("--frequency", type=float, default=10.0, help="actor 의 --frequency (Hz).")
    ap.add_argument("--world_rot_x_deg", type=float, default=135.0,
                    help="base→world 변환용 X축 회전각(도). 로봇 base 가 X축으로 이만큼 기울어져 "
                         "있다는 뜻. 3D 패널에 base/world 토글로 들어간다(기본 world). "
                         "export_traj3d_playback_html.py 와 같은 규약 — 부호가 반대로 보이면 뒤집을 것.")
    # 합격 기준은 '지연 제거 후' 값 기준. 관측된 분포(5 에피소드): 좋은 구간의 병진 교정량은
    # 1.2~1.6mm, coherence 0.74~0.77 / 신호 없는 구간은 0.33mm, 0.21.
    ap.add_argument("--view_az", type=float, default=135.0,
                    help="3D 기본 시점 방위각(도). 기본 135 = residual_playback.html 과 동일.")
    ap.add_argument("--view_el", type=float, default=-35.26,
                    help="3D 기본 시점 고도각(도).")
    ap.add_argument("--view_flip_x", action="store_true", help="3D 기본 시점 좌우반전(거울, det<0).")
    ap.add_argument("--view_flip_y", action="store_true", help="3D 기본 시점 상하반전.")
    ap.add_argument("--pass_mag", type=float, default=0.8,
                    help="합격: 개입 병진 교정량 ‖평균벡터‖(mm) >= . **학습 타깃 공간**(실측 변위) "
                         "기준 — 판정은 head 가 회귀하는 값의 크기·일관성으로 한다.")
    ap.add_argument("--pass_rmag", type=float, default=0.12,
                    help="합격: 개입 회전 교정량 ‖평균벡터‖(deg) >= (학습 타깃 공간)")
    ap.add_argument("--pass_coherence", type=float, default=0.5,
                    help="합격: 개입 coherence >= (지연 제거 후엔 raw 보다 낮게 나온다)")
    a = ap.parse_args()

    files = []
    for p in a.episodes:
        src = p if os.path.isabs(p) else os.path.join(ROOT, p)
        files += sorted(glob.glob(os.path.join(src, "*.hdf5"))) if os.path.isdir(src) else [src]
    dt = 1.0 / a.frequency

    # 1) 원시 로드 -> 2) nominal 전체로 추종지연 모델 적합 -> 3) 지연 제거 타깃으로 분석
    raws = []
    for fp in files:
        try:
            r = load_raw(fp)
        except Exception as e:
            print(f"[report] 스킵 {os.path.basename(fp)}: {e}")
            continue
        if r is None:
            print(f"[report] 스킵 {os.path.basename(fp)}: is_intervention 없음(개입형 데이터 아님)")
            continue
        raws.append(r)
    if not raws:
        print("[report] 개입 에피소드 없음."); return

    try:
        lag = fit_lag(raws)
    except Exception as e:
        print(f"[report] 지연 모델 적합 실패 — 분석 불가: {e}")
        return
    print(f"[report] 추종지연 모델: {lag_model.describe(lag)}")
    print(f"[report]   → raw residual 의 {100*lag['r2_t']:.0f}% 는 사람 교정이 아니라 "
          f"명령 스텝에 비례하는 지연항. 아래 수치는 이걸 뺀 '교정 성분'.")

    eps = []
    for raw in raws:
        r = analyze(raw, lag, a.steps_per_inference, dt,
                    a.pass_mag, a.pass_rmag, a.pass_coherence)
        s = r["summary"]
        tag = ("PASS(" + "+".join([x for x in ["병진" if s['ok_t'] else None,
                                                "회전" if s['ok_r'] else None] if x]) + ")"
               ) if s['ok'] else "FAIL"
        print(f"[report] {r['name']}: {r['T']} frames, 개입 {s['pct']}% ({s['n_spans']} span) | "
              f"교정 병진 {s['mag_int']}mm coh {s['coh_int']} (누설 {s['leak']}mm) "
              f"→ 명령 {s['cmd_int']}mm (base 스텝 {s['base_step_int']}mm 의 {s['frac_of_base']}%) | "
              f"회전 {s['rmag_int']}° coh {s['rcoh_int']} → 명령 {s['rcmd_int']}° -> {tag}")
        if s['has_head']:
            print(f"           [head] 기여 {s['head_mag']}mm / 필요 {s['mag_int']}mm "
                  f"({s['head_frac']}%) · 사람 추가 {s['human_mag']}mm · "
                  f"방향일치 cos {s['head_cos']} · nominal 오출력 {s['head_nom']}mm")
        for sp in r['spans']:
            mark = ("✓" + ("병진" if sp['ok_t'] else "") + ("회전" if sp['ok_r'] else "")) \
                   if sp['ok'] else "✗"
            print(f"           구간{sp['idx']} {sp['dur']}s  명령 병진 {sp['mag']}mm/coh {sp['coh']}  "
                  f"회전 {sp['rmag']}°/coh {sp['rcoh']}  (raw {sp['rawmag']}mm)  {mark}")
        eps.append(r)

    if not eps:
        print("[report] 개입 에피소드 없음."); return

    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)
    th = np.radians(a.world_rot_x_deg)
    c, sn = float(np.cos(th)), float(np.sin(th))
    cfg = dict(hz=a.frequency, spi=a.steps_per_inference,
               pass_mag=a.pass_mag, pass_rmag=a.pass_rmag, pass_coh=a.pass_coherence,
               lag=dict(alpha=[round(float(x), 3) for x in np.diag(lag['alpha'])],
                        alpha_inv=[round(float(x), 2) for x in np.diag(lag['alpha_inv'])],
                        r2_t=round(lag['r2_t'], 3), r2_r=round(lag['r2_r'], 3),
                        n=lag['n'], cond=round(lag['cond'], 1)),
               world_rot_x_deg=a.world_rot_x_deg,
               view=dict(az=a.view_az, el=a.view_el,
                         fx=bool(a.view_flip_x), fy=bool(a.view_flip_y)),
               world_R=[[1.0, 0.0, 0.0], [0.0, c, -sn], [0.0, sn, c]])
    html = (HTML.replace("__DATA__", json.dumps(eps, ensure_ascii=False))
                .replace("__CFG__", json.dumps(cfg)))
    fp = os.path.join(out, "intervention_report.html")
    with open(fp, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[report] HTML 저장: {fp}")


if __name__ == "__main__":
    main()
