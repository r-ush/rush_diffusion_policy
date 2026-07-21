#!/usr/bin/env python
"""시간 순 재생(playback)형 residual 타당성 HTML — "inference 시점에 맞게, 각 프레임에서
base(slow) 가 가려던 곳 / 사람이 실제로 간 곳(교정) / (선택)residual head 가 낼 곳"을
한 스텝씩 보여준다.

export_traj3d_html.py 는 에피소드 전체 궤적을 **한 번에** 정적으로 그린다. 이 스크립트는
같은 데이터를 **재생 슬라이더 + play** 로 감싸, step 을 넘길 때마다:
  * Panel A(3D 전체): achieved 궤적이 현재 프레임까지 자라나고 현재 위치 표시.
  * Panel B(3D 국소, cm): 현재 위치를 원점으로 base/human/(head) 목표 오프셋 벡터 비교
      → head(주황)가 human(파랑) 쪽으로 향하면 "그 순간의 교정이 타당"(#4).
  * Panel C(타임라인): ‖필요교정‖ (+선택 ‖head예측‖) (cm) 에 현재 프레임 커서.

두 가지 모드:
  (1) --head none  → **로깅 재생 모드**. transitions/*.hdf5 에 actor 가 inference 시점에
      로깅한 slow_pred_target_abs / virtual_target_abs / residual GT 만 읽어 그린다.
      base·head·torch 불필요(순수 numpy). "실제로 무슨 일이 있었나"에 가장 충실.
  (2) --head <weights_vN.pt>  → **head 오버레이 모드**. 그 head+slow base 를 로드해
      프레임별 head 예측 residual 을 계산해 3번째 선(head)으로 겹친다.
      ※ head 는 그 base(=slow_ckpt)로 학습된 것과 **입력차원이 일치**해야 한다.
        (로봇 v4 는 wrench 인코더(force_dim=512) base 로 학습됨 → 그 base 가 있어야 로드됨.)

실행 예:
  # (1) 로깅 재생 — 아무 의존성 없이 바로
  <py> online_learning/export_traj3d_playback_html.py \
      --episodes data/online_runs/run_hand_residual_abs/transitions \
      --heldout ep_00004 ep_00005 --head none --out data/verify_playback

  # (2) head 오버레이 (base 일치 필요)
  RESIDUAL_SLOW_CKPT=<matching base> RESIDUAL_CONFIG_NAME=residual_policy/hand_online_abs_mlp \
  RESIDUAL_ONLINE_WORKDIR=<scratch wd> <bae_py> .../export_traj3d_playback_html.py \
      --head <wd>/weights/weights_vN.pt --episodes .../transitions --out data/verify_playback
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

LOGGED_KEYS = ["slow_pred_target_abs", "virtual_target_abs",
               "residual_delta6_slow_pred_to_virtual", "robot_pose_R"]


def _ep_files(source):
    if os.path.isdir(source):
        return sorted(glob.glob(os.path.join(source, "*.hdf5")))
    return [source]


def _read_logged(fp):
    """transitions 에피소드에서 로깅된 배열들을 그대로 읽는다(재계산 없음).
    개입형(intervention) 데이터면 obs/is_intervention(per-step 0/1)도 함께 읽는다(있을 때만)."""
    with h5py.File(fp, "r") as f:
        if "data" not in f:
            return None
        dm = sorted(f["data"].keys())[0]
        o = f["data"][dm]["obs"]
        if any(k not in o for k in LOGGED_KEYS):
            return None
        out = {k: np.asarray(o[k]) for k in LOGGED_KEYS}
        if "is_intervention" in o:
            out["is_intervention"] = np.asarray(o["is_intervention"]).reshape(-1)
        return out


def _corr_stats(gt_norm_cm):
    a = np.asarray(gt_norm_cm, dtype=np.float64)
    return {"n": int(len(a)), "mean_cm": float(a.mean()), "median_cm": float(np.median(a)),
            "p90_cm": float(np.quantile(a, 0.9)), "max_cm": float(a.max())}


def _seg_metrics(gt, pred, is_int, metrics_fn):
    """개입형 분할 지표. intervention 프레임=head 가 밀림을 따라가야(cosine/capture),
    nominal(안 민) 프레임=head 가 조용해야(‖pred‖≈0). teleop 데이터엔 is_int 가 없어 호출 안 됨."""
    is_int = np.asarray(is_int).reshape(-1).astype(bool)
    out = {}
    if is_int.any():
        out["intervention"] = metrics_fn(gt, pred, is_int)
    nom = ~is_int
    if nom.any():
        pn = np.linalg.norm(pred[nom, :3], axis=1) * 100.0    # cm
        gn = np.linalg.norm(gt[nom, :3], axis=1) * 100.0
        out["nominal"] = {
            "n": int(nom.sum()),
            "gt_trans_mean_cm": float(gn.mean()),             # ~0 이어야 정상(안 밀었으니)
            "pred_trans_mean_cm": float(pn.mean()),           # head 가 조용한가(작을수록 좋음)
            "pred_trans_p90_cm": float(np.quantile(pn, 0.9)),
            "quiet_rate_lt1cm": float((pn < 1.0).mean()),     # nominal 에서 ‖pred‖<1cm 비율
        }
    return out


def build_episode_logged(fp, name, heldout, max_frames):
    lg = _read_logged(fp)
    if lg is None:
        return None
    n = len(lg["robot_pose_R"])
    if max_frames:
        n = min(n, max_frames)
    achieved = lg["robot_pose_R"][:n, :3]
    slow = lg["slow_pred_target_abs"][:n, :3]
    human = lg["virtual_target_abs"][:n, :3]
    gt = lg["residual_delta6_slow_pred_to_virtual"][:n]
    gn = np.linalg.norm(gt[:, :3], axis=1) * 100.0        # cm
    ep = {
        "name": name, "heldout": heldout, "n": int(n), "has_head": False,
        "achieved": achieved.round(5).tolist(),
        "slow": slow.round(5).tolist(),
        "human": human.round(5).tolist(),
        "gt_norm": gn.round(3).tolist(),
    }
    if "is_intervention" in lg:                            # 개입형 데이터만
        ep["is_int"] = (lg["is_intervention"][:n] > 0.5).astype(int).tolist()
    return ep, gn


def add_head_predictions(episodes_out, args, heldset):
    """head 오버레이 모드: slow base + head 로드 후 프레임별 head 예측 residual 을 계산해
    각 에피소드 dict 에 head/pred_norm/cosine/capture/metrics 를 채운다."""
    import torch
    from omegaconf import OmegaConf
    from diffusion_policy.common.pytorch_util import dict_apply
    from diffusion_policy.residual_policy.pose_util import apply_residual_action_to_pose9, pose_like_to_pose9
    from diffusion_policy.residual_policy.step_dataset import FastResidualContextStepDataset
    from online_learning.residual_teleop_learner import ResidualOnlineLearner
    from online_learning.verify_residual_on_data import metrics

    learner = ResidualOnlineLearner()
    dev = learner.device
    sd = torch.load(args.head, map_location=dev)
    learner.policy.head.load_state_dict(sd["head_state"])
    learner.policy.normalizer.load_state_dict(sd["normalizer_state"])
    if sd.get("force_encoder_state") is not None and learner.policy.force_encoder is not None:
        learner.policy.force_encoder.load_state_dict(sd["force_encoder_state"])
    learner.policy.to(dev).eval()
    print(f"[playback] head 로드: {args.head} (v{sd.get('version','?')}, demos={sd.get('num_demos','?')})", flush=True)

    ds_cfg = OmegaConf.to_container(learner.cfg.task.dataset, resolve=True)
    ds_cfg.pop("_target_"); ds_cfg["val_ratio"] = 0.0

    all_gt, all_pred, all_isint = [], [], []
    by_name = {e["name"]: e for e, _ in episodes_out}
    for fp in _ep_files(args.episodes):
        name = os.path.basename(fp).replace(".hdf5", "")
        if name not in by_name:
            continue
        lg = _read_logged(fp)
        ds_cfg["dataset_path"] = fp
        ds = FastResidualContextStepDataset(**ds_cfg)
        gts, preds = [], []
        with torch.no_grad():
            for b in torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False):
                b = dict_apply(b, lambda x: x.to(dev))
                preds.append(learner.policy.predict_action(b["obs"])["action"][:, 0].cpu().numpy())
                gts.append(b["action"][:, -1].cpu().numpy())
        gt = np.concatenate(gts); pred = np.concatenate(preds)
        n = len(gt)
        slow9 = pose_like_to_pose9(lg["slow_pred_target_abs"])[:n]
        head_pos = np.stack([apply_residual_action_to_pose9(slow9[i], pred[i])[:3] for i in range(n)])
        gn = np.linalg.norm(gt[:, :3], axis=1); pn = np.linalg.norm(pred[:, :3], axis=1)
        err = np.linalg.norm(gt[:, :3] - pred[:, :3], axis=1)
        cap = np.clip((gn - err) / (gn + 1e-9), -1, 1)
        cos = np.clip(np.sum(gt[:, :3] * pred[:, :3], axis=1) / (gn * pn + 1e-9), -1, 1)
        e = by_name[name]
        m = min(n, e["n"])
        e["has_head"] = True
        e["n"] = int(m)
        # 로깅 배열(길이 T)과 head 배열(길이 n=T-1)의 길이를 공통 m 으로 맞춤(프레임 정렬).
        e["achieved"] = e["achieved"][:m]
        e["slow"] = e["slow"][:m]
        e["human"] = e["human"][:m]
        e["gt_norm"] = e["gt_norm"][:m]
        e["head"] = head_pos[:m].round(5).tolist()
        e["pred_norm"] = (pn[:m] * 100).round(3).tolist()
        e["cosine"] = cos[:m].round(3).tolist()
        e["capture"] = cap[:m].round(3).tolist()
        e["metrics"] = metrics(gt, pred)
        all_gt.append(gt); all_pred.append(pred)
        if "is_int" in e:                                  # 개입형: 길이 m 로 맞추고 분할지표
            e["is_int"] = e["is_int"][:m]
            ii = np.asarray(e["is_int"]); L = len(ii)
            e["seg_metrics"] = _seg_metrics(gt[:L], pred[:L], ii, metrics)
            all_isint.append(ii)
        else:
            all_isint.append(None)
        print(f"[playback] head 예측 {name}: {n} frames"
              + (f"  (개입 {int(np.asarray(e['is_int']).sum())})" if 'is_int' in e else ""), flush=True)

    if all_gt:
        gt = np.concatenate(all_gt); pred = np.concatenate(all_pred)
        gnall = np.linalg.norm(gt[:, :3], axis=1)
        hard = gnall >= np.quantile(gnall, 0.75)
        hs = {"overall": metrics(gt, pred), "hard_top25pct": metrics(gt, pred, hard),
              "head_file": os.path.basename(args.head)}
        if any(x is not None for x in all_isint):          # 개입형 전체 분할지표
            gi = np.concatenate([g[:len(ii)] for g, ii in zip(all_gt, all_isint) if ii is not None])
            pi = np.concatenate([p[:len(ii)] for p, ii in zip(all_pred, all_isint) if ii is not None])
            ia = np.concatenate([ii for ii in all_isint if ii is not None])
            hs["by_intervention"] = _seg_metrics(gi, pi, ia, metrics)
        return hs
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True, help="transitions 디렉토리 또는 단일 *.hdf5")
    ap.add_argument("--head", default="none", help="'none'=로깅 재생 / 또는 weights_vN.pt(오버레이)")
    ap.add_argument("--heldout", nargs="*", default=[], help="held-out 뱃지 ep 이름")
    ap.add_argument("--out", default="data/verify_playback")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--world_rot_x_deg", type=float, default=-135.0,
                    help="base→world 변환용 X축 회전각(도). 로봇이 X축으로 이만큼 돌아가 있음. "
                         "HTML 에 base/world 토글로 들어감. 부호가 반대로 보이면 값 뒤집기.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    heldset = {h.replace(".hdf5", "") for h in args.heldout}
    logged_mode = args.head in (None, "none", "logged", "")

    episodes_out = []
    all_gn = []
    for fp in _ep_files(args.episodes):
        name = os.path.basename(fp).replace(".hdf5", "")
        built = build_episode_logged(fp, name, name in heldset, args.max_frames)
        if built is None:
            print(f"[skip] {name} (로깅 키 없음)", flush=True); continue
        episodes_out.append(built)
        all_gn.append(built[1])
        print(f"[playback] {name}: {built[0]['n']} frames  (heldout={name in heldset})", flush=True)

    head_summary = None
    if not logged_mode:
        head_summary = add_head_predictions(episodes_out, args, heldset)

    episodes = [e for e, _ in episodes_out]
    gn_all = np.concatenate(all_gn) if all_gn else np.zeros(1)
    th = np.radians(args.world_rot_x_deg); c, s = np.cos(th), np.sin(th)
    world_R = [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]   # p_world = Rx(deg) @ p_base
    has_int = any("is_int" in e for e in episodes)
    summary = {
        "n_episodes": len(episodes),
        "samples": int(sum(e["n"] for e in episodes)),
        "heldout": sorted(heldset),
        "mode": "logged" if logged_mode else "head_overlay",
        "has_intervention": has_int,
        "correction_stats": _corr_stats(gn_all),
        "head": head_summary,
        "world_rot_x_deg": args.world_rot_x_deg,
        "world_R": world_R,
    }
    if has_int:   # 개입/nominal 로 나눈 '필요교정(GT)' 크기 (head 없이도)
        G = np.concatenate([np.asarray(e["gt_norm"]) for e in episodes if "is_int" in e])
        I = np.concatenate([np.asarray(e["is_int"]) for e in episodes if "is_int" in e]).astype(bool)
        summary["intervention_gt_split"] = {
            "intervention": {"n": int(I.sum()), "gt_mean_cm": float(G[I].mean()) if I.any() else 0.0},
            "nominal": {"n": int((~I).sum()), "gt_mean_cm": float(G[~I].mean()) if (~I).any() else 0.0},
        }
    data = {"summary": summary, "episodes": episodes}
    with open(os.path.join(args.out, "traj3d_playback.json"), "w") as f:
        json.dump(data, f)
    print("[playback] traj3d_playback.json 저장", flush=True)

    import plotly
    plotly_js = os.path.join(os.path.dirname(plotly.offline.__file__), "..", "package_data", "plotly.min.js")
    with open(plotly_js) as f:
        PLOTLY = f.read()
    out_html = os.path.join(args.out, "residual_playback.html")
    with open(out_html, "w") as f:
        f.write(TEMPLATE.replace("/*PLOTLY*/", PLOTLY).replace("/*DATA*/", json.dumps(data)))
    print(f"[playback] HTML 저장: {out_html}", flush=True)


TEMPLATE = r"""<!DOCTYPE html><html lang=ko><head><meta charset=utf-8>
<title>Residual Policy 재생(playback) 타당성</title>
<script>/*PLOTLY*/</script>
<style>
 :root{--bg:#0f1115;--card:#171a21;--fg:#e6e9ef;--mut:#9aa4b2;--line:#262b36;
       --achv:#2ecc71;--slow:#8892a0;--human:#4c8dff;--head:#ff9f43;--key:#1f2937;--warn:#ff5d5d}
 @media(prefers-color-scheme:light){:root{--bg:#f6f7f9;--card:#fff;--fg:#1a1d24;--mut:#5b6472;--line:#e3e6ec;--key:#eef2ff}}
 *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
   font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
 .wrap{max-width:1180px;margin:0 auto;padding:24px 20px 60px}
 h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;margin:0 0 8px;color:var(--fg)}
 .sub{color:var(--mut);margin:0 0 16px;font-size:13px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:12px 0}
 .tabs{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 2px}
 .tab{padding:6px 12px;border:1px solid var(--line);border-radius:8px;cursor:pointer;color:var(--mut);background:transparent;font-size:13px}
 .tab.on{background:var(--human);color:#fff;border-color:var(--human)}
 .tab .badge{font-size:10px;padding:1px 5px;border-radius:6px;margin-left:6px;background:var(--warn);color:#fff;vertical-align:middle}
 .ctl{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:4px 0 2px}
 .btn{width:40px;height:34px;border:1px solid var(--line);border-radius:8px;background:var(--human);color:#fff;font-size:15px;cursor:pointer}
 .btn.sec{background:transparent;color:var(--fg);width:auto;padding:0 10px}
 input[type=range]{flex:1;min-width:200px;accent-color:var(--human)}
 select{background:transparent;color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:5px 8px}
 .grid{display:grid;grid-template-columns:1.15fr 1fr;gap:12px}
 @media(max-width:860px){.grid{grid-template-columns:1fr}}
 .plotA{width:100%;height:430px}.plotB{width:100%;height:430px}.tl{width:100%;height:210px}
 .legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12.5px;margin:2px 0 8px;color:var(--mut)}
 .legend span{display:inline-flex;align-items:center;gap:6px}
 .dot{width:11px;height:11px;border-radius:3px;display:inline-block}
 .hud{display:flex;gap:10px;flex-wrap:wrap;margin-top:8px}
 .stat{background:var(--key);border:1px solid var(--line);border-radius:9px;padding:7px 12px;min-width:96px}
 .stat b{display:block;font-size:18px;font-variant-numeric:tabular-nums}
 .stat s{display:block;text-decoration:none;color:var(--mut);font-size:11px}
 .chip{padding:6px 12px;border-radius:9px;font-weight:600;font-size:13px}
 .chip.ok{background:rgba(46,204,113,.16);color:var(--achv)}
 .chip.mid{background:rgba(255,159,67,.16);color:var(--head)}
 .chip.bad{background:rgba(255,93,93,.16);color:var(--warn)}
 .chip.info{background:var(--key);color:var(--mut)}
 table{border-collapse:collapse;width:100%;font-size:13.5px}
 th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}
 th{color:var(--mut);font-weight:600}
 td:nth-child(n+2),th:nth-child(n+2){text-align:right;font-variant-numeric:tabular-nums}
 tr.key td{background:var(--key)}
 .note{color:var(--mut);font-size:12.5px;margin:8px 2px 0}
 .banner{background:var(--key);border:1px solid var(--line);border-radius:10px;padding:10px 14px;font-size:13px;color:var(--mut)}
</style></head><body><div class=wrap>
 <h1>Residual Policy 재생(playback) — 각 inference 시점의 교정</h1>
 <p class=sub id=subline></p>
 <div class=banner id=banner></div>

 <div class=card id=metricard>
  <h2 style="margin-top:0" id=mtitle>지표</h2>
  <table><thead id=mhead></thead><tbody id=mrows></tbody></table>
  <p class=note id=headnote></p>
 </div>

 <div class=card id=intcard style="display:none">
  <h2 style="margin-top:0">개입형(intervention) 분할 — 밀린 프레임 vs 안 민 프레임</h2>
  <table><thead id=inthead></thead><tbody id=introws></tbody></table>
  <p class=note id=intnote></p>
 </div>

 <div class=tabs id=tabs></div>

 <div class=card>
  <div class=ctl style="margin-bottom:8px">
    <span class=note>좌표계</span>
    <button class="tab on" id=frbase>base 기준</button>
    <button class=tab id=frworld>world 기준</button>
    <span class=note id=frnote></span>
  </div>
  <div class=ctl>
    <button class=btn id=play>▶</button>
    <button class="btn sec" id=stepb>◀ 스텝</button>
    <button class="btn sec" id=stepf>스텝 ▶</button>
    <input type=range id=slider min=0 max=0 value=0>
    <span id=framelbl class=note style="min-width:96px"></span>
    <label class=note>속도
      <select id=speed><option value=24>0.5x</option><option value=12 selected>1x</option>
      <option value=6>2x</option><option value=3>4x</option></select>
    </label>
  </div>
  <div class=hud id=hud></div>
 </div>

 <div class=grid>
  <div class=card>
   <h2>A · 전체 궤적 — residual 없을 때 vs 있을 때</h2>
   <div class=legend id=legendA></div>
   <div id=plotA class=plotA></div>
   <p class=note id=noteA></p>
  </div>
  <div class=card>
   <h2>B · 이 순간의 교정 벡터 (cm · 현재=원점)</h2>
   <div class=legend id=legendB></div>
   <div id=plotB class=plotB></div>
   <p class=note id=noteB></p>
  </div>
 </div>

 <div class=card>
  <h2>C · 타임라인 (cm)</h2>
  <div id=timeline class=tl></div>
  <p class=note>클릭/드래그로 프레임 이동. 파랑=사람이 필요로 한 교정 크기(‖GT‖)<span id=tlnote></span>.</p>
 </div>

<script>
const DATA=/*DATA*/;
const HAS_HEAD=DATA.episodes.length>0 && DATA.episodes[0].has_head;
const HAS_INT=!!DATA.summary.has_intervention;
const C={achv:'#2ecc71',slow:'#8892a0',human:'#4c8dff',head:'#ff9f43',warn:'#ff5d5d'};
const dark=matchMedia('(prefers-color-scheme:dark)').matches;
const paper=dark?'#171a21':'#fff',grid=dark?'#262b36':'#e3e6ec',fg=dark?'#e6e9ef':'#1a1d24',mut='#8892a0';
let cur=0,frame=0,timer=null;
// 좌표계: base(로깅 그대로) ↔ world(로봇 X축 회전 보정). world = RW·base.
const RW=DATA.summary.world_R;
let FRAME='base';
function mv(R,p){return [R[0][0]*p[0]+R[0][1]*p[1]+R[0][2]*p[2],
                        R[1][0]*p[0]+R[1][1]*p[1]+R[1][2]*p[2],
                        R[2][0]*p[0]+R[2][1]*p[1]+R[2][2]*p[2]];}
function xf(p){return (FRAME==='world'&&RW)?mv(RW,p):p;}
// 연속된 개입(is_int==1) 구간 [start,end] 목록.
function intSpans(e){const sp=[];if(!e.is_int)return sp;let st=-1;const A=e.is_int;
  for(let k=0;k<A.length;k++){if(A[k]&&st<0)st=k;else if(!A[k]&&st>=0){sp.push([st,k-1]);st=-1;}}
  if(st>=0)sp.push([st,A.length-1]);return sp;}

// ── 헤더 / 배너 / 지표 ──
(function(){
  const s=DATA.summary,cs=s.correction_stats,pc=x=>(x*100).toFixed(0)+'%';
  document.getElementById('subline').innerHTML=
    `${s.n_episodes}개 에피소드 · ${s.samples} 프레임 · 모드=<b>${s.mode==='logged'?'로깅 재생':'head 오버레이'}</b> · `+
    (s.heldout.length?`held-out 뱃지: ${s.heldout.join(', ')}`:'held-out 표시 없음');
  if(!HAS_HEAD){
    document.getElementById('banner').innerHTML=
      '📼 <b>로깅 재생 모드</b> — actor 가 inference 시점에 로깅한 실제 값(base 예측·사람 교정·achieved)을 그대로 재생합니다. '+
      'head 자체의 예측선은 없습니다(로봇 v4 head 는 wrench base 가 이 머신에 없어 로드 불가). '+
      '“base 가 얼마나·언제 틀렸고 사람이 어디로 교정했나”가 곧 residual 이 학습해야 할 정답입니다.';
    document.getElementById('mtitle').textContent='교정(정답) 크기 통계 — base 가 틀린 정도';
    document.getElementById('mhead').innerHTML='<tr><th>지표</th><th>값</th></tr>';
    document.getElementById('mrows').innerHTML=
      `<tr><td>필요 교정 mean</td><td>${cs.mean_cm.toFixed(2)} cm</td></tr>`+
      `<tr><td>필요 교정 median</td><td>${cs.median_cm.toFixed(2)} cm</td></tr>`+
      `<tr class=key><td>필요 교정 90퍼센타일</td><td><b>${cs.p90_cm.toFixed(2)} cm</b></td></tr>`+
      `<tr><td>필요 교정 최대</td><td>${cs.max_cm.toFixed(2)} cm</td></tr>`+
      `<tr><td>표본 수</td><td>${cs.n}</td></tr>`;
    document.getElementById('headnote').innerHTML=
      'head 예측을 겹쳐 보려면: (A) 이 데이터로 로컬 head 재학습 후 --head 로 전달, 또는 (B) 로봇의 wrench base 를 복사해 v4 로드.';
    document.getElementById('tlnote').textContent='';
  } else {
    const ov=s.head.overall,hd=s.head.hard_top25pct;
    document.getElementById('banner').innerHTML=
      `🧠 <b>head 오버레이 모드</b> — head=<b>${s.head.head_file}</b>. 각 프레임 head 예측 residual 을 base 에 적용한 선(주황)을 겹칩니다.`;
    document.getElementById('mtitle').textContent='핵심 지표 (head 예측 vs 사람 교정)';
    document.getElementById('mhead').innerHTML='<tr><th>지표</th><th>전체</th><th>하드 상위25%</th></tr>';
    document.getElementById('mrows').innerHTML=
      `<tr><td>필요 교정 크기 (mean)</td><td>${(ov.gt_trans_mean_m*100).toFixed(1)} cm</td><td>${(hd.gt_trans_mean_m*100).toFixed(1)} cm</td></tr>`+
      `<tr class=key><td>방향 cosine (mean)</td><td>${ov.direction_cosine_mean.toFixed(2)}</td><td><b>${hd.direction_cosine_mean.toFixed(2)}</b></td></tr>`+
      `<tr class=key><td>개선 프레임 %</td><td>${pc(ov.frac_capture_pos)}</td><td><b>${pc(hd.frac_capture_pos)}</b></td></tr>`+
      `<tr><td>교정 포착률 (median)</td><td>${pc(ov.capture_ratio_median)}</td><td>${pc(hd.capture_ratio_median)}</td></tr>`+
      `<tr><td>표본 수</td><td>${ov.n}</td><td>${hd.n}</td></tr>`;
    document.getElementById('headnote').innerHTML=
      '⚠ held-out 뱃지가 없는 에피소드는 head 가 학습에 사용했을 수 있어 낙관적일 수 있음. 일반화는 held-out 으로 판단.';
    document.getElementById('tlnote').innerHTML=', 주황=head 예측 교정 크기(‖pred‖)';
  }
  // 개입형(intervention) 분할 카드
  if(HAS_INT){
    const card=document.getElementById('intcard');card.style.display='';
    if(HAS_HEAD && s.head && s.head.by_intervention){
      const bi=s.head.by_intervention, iv=bi.intervention, nm=bi.nominal;
      document.getElementById('inthead').innerHTML='<tr><th>구간</th><th>프레임</th><th>방향 cos</th><th>개선%</th><th>head ‖pred‖ mean</th></tr>';
      let rows='';
      if(iv) rows+=`<tr class=key><td>🖐 개입(밀림)</td><td>${iv.n}</td><td><b>${iv.direction_cosine_mean.toFixed(2)}</b></td><td><b>${pc(iv.frac_capture_pos)}</b></td><td>—</td></tr>`;
      if(nm) rows+=`<tr><td>nominal(안 민)</td><td>${nm.n}</td><td>—</td><td>—</td><td><b>${nm.pred_trans_mean_cm.toFixed(2)} cm</b> (‖GT‖ ${nm.gt_trans_mean_cm.toFixed(2)})</td></tr>`;
      document.getElementById('introws').innerHTML=rows;
      document.getElementById('intnote').innerHTML=
        '개입형 타당성 = <b>밀린 프레임</b>에서 head 가 밀림을 따라가고(cos↑·개선%↑), '+
        '<b>안 민 프레임</b>에서 head 가 조용해야(‖pred‖→0) 함. '+(nm?`현재 nominal ‖pred‖=${nm.pred_trans_mean_cm.toFixed(2)}cm, ‖pred‖<1cm 비율 ${pc(nm.quiet_rate_lt1cm)}.`:'');
    } else if(s.intervention_gt_split){
      const g=s.intervention_gt_split;
      document.getElementById('inthead').innerHTML='<tr><th>구간</th><th>프레임</th><th>필요교정 ‖GT‖ mean</th></tr>';
      document.getElementById('introws').innerHTML=
        `<tr class=key><td>🖐 개입(밀림)</td><td>${g.intervention.n}</td><td><b>${g.intervention.gt_mean_cm.toFixed(2)} cm</b></td></tr>`+
        `<tr><td>nominal(안 민)</td><td>${g.nominal.n}</td><td>${g.nominal.gt_mean_cm.toFixed(2)} cm</td></tr>`;
      document.getElementById('intnote').innerHTML=
        '로깅 재생(head 없음): 개입 프레임의 ‖GT‖가 nominal 보다 크면 라벨이 실제 밀림과 일치. head 오버레이 모드로 보면 head 예측 분할 지표(cos·조용함)까지 나옴.';
    }
  }
  // Panel A legend / note
  const lgA=document.getElementById('legendA');
  lgA.innerHTML='<span><i class=dot style="background:var(--slow)"></i> residual 없음 (base만 목표)</span>'+
    '<span><i class=dot style="background:var(--achv)"></i> 실제 이동 (residual 있음)</span>'+
    (HAS_HEAD?'<span><i class=dot style="background:var(--head)"></i> head 예측 경로 (base⊕residual)</span>':'')+
    (HAS_INT?'<span><i class=dot style="background:var(--warn)"></i> 개입(밀림) 프레임</span>':'');
  document.getElementById('noteA').innerHTML=
    '회색=각 시점 <b>base만이라면 가려던 목표</b>(residual 없었으면), 초록=<b>실제로 움직인 경로</b>(수집 당시 residual+사람 반영). '+
    (HAS_HEAD?'주황=우리 head 가 냈을 목표 경로. ':'')+
    '벌어질수록 그 구간에서 residual/교정이 개입. <i>(각 점은 그 시점의 one-step 목표 — 독립 rollout 아님)</i>';
  // Panel B legend / note
  const lg=document.getElementById('legendB');
  lg.innerHTML='<span><i class=dot style="background:var(--slow)"></i> base 목표(slow)</span>'+
    '<span><i class=dot style="background:var(--human)"></i> 사람 교정(정답)</span>'+
    (HAS_HEAD?'<span><i class=dot style="background:var(--head)"></i> head 보정</span>':'');
  document.getElementById('noteB').innerHTML=HAS_HEAD
    ? 'head(주황)가 base(회색)에서 <b>사람(파랑)</b> 쪽으로 향할수록 그 순간의 교정이 타당(#4).'
    : 'base(회색)→사람(파랑) 벡터가 곧 그 순간 필요한 교정. 길수록 base 가 크게 틀린 순간.';
})();

// ── 탭 ──
const tabs=document.getElementById('tabs');
DATA.episodes.forEach((e,i)=>{
  const b=document.createElement('div');b.className='tab'+(i===0?' on':'');
  const cosTxt=(e.has_head&&e.metrics)?` <span class=note>(cos ${e.metrics.direction_cosine_mean.toFixed(2)})</span>`:'';
  b.innerHTML=e.name+cosTxt+(e.heldout?' <span class=badge>held-out</span>':'');
  b.onclick=()=>{[...tabs.children].forEach((c,j)=>c.className='tab'+(j===i?' on':''));selectEp(i);};
  tabs.appendChild(b);
});

const base3d=(rng)=>({paper_bgcolor:paper,plot_bgcolor:paper,font:{color:fg},showlegend:false,
  margin:{l:0,r:0,t:4,b:0},scene:{aspectmode:rng?'cube':'data',
  xaxis:{gridcolor:grid,color:fg,title:'x',range:rng&&rng[0]},
  yaxis:{gridcolor:grid,color:fg,title:'y',range:rng&&rng[1]},
  zaxis:{gridcolor:grid,color:fg,title:'z',range:rng&&rng[2]}}});
const cfg={responsive:true,displaylogo:false};

// Panel A trace 배치: 각 경로마다 [faint 전체선, 자라는 trail, 현재 마커] 3개.
//   idx 0-2 slow(residual없음), 3-5 achieved(실제), [6-8 head 예측]
const fL=(P,c)=>({type:'scatter3d',mode:'lines',x:P.map(p=>p[0]),y:P.map(p=>p[1]),z:P.map(p=>p[2]),
  line:{color:c,width:2},opacity:.25,hoverinfo:'skip'});
const eL=(c,w)=>({type:'scatter3d',mode:'lines',x:[],y:[],z:[],line:{color:c,width:w},hoverinfo:'skip'});
const eM=(c)=>({type:'scatter3d',mode:'markers',x:[],y:[],z:[],marker:{color:c,size:6},hoverinfo:'skip'});
function drawStaticA(i){
  const e=DATA.episodes[i];
  const SP=e.slow.map(xf), AP=e.achieved.map(xf);
  const traces=[fL(SP,C.slow),eL(C.slow,4),eM(C.slow),
                fL(AP,C.achv),eL(C.achv,5),eM(C.achv)];
  if(e.has_head){const HP=e.head.map(xf);traces.push(fL(HP,C.head),eL(C.head,4),eM(C.head));}
  if(e.is_int){   // 개입 프레임 위치(정적) — achieved 경로 위 빨간 마커
    const IP=AP.filter((_,k)=>e.is_int[k]);
    traces.push({type:'scatter3d',mode:'markers',x:IP.map(p=>p[0]),y:IP.map(p=>p[1]),z:IP.map(p=>p[2]),
      marker:{color:C.warn,size:2.5},opacity:.85,hoverinfo:'skip'});
  }
  Plotly.react('plotA',traces,base3d(null),cfg);
}
function drawTimeline(i){
  const e=DATA.episodes[i];
  const fr=Array.from({length:e.n},(_,k)=>k);
  const tlTraces=[{type:'scatter',mode:'lines',name:'‖필요교정‖',x:fr,y:e.gt_norm,line:{color:C.human,width:1.6}}];
  if(e.has_head) tlTraces.push({type:'scatter',mode:'lines',name:'‖head예측‖',x:fr,y:e.pred_norm,line:{color:C.head,width:1.4}});
  // shapes[0]=현재프레임 커서(update 가 relayout), 이후=개입 구간 음영.
  const shapes=[{type:'line',x0:0,x1:0,yref:'paper',y0:0,y1:1,line:{color:C.warn,width:1.5}}];
  intSpans(e).forEach(([a,b])=>shapes.push({type:'rect',xref:'x',yref:'paper',
    x0:a-0.5,x1:b+0.5,y0:0,y1:1,fillcolor:'rgba(255,93,93,0.13)',line:{width:0},layer:'below'}));
  Plotly.react('timeline',tlTraces,
    {paper_bgcolor:paper,plot_bgcolor:paper,font:{color:fg},margin:{l:44,r:12,t:6,b:34},
     legend:{orientation:'h',x:0,y:1.18},xaxis:{gridcolor:grid,color:fg,title:'frame'},
     yaxis:{gridcolor:grid,color:fg,title:'cm'},shapes},cfg);
}
function selectEp(i){
  cur=i;frame=0;const e=DATA.episodes[i];
  const sl=document.getElementById('slider');sl.max=e.n-1;sl.value=0;
  drawStaticA(i);drawTimeline(i);update(0);
}

function update(f){
  const e=DATA.episodes[cur];frame=f;
  document.getElementById('slider').value=f;
  document.getElementById('framelbl').textContent=`frame ${f} / ${e.n-1}`;
  const upd=(arr,ti,mi)=>{const t=arr.slice(0,f+1).map(xf),c=xf(arr[f]);
    Plotly.restyle('plotA',{x:[t.map(p=>p[0])],y:[t.map(p=>p[1])],z:[t.map(p=>p[2])]},[ti]);
    Plotly.restyle('plotA',{x:[[c[0]]],y:[[c[1]]],z:[[c[2]]]},[mi]);};
  upd(e.slow,1,2); upd(e.achieved,4,5);
  if(e.has_head) upd(e.head,7,8);
  const o=xf(e.achieved[f]),cm=p=>{const w=xf(p);return [(w[0]-o[0])*100,(w[1]-o[1])*100,(w[2]-o[2])*100];};
  const S=cm(e.slow[f]),H=cm(e.human[f]),D=e.has_head?cm(e.head[f]):null;
  const seg=(v,c,w)=>({type:'scatter3d',mode:'lines+markers',x:[0,v[0]],y:[0,v[1]],z:[0,v[2]],
     line:{color:c,width:w},marker:{color:c,size:[3,7]},hoverinfo:'skip'});
  const vecs=[S,H].concat(D?[D]:[]);
  const R=Math.max(1.5,...vecs.flatMap(v=>v.map(Math.abs)))*1.2;
  const traces=[{type:'scatter3d',mode:'markers',x:[0],y:[0],z:[0],marker:{color:C.achv,size:6},hoverinfo:'skip'},
    seg(S,C.slow,3),seg(H,C.human,6)];
  if(D) traces.push(seg(D,C.head,6));
  Plotly.react('plotB',traces,base3d([[-R,R],[-R,R],[-R,R]]),cfg);
  Plotly.relayout('timeline',{'shapes[0].x0':f,'shapes[0].x1':f});
  // HUD
  const gt=e.gt_norm[f];
  let hud=`<div class=stat><b>${gt.toFixed(2)}</b><s>필요교정 ‖GT‖ cm</s></div>`;
  if(e.has_head){
    const pr=e.pred_norm[f],cs=e.cosine[f],cp=e.capture[f];
    let cls='bad',msg='교정 반대/무시';
    if(cs>0.5&&cp>0.3){cls='ok';msg='사람 쪽으로 당김 ✓';}
    else if(cs>0.2){cls='mid';msg='부분적으로 당김';}
    hud+=`<div class=stat><b>${pr.toFixed(2)}</b><s>head 예측 ‖ cm</s></div>`+
      `<div class=stat><b>${cs.toFixed(2)}</b><s>방향 cosine</s></div>`+
      `<div class=stat><b>${(cp*100).toFixed(0)}%</b><s>포착률</s></div>`+
      `<div class="chip ${cls}" style="align-self:center">${msg}</div>`;
  } else {
    const lvl=gt>=5?['bad','큰 교정(≥5cm) — 위험/복구 구간']:gt>=2?['mid','중간 교정']:['info','작은 국소 보정'];
    hud+=`<div class="chip ${lvl[0]}" style="align-self:center">${lvl[1]}</div>`;
  }
  if(e.is_int){   // 개입형: 현재 프레임 개입 여부 뱃지
    hud += e.is_int[f]
      ? '<div class="chip bad" style="align-self:center">🖐 개입(밀림) 중</div>'
      : '<div class="chip info" style="align-self:center">nominal (안 민)</div>';
  }
  document.getElementById('hud').innerHTML=hud;
}

// 좌표계 토글
const frbase=document.getElementById('frbase'),frworld=document.getElementById('frworld');
frworld.textContent='world 기준 (X '+DATA.summary.world_rot_x_deg+'°)';
document.getElementById('frnote').textContent='world = Rx('+DATA.summary.world_rot_x_deg+'°)·base · 평행이동 생략(상대 기하 동일)';
function setFrame(fr){FRAME=fr;frbase.className='tab'+(fr==='base'?' on':'');
  frworld.className='tab'+(fr==='world'?' on':'');drawStaticA(cur);update(frame);}
frbase.onclick=()=>setFrame('base');
frworld.onclick=()=>setFrame('world');

const slider=document.getElementById('slider'),playbtn=document.getElementById('play');
slider.oninput=()=>{stop();update(+slider.value);};
document.getElementById('stepb').onclick=()=>{stop();update(Math.max(0,frame-1));};
document.getElementById('stepf').onclick=()=>{stop();update(Math.min(DATA.episodes[cur].n-1,frame+1));};
function tick(){const e=DATA.episodes[cur];if(frame>=e.n-1){stop();return;}update(frame+1);}
function play(){const ms=(+document.getElementById('speed').value)*1000/24|0;
  if(frame>=DATA.episodes[cur].n-1)update(0);timer=setInterval(tick,Math.max(30,ms*2));playbtn.textContent='⏸';}
function stop(){if(timer){clearInterval(timer);timer=null;}playbtn.textContent='▶';}
playbtn.onclick=()=>timer?stop():play();
document.getElementById('speed').onchange=()=>{if(timer){stop();play();}};
document.getElementById('timeline').addEventListener('plotly_click',ev=>{stop();update(Math.round(ev.points[0].x));});

selectEp(0);
</script>
</div></body></html>"""


if __name__ == "__main__":
    main()
