#!/usr/bin/env python
"""held-out 에피소드에서 residual policy 타당성을 3D로 확인하는 self-contained HTML 생성.

각 held-out 에피소드마다 세 궤적을 겹쳐 그린다:
  * slow(base) 예측 pose         (frozen diffusion base 가 가려던 곳)
  * 사람 교정 target(virtual)     (사람이 실제로 민 곳 = 학습 정답)
  * head 보정 pose (slow ⊕ pred)  (residual policy 가 낼 곳)

head 가 slow 를 사람 교정 쪽으로 끌어당기면 타당(#4). 학습된 head(weights_vN.pt)를
로드해 held-out 에 돌려 per-frame 예측을 뽑고, 지표(방향 cos / 개선% / 포착률)와 함께
plotly.min.js 를 인라인한 단일 HTML 로 굽는다.

  RESIDUAL_SLOW_CKPT=<0500 abs ckpt> RESIDUAL_CONFIG_NAME=residual_policy/hand_online_abs_mlp \
  RESIDUAL_ONLINE_WORKDIR=<scratch wd(무해)> \
  <py> online_learning/export_traj3d_html.py \
      --head <scratch>/verify_wd_0500/weights/weights_v3.pt \
      --heldout data/online_runs/run_hand_residual_abs/transitions_heldout \
      --out data/verify_0500
"""
import os
import sys
import glob
import json
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.residual_policy.pose_util import (
    apply_residual_action_to_pose9, pose_like_to_pose9)
from online_learning import config_residual_online as C
from online_learning.replay_feed_episodes import iter_source_episodes, compute_slow_pred_target_abs
from online_learning.residual_relabel_utils import write_residual_episode_hdf5, RAW_OBS_KEYS
from online_learning.verify_residual_on_data import metrics


def _episodes(source):
    files = sorted(glob.glob(os.path.join(source, "*.hdf5"))) if os.path.isdir(source) else [source]
    for fp in files:
        for tag, obs in iter_source_episodes(fp):
            yield os.path.basename(fp), tag, obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, help="학습된 head weights_vN.pt")
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--out", default="data/verify_0500")
    ap.add_argument("--num_inference_steps", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from online_learning.residual_teleop_learner import ResidualOnlineLearner
    learner = ResidualOnlineLearner()
    slow = learner.policy.slow_policy
    slow.num_inference_steps = args.num_inference_steps
    slow.n_action_steps = int(slow.horizon) - int(slow.n_obs_steps) + 1
    apr = getattr(slow, "action_pose_repr", "abs")
    dev = learner.device

    # ── 학습된 head + normalizer 로드 ──
    sd = torch.load(args.head, map_location=dev)
    learner.policy.head.load_state_dict(sd["head_state"])
    learner.policy.normalizer.load_state_dict(sd["normalizer_state"])
    learner.policy.to(dev).eval()
    print(f"[export] head 로드: {args.head}", flush=True)

    from diffusion_policy.residual_policy.step_dataset import FastResidualContextStepDataset
    ds_cfg = OmegaConf.to_container(learner.cfg.task.dataset, resolve=True); ds_cfg.pop("_target_")
    ds_cfg["val_ratio"] = 0.0

    episodes_out = []
    all_gt, all_pred = [], []
    for fname, tag, obs in _episodes(args.heldout):
        if any(k not in obs for k in RAW_OBS_KEYS) or len(obs["robot_pose_R"]) < 2:
            continue
        # 재계산 slow_pred (report 와 동일 경로)
        sp = compute_slow_pred_target_abs(slow, obs, dev, action_pose_repr=apr, batch_size=24)
        ep = {k: obs[k] for k in RAW_OBS_KEYS}; ep["slow_pred_target_abs"] = sp
        tmp = os.path.join(args.out, "_held_export.hdf5")
        write_residual_episode_hdf5(tmp, ep, "demo_0")
        ds_cfg["dataset_path"] = tmp
        ds = FastResidualContextStepDataset(**ds_cfg)

        gts, preds = [], []
        with torch.no_grad():
            for b in torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False):
                b = dict_apply(b, lambda x: x.to(dev))
                preds.append(learner.policy.predict_action(b["obs"])["action"][:, 0].cpu().numpy())
                gts.append(b["action"][:, -1].cpu().numpy())
        gt = np.concatenate(gts); pred = np.concatenate(preds)   # (n,6) residual delta6
        n = len(gt)

        # 정렬: sample i ↔ frame i (residual relabel: T -> T-1, shuffle=False)
        slow9 = pose_like_to_pose9(np.asarray(sp))[:n]                 # (n,9) base pose
        slow_pos = slow9[:, :3]
        human_pos = np.stack([apply_residual_action_to_pose9(slow9[i], gt[i])[:3] for i in range(n)])
        head_pos = np.stack([apply_residual_action_to_pose9(slow9[i], pred[i])[:3] for i in range(n)])

        gn = np.linalg.norm(gt[:, :3], axis=1)
        err = np.linalg.norm(gt[:, :3] - pred[:, :3], axis=1)
        cap = (gn - err) / (gn + 1e-9)
        cos = np.sum(gt[:, :3] * pred[:, :3], axis=1) / (gn * np.linalg.norm(pred[:, :3], axis=1) + 1e-9)

        episodes_out.append({
            "name": fname, "n": int(n),
            "slow": slow_pos.round(5).tolist(),
            "human": human_pos.round(5).tolist(),
            "head": head_pos.round(5).tolist(),
            "gt_norm": (gn * 100).round(3).tolist(),      # cm
            "pred_norm": (np.linalg.norm(pred[:, :3], axis=1) * 100).round(3).tolist(),
            "capture": np.clip(cap, -1, 1).round(3).tolist(),
            "cosine": np.clip(cos, -1, 1).round(3).tolist(),
            "metrics": metrics(gt, pred),
        })
        all_gt.append(gt); all_pred.append(pred)
        print(f"[export] {fname}: {n} frames", flush=True)

    gt = np.concatenate(all_gt); pred = np.concatenate(all_pred)
    gnall = np.linalg.norm(gt[:, :3], axis=1)
    hard = gnall >= np.quantile(gnall, 0.75)
    summary = {
        "heldout_episodes": [e["name"] for e in episodes_out],
        "heldout_samples": int(len(gt)),
        "overall": metrics(gt, pred),
        "hard_top25pct": metrics(gt, pred, hard),
    }
    data = {"summary": summary, "episodes": episodes_out}
    with open(os.path.join(args.out, "traj3d.json"), "w") as f:
        json.dump(data, f)
    print("[export] traj3d.json 저장", flush=True)

    # ── HTML (plotly 인라인) ──
    import plotly
    plotly_js = os.path.join(os.path.dirname(plotly.offline.__file__),
                             "..", "package_data", "plotly.min.js")
    with open(plotly_js) as f:
        PLOTLY = f.read()
    html = build_html(data, PLOTLY)
    out_html = os.path.join(args.out, "residual_verify.html")
    with open(out_html, "w") as f:
        f.write(html)
    print(f"[export] HTML 저장: {out_html}", flush=True)


def build_html(data, plotly_js):
    s = data["summary"]
    ov, hd = s["overall"], s["hard_top25pct"]

    def pct(x): return f"{x*100:.0f}%"
    rows = f"""
      <tr><td>필요 교정 크기 (mean)</td><td>{ov['gt_trans_mean_m']*100:.1f} cm</td><td>{hd['gt_trans_mean_m']*100:.1f} cm</td></tr>
      <tr class=key><td>방향 cosine (mean)</td><td>{ov['direction_cosine_mean']:.2f}</td><td><b>{hd['direction_cosine_mean']:.2f}</b></td></tr>
      <tr class=key><td>개선 프레임 %</td><td>{pct(ov['frac_capture_pos'])}</td><td><b>{pct(hd['frac_capture_pos'])}</b></td></tr>
      <tr><td>교정 포착률 (median)</td><td>{pct(ov['capture_ratio_median'])}</td><td>{pct(hd['capture_ratio_median'])}</td></tr>
      <tr><td>표본 수</td><td>{ov['n']}</td><td>{hd['n']}</td></tr>
    """
    data_json = json.dumps(data)
    return TEMPLATE.replace("/*PLOTLY*/", plotly_js) \
                   .replace("/*ROWS*/", rows) \
                   .replace("/*DATA*/", data_json) \
                   .replace("__HELDOUT__", ", ".join(s["heldout_episodes"])) \
                   .replace("__NSAMP__", str(s["heldout_samples"]))


TEMPLATE = r"""<!DOCTYPE html><html lang=ko><head><meta charset=utf-8>
<title>Residual Policy 타당성 (held-out)</title>
<script>/*PLOTLY*/</script>
<style>
 :root{--bg:#0f1115;--card:#171a21;--fg:#e6e9ef;--mut:#9aa4b2;--line:#262b36;--slow:#8892a0;--human:#4c8dff;--head:#ff9f43;--key:#1f2937}
 @media(prefers-color-scheme:light){:root{--bg:#f6f7f9;--card:#fff;--fg:#1a1d24;--mut:#5b6472;--line:#e3e6ec;--key:#eef2ff}}
 *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
 .wrap{max-width:1100px;margin:0 auto;padding:28px 20px 60px}
 h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:30px 0 10px;color:var(--fg)}
 .sub{color:var(--mut);margin:0 0 20px;font-size:13px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:14px 0}
 table{border-collapse:collapse;width:100%;font-size:14px}
 th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}
 th{color:var(--mut);font-weight:600}td:nth-child(2),td:nth-child(3),th:nth-child(2),th:nth-child(3){text-align:right;font-variant-numeric:tabular-nums}
 tr.key td{background:var(--key)}
 .legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;margin:6px 0 2px}
 .legend span{display:inline-flex;align-items:center;gap:6px}.dot{width:12px;height:12px;border-radius:3px;display:inline-block}
 .plot{width:100%;height:520px}.tl{width:100%;height:240px}
 .tabs{display:flex;gap:8px;margin:8px 0}.tab{padding:6px 14px;border:1px solid var(--line);border-radius:8px;cursor:pointer;color:var(--mut);background:transparent}
 .tab.on{background:var(--human);color:#fff;border-color:var(--human)}
 .note{color:var(--mut);font-size:13px}.hi{color:var(--head);font-weight:600}
</style></head><body><div class=wrap>
 <h1>Residual Policy 타당성 분석 — held-out</h1>
 <p class=sub>안 본 에피소드(__HELDOUT__), __NSAMP__ 프레임 · 학습된 residual head 를 held-out 에 적용 · slow base=260710 abs(0500)</p>

 <div class=card>
  <h2 style="margin-top:0">핵심 지표</h2>
  <table><thead><tr><th>지표</th><th>전체</th><th>하드 상위25% (중요)</th></tr></thead>
  <tbody>/*ROWS*/</tbody></table>
  <p class=note style="margin:12px 2px 0">하드 상위25% = slow base 가 가장 크게 틀린(=교정이 큰) 프레임. <span class=hi>여기서 방향 cos·개선%</span>가 residual 이 실제로 교정 쪽으로 되돌리는지를 말해준다.</p>
 </div>

 <h2>3D 궤적 — head 가 slow 를 사람 교정 쪽으로 끌어당기나?</h2>
 <div class=legend>
   <span><i class=dot style="background:var(--slow)"></i> slow base 예측</span>
   <span><i class=dot style="background:var(--human)"></i> 사람 교정 target(정답)</span>
   <span><i class=dot style="background:var(--head)"></i> head 보정 (slow⊕residual)</span>
 </div>
 <div class=tabs id=tabs></div>
 <div class=card><div id=plot3d class=plot></div></div>
 <div class=card><div id=timeline class=tl></div>
   <p class=note style="margin:8px 2px 0">파랑 ‖필요교정‖ vs 주황 ‖head 예측‖ (cm). head 가 봉우리를 따라가면 "언제 큰 교정이 필요한지" 예측하는 것.</p>
 </div>

<script>
const DATA = /*DATA*/;
const C={slow:'#8892a0',human:'#4c8dff',head:'#ff9f43'};
let cur=0;
function line(name,P,color,width){return{type:'scatter3d',mode:'lines',name,
  x:P.map(p=>p[0]),y:P.map(p=>p[1]),z:P.map(p=>p[2]),line:{color,width}};}
const dark=matchMedia('(prefers-color-scheme:dark)').matches;
const paper=dark?'#171a21':'#fff', grid=dark?'#262b36':'#e3e6ec', fg=dark?'#e6e9ef':'#1a1d24';
function draw(i){
  const e=DATA.episodes[i];
  const ax={gridcolor:grid,zerolinecolor:grid,color:fg,title:''};
  Plotly.react('plot3d',[
    line('slow',e.slow,C.slow,3),
    line('human',e.human,C.human,4),
    line('head',e.head,C.head,4),
  ],{paper_bgcolor:paper,plot_bgcolor:paper,font:{color:fg},margin:{l:0,r:0,t:6,b:0},
     legend:{orientation:'h'},scene:{xaxis:{...ax,title:'x'},yaxis:{...ax,title:'y'},zaxis:{...ax,title:'z'},aspectmode:'data'}},
     {responsive:true,displaylogo:false});
  const x=Array.from({length:e.n},(_,k)=>k);
  Plotly.react('timeline',[
    {type:'scatter',mode:'lines',name:'‖필요교정‖',x,y:e.gt_norm,line:{color:C.human}},
    {type:'scatter',mode:'lines',name:'‖head예측‖',x,y:e.pred_norm,line:{color:C.head}},
  ],{paper_bgcolor:paper,plot_bgcolor:paper,font:{color:fg},margin:{l:44,r:10,t:6,b:34},
     legend:{orientation:'h'},xaxis:{gridcolor:grid,color:fg,title:'frame'},yaxis:{gridcolor:grid,color:fg,title:'cm'}},
     {responsive:true,displaylogo:false});
}
const tabs=document.getElementById('tabs');
DATA.episodes.forEach((e,i)=>{const b=document.createElement('div');b.className='tab'+(i==0?' on':'');
  b.textContent=e.name.replace('.hdf5','')+` (cos ${e.metrics.direction_cosine_mean.toFixed(2)})`;
  b.onclick=()=>{cur=i;[...tabs.children].forEach((c,j)=>c.className='tab'+(j==i?' on':''));draw(i);};tabs.appendChild(b);});
draw(0);
</script>
</div></body></html>"""


if __name__ == "__main__":
    main()
