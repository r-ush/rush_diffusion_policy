#!/usr/bin/env python
"""Residual policy attribution + 3D 궤적 + saliency HTML (모든 에피소드).

한 HTML 에서 에피소드 탭으로 전환하며 프레임 재생. 프레임마다:
  (A) 3D 궤적 — base 예측 경로(회색) vs base+residual 경로(주황) vs achieved(초록).
      현재 스텝에서 **residual 이 base 위에 더하는 delta 벡터**(회색→주황)를 강조 +
      국소 확대(cm, base=원점)로 그 추가분을 또렷하게.
  (B) modality ablation Δresidual 타임라인 — vision/force/base_action/low_dim 중 무엇이
      residual 을 좌우하나(+개입 구간 음영).
  (C) Grad-CAM saliency — 원본 | residual saliency | base saliency (공유 ResNet34).
      느린 base CAM 때문에 에피소드당 몇 프레임(--img_per_ep, ‖e‖ 큰 순)만 임베드.

residual delta = α⁻¹·e (lag_model.to_command) 를 cap(5cm/0.4rad) 후 base 목표에 SE3 합성 —
actor 와 동일. lag(α)는 head payload 에 실려 온다.

실행(bae_robodiff):
  RESIDUAL_INTERVENTION_SLOW_CKPT=<260714 rel base> \
  RESIDUAL_INTERVENTION_CONFIG_NAME=residual_policy/hand_intervention_mlp \
  RESIDUAL_INTERVENTION_WORKDIR=<scratch wd> \
  <bae_py> analysis/modality_attribution/residual_attribution_html.py \
     --head data/online_runs/run_hand_intervention/weights/weights_v3.pt \
     --episodes data/online_runs/run_hand_intervention/transitions \
     --stride 6 --img_per_ep 6 --out data/verify_attr
"""
import os
import sys
import io
import json
import base64
import glob
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import warnings
warnings.filterwarnings("ignore")
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.residual_policy.create_slow_pred_fast_dataset import (
    build_policy_obs, collate_policy_obs)
from diffusion_policy.residual_policy.pose_util import (
    pose_like_to_pose9, apply_residual_action_to_pose9)
from analysis.modality_attribution import attribution as A
from online_learning import lag_model


# ─────────────────────────── 로드 / 유틸 ───────────────────────────
def load_residual(head_path):
    from online_learning.residual_intervention_learner import ResidualInterventionLearner
    learner = ResidualInterventionLearner()
    policy, dev = learner.policy, learner.device
    sd = torch.load(head_path, map_location=dev)
    policy.head.load_state_dict(sd["head_state"])
    policy.normalizer.load_state_dict(sd["normalizer_state"])
    if sd.get("force_encoder_state") and policy.force_encoder is not None:
        policy.force_encoder.load_state_dict(sd["force_encoder_state"])
    policy.to(dev).eval()
    lag = lag_model.from_payload(sd.get("lag"))
    return learner, policy, dev, sd, lag


def cap6(r, tcap=0.05, rcap=0.4):
    r = np.asarray(r, dtype=np.float64).copy()
    tn = np.linalg.norm(r[:3])
    if tn > tcap:
        r[:3] *= tcap / tn
    rn = np.linalg.norm(r[3:6])
    if rn > rcap:
        r[3:6] *= rcap / rn
    return r


def _cam_from(feat, grad):
    w = grad.mean(dim=(1, 2))
    cam = torch.relu((w[:, None, None] * feat).sum(0))
    m = float(cam.max())
    return (cam / m).cpu().numpy() if m > 1e-9 else np.zeros(tuple(cam.shape), np.float32)


def residual_pred(policy, obs_dev):
    with torch.no_grad():
        return policy.predict_action(obs_dev)["action"][:, 0]  # (B,6)


def ablation_mm(policy, obs_dev, r0):
    def resid(o):
        with torch.no_grad():
            return policy.predict_action(o)["action"][:, 0]
    def dmm(a, b):
        return float((a[:, :3] - b[:, :3]).norm(dim=-1).mean()) * 1000.0
    key = policy.base_action_key
    bmean = obs_dev[key].mean(dim=tuple(range(obs_dev[key].ndim - 1)), keepdim=True)
    def repl_base(o):
        out = A.clone_obs(o); out[key] = bmean.expand_as(out[key]).clone(); return out
    bl = {"vision": A.make_blank_vision(policy), "force": A.make_zero_wrench(policy),
          "low_dim": A.make_replace_low_dim(policy, obs_dev), "base_action": repl_base}
    return {nm: round(dmm(r0, resid(fn(obs_dev))), 3) for nm, fn in bl.items()}


def residual_cam(policy, obs_dev):
    cap = {}
    def hook(m, i, o):
        if o.requires_grad:
            o.retain_grad(); cap["a"] = o
    h = policy.vision_encoder.register_forward_hook(hook)
    orig = policy._encode_initial_image
    def enc_grad(nobs):
        return torch.cat([policy._pool_image_feature(policy.vision_encoder(
            nobs[k][:, 0] if nobs[k].ndim == 5 else nobs[k])) for k in policy.rgb_keys], dim=-1)
    policy._encode_initial_image = enc_grad
    obs_g = {k: (v.clone().requires_grad_(True) if k in policy.rgb_keys else v)
             for k, v in obs_dev.items()}
    policy.zero_grad(set_to_none=True)
    with torch.enable_grad():
        policy.predict_action(obs_g)["action"][:, 0][:, :3].pow(2).sum().backward()
    a = cap.get("a")
    cam = _cam_from(a[0].detach(), a.grad[0].detach()) if (a is not None and a.grad is not None) else np.zeros((7, 7), np.float32)
    h.remove(); policy._encode_initial_image = orig
    return cam


def base_cam(slow, sobs):
    cap = {}
    def hook(m, i, o):
        if o.requires_grad:
            o.retain_grad(); cap["a"] = o
    h = slow.vision_encoder.register_forward_hook(hook)
    sobs_g = {k: (v.clone().requires_grad_(True) if k in getattr(slow, "rgb_keys", []) else v)
              for k, v in sobs.items()}
    slow.zero_grad(set_to_none=True)
    try:
        with torch.enable_grad():
            slow.predict_action(sobs_g)["action"][..., :3].pow(2).sum().backward()
        a = cap.get("a")
        return _cam_from(a[-1].detach(), a.grad[-1].detach()) if (a is not None and a.grad is not None) else np.zeros((7, 7), np.float32)
    finally:
        h.remove()


def composite_b64(img, cam_r, cam_b, title):
    fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.5))
    for a in ax:
        a.axis("off")
    ext = [0, img.shape[1], img.shape[0], 0]
    ax[0].imshow(img); ax[0].set_title("원본", fontsize=8)
    ax[1].imshow(img); ax[1].imshow(cam_r, cmap="jet", alpha=0.45, interpolation="bilinear", extent=ext); ax[1].set_title("residual", fontsize=8)
    ax[2].imshow(img); ax[2].imshow(cam_b, cmap="jet", alpha=0.45, interpolation="bilinear", extent=ext); ax[2].set_title("base", fontsize=8)
    fig.suptitle(title, fontsize=8); fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=80, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def process_episode(fp, policy, slow, dev, lag, ds_cfg, stride, img_per_ep, do_base_cam, gain):
    name = os.path.basename(fp).replace(".hdf5", "")
    with h5py.File(fp, "r") as f:
        o = f["data"]["demo_0"]["obs"]
        need = ["slow_pred_target_abs", "virtual_target_abs", "robot_pose_R"]
        if any(k not in o for k in need):
            return None
        obs_np = {k: np.asarray(o[k]) for k in o.keys()}
        img_all = np.asarray(o["image0"])
        slow9_all = pose_like_to_pose9(np.asarray(o["slow_pred_target_abs"]))
        human_all = np.asarray(o["virtual_target_abs"])[:, :3]
        achieved_all = np.asarray(o["robot_pose_R"])[:, :3]
        isint_all = (np.asarray(o["is_intervention"]).reshape(-1) > 0.5
                     if "is_intervention" in o else np.zeros(len(img_all), bool))

    from diffusion_policy.residual_policy.step_dataset import FastResidualContextStepDataset
    cfg = dict(ds_cfg); cfg["dataset_path"] = fp
    ds = FastResidualContextStepDataset(**cfg)
    n = len(ds)
    idxs = list(range(0, n, max(1, stride)))

    fr = []
    for t in idxs:
        obs_dev = dict_apply(ds[t]["obs"], lambda x: x.to(dev)[None])
        e6 = residual_pred(policy, obs_dev)[0].cpu().numpy()
        abl = ablation_mm(policy, obs_dev, residual_pred(policy, obs_dev))
        delta = lag_model.to_command(e6, lag, gain_scale=gain) if lag is not None else e6.copy()
        delta = cap6(delta)
        base_p = slow9_all[t][:3]
        resid_p = apply_residual_action_to_pose9(slow9_all[t], delta)[:3]
        fr.append({"t": int(t),
                   "e_mm": round(float(np.linalg.norm(e6[:3])) * 1000, 3),
                   "d_mm": round(float(np.linalg.norm(resid_p - base_p)) * 1000, 3),
                   "is_int": bool(isint_all[t]) if t < len(isint_all) else False,
                   "abl": abl,
                   "base": base_p.round(5).tolist(),
                   "resid": resid_p.round(5).tolist(),
                   "achieved": achieved_all[t].round(5).tolist(),
                   "human": human_all[t].round(5).tolist(),
                   "img": None})

    # saliency 이미지: ‖e‖ 큰 순 img_per_ep 프레임에만 (느린 base CAM 최소화)
    if img_per_ep > 0 and fr:
        order = sorted(range(len(fr)), key=lambda i: -fr[i]["e_mm"])[:img_per_ep]
        for i in sorted(order):
            t = fr[i]["t"]
            obs_dev = dict_apply(ds[t]["obs"], lambda x: x.to(dev)[None])
            cam_r = residual_cam(policy, obs_dev)
            cam_b = base_cam(slow, collate_policy_obs([build_policy_obs(slow, obs_np, t, dev)])) if do_base_cam else np.zeros((7, 7), np.float32)
            ttl = f"{name} f{t}  ‖e‖={fr[i]['e_mm']:.1f}mm  {'개입' if fr[i]['is_int'] else 'nominal'}"
            fr[i]["img"] = composite_b64(img_all[t], cam_r, cam_b, ttl)
    return {"name": name, "n": len(fr), "n_int": int(sum(f["is_int"] for f in fr)), "frames": fr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--episodes", required=True, help="transitions 디렉토리 또는 단일 *.hdf5")
    ap.add_argument("--out", default="data/verify_attr")
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--img_per_ep", type=int, default=6, help="에피소드당 saliency 이미지 수(‖e‖ 큰 순). 0=없음")
    ap.add_argument("--no_base_cam", action="store_true", help="느린 base diffusion Grad-CAM 생략")
    ap.add_argument("--num_inference_steps", type=int, default=8)
    ap.add_argument("--gain", type=float, default=1.0, help="residual 게인(δ=gain·α⁻¹·e). actor 와 맞추려면 1.0")
    ap.add_argument("--max_episodes", type=int, default=0)
    ap.add_argument("--world_rot_x_deg", type=float, default=135.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    learner, policy, dev, sd, lag = load_residual(args.head)
    slow = policy.slow_policy
    slow.num_inference_steps = args.num_inference_steps
    slow.n_action_steps = int(slow.horizon) - int(slow.n_obs_steps) + 1
    print(f"[attr] head v{sd.get('version','?')} demos={sd.get('num_demos','?')} lag={'있음' if lag is not None else '없음'}", flush=True)

    ds_cfg = OmegaConf.to_container(learner.cfg.task.dataset, resolve=True); ds_cfg.pop("_target_")
    ds_cfg["val_ratio"] = 0.0

    files = sorted(glob.glob(os.path.join(args.episodes, "*.hdf5"))) if os.path.isdir(args.episodes) else [args.episodes]
    if args.max_episodes:
        files = files[:args.max_episodes]
    episodes = []
    for k, fp in enumerate(files):
        ep = process_episode(fp, policy, slow, dev, lag, ds_cfg, args.stride,
                             args.img_per_ep, not args.no_base_cam, args.gain)
        if ep is None:
            print(f"[attr] skip {os.path.basename(fp)}", flush=True); continue
        episodes.append(ep)
        m = np.mean([[f["abl"][x] for x in ("vision", "force", "base_action", "low_dim")] for f in ep["frames"]], axis=0)
        print(f"[attr] {k+1}/{len(files)} {ep['name']}: {ep['n']}f 개입{ep['n_int']}  "
              f"Δmean v/f/b/l={m[0]:.2f}/{m[1]:.2f}/{m[2]:.2f}/{m[3]:.2f}", flush=True)

    th = np.radians(args.world_rot_x_deg); c, s = np.cos(th), np.sin(th)
    data = {"episodes": episodes, "head_file": os.path.basename(args.head),
            "world_R": [[1, 0, 0], [0, c, -s], [0, s, c]], "world_deg": args.world_rot_x_deg}
    import plotly
    pj = os.path.join(os.path.dirname(plotly.offline.__file__), "..", "package_data", "plotly.min.js")
    with open(pj) as f:
        PLOTLY = f.read()
    out_html = os.path.join(args.out, "residual_attribution.html")
    with open(out_html, "w") as f:
        f.write(TEMPLATE.replace("/*PLOTLY*/", PLOTLY).replace("/*DATA*/", json.dumps(data)))
    sz = os.path.getsize(out_html) / 1024 / 1024
    print(f"[attr] 저장: {out_html}  ({len(episodes)} 에피소드, {sz:.1f}MB)", flush=True)


TEMPLATE = r"""<!DOCTYPE html><html lang=ko><head><meta charset=utf-8>
<title>Residual Attribution · 3D · Saliency</title>
<script>/*PLOTLY*/</script>
<style>
 :root{--bg:#0f1115;--card:#171a21;--fg:#e6e9ef;--mut:#9aa4b2;--line:#262b36;
   --base:#8892a0;--resid:#ff9f43;--achv:#2ecc71;--human:#4c8dff;--force:#ff5d5d;--vision:#4c8dff;--low:#2ecc71;--int:#ff5d5d}
 @media(prefers-color-scheme:light){:root{--bg:#f6f7f9;--card:#fff;--fg:#1a1d24;--mut:#5b6472;--line:#e3e6ec}}
 *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
 .wrap{max-width:1120px;margin:0 auto;padding:20px 16px 60px}
 h1{font-size:19px;margin:0 0 4px}.sub{color:var(--mut);font-size:12.5px;margin:0 0 12px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin:11px 0}
 .tabs{display:flex;gap:6px;flex-wrap:wrap;margin:4px 0}
 .tab{padding:5px 10px;border:1px solid var(--line);border-radius:8px;cursor:pointer;color:var(--mut);background:transparent;font-size:12px}
 .tab.on{background:var(--human);color:#fff;border-color:var(--human)}.tab .b{font-size:9px;padding:1px 4px;border-radius:5px;background:var(--int);color:#fff;margin-left:4px}
 .grid{display:grid;grid-template-columns:1.25fr 1fr;gap:10px}@media(max-width:880px){.grid{grid-template-columns:1fr}}
 .p3d{width:100%;height:400px}.ploc{width:100%;height:400px}.ptl{width:100%;height:210px}
 .ctl{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:8px 0 2px}
 .btn{width:38px;height:30px;border:1px solid var(--line);border-radius:8px;background:var(--resid);color:#fff;cursor:pointer}
 input[type=range]{flex:1;min-width:200px;accent-color:var(--resid)}
 select{background:transparent;color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:4px 8px}
 .lab{color:var(--mut);font-size:12px;min-width:160px}
 .legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--mut);margin:4px 0}
 .legend span{display:inline-flex;align-items:center;gap:5px}.dot{width:10px;height:10px;border-radius:3px}
 img.sal{width:100%;border-radius:8px;display:block;background:#000}
 .note{color:var(--mut);font-size:12px;margin:7px 2px 0}.hi{color:var(--resid);font-weight:600}
 table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line)}
 th{color:var(--mut)}td:nth-child(n+2),th:nth-child(n+2){text-align:right;font-variant-numeric:tabular-nums}tr.key td{background:rgba(255,93,93,.10)}
</style></head><body><div class=wrap>
<h1>Residual Attribution · 3D 궤적 · Saliency</h1>
<p class=sub id=sub></p>
<div class=tabs id=tabs></div>

<div class=card>
 <div class=ctl>
   <button class=btn id=play>▶</button>
   <input type=range id=sl min=0 max=0 value=0>
   <span class=lab id=flab></span>
   <label class=lab>좌표<select id=fr><option value=world>world</option><option value=base>base</option></select></label>
 </div>
</div>

<div class=grid>
 <div class=card>
   <h3 style="margin:0 0 4px;font-size:14px">A · 전체 궤적</h3>
   <div class=legend>
     <span><i class=dot style="background:var(--base)"></i>base 예측</span>
     <span><i class=dot style="background:var(--resid)"></i>base+residual</span>
     <span><i class=dot style="background:var(--achv)"></i>achieved</span>
   </div>
   <div id=p3d class=p3d></div>
 </div>
 <div class=card>
   <h3 style="margin:0 0 4px;font-size:14px">B · 이 스텝 residual 추가분 (cm, base=원점)</h3>
   <div class=legend>
     <span><i class=dot style="background:var(--resid)"></i>residual delta (base→base+resid)</span>
     <span><i class=dot style="background:var(--human)"></i>사람 교정(참고)</span>
   </div>
   <div id=ploc class=ploc></div>
 </div>
</div>

<div class=card>
 <img id=sal class=sal>
 <p class=note>원본 | <span class=hi>residual</span> saliency | base saliency (Grad-CAM). ‖e‖ 큰 프레임만 계산 — 슬라이더가 가장 가까운 계산 프레임을 보여줍니다.</p>
</div>

<div class=card>
 <div class=legend>
   <span><i class=dot style="background:var(--vision)"></i>vision</span><span><i class=dot style="background:var(--force)"></i>force</span>
   <span><i class=dot style="background:var(--resid)"></i>base_action</span><span><i class=dot style="background:var(--low)"></i>low_dim</span>
   <span><i class=dot style="background:var(--mut)"></i>‖e‖</span><span><i class=dot style="background:rgba(255,93,93,.35)"></i>개입</span>
 </div>
 <div id=tl class=ptl></div>
 <p class=note>프레임별 Δresidual(입력 지우면 교정이 얼마나 바뀌나) + ‖e‖. 클릭으로 이동.</p>
</div>

<script>
const DATA=/*DATA*/, EPS=DATA.episodes, RW=DATA.world_R;
const dark=matchMedia('(prefers-color-scheme:dark)').matches;
const paper=dark?'#171a21':'#fff',grid=dark?'#262b36':'#e3e6ec',fg=dark?'#e6e9ef':'#1a1d24';
const C={base:'#8892a0',resid:'#ff9f43',achv:'#2ecc71',human:'#4c8dff',force:'#ff5d5d',vision:'#4c8dff',low:'#2ecc71',mut:'#9aa4b2'};
let ep=0,cur=0,timer=null,FRAME='world';
document.getElementById('sub').innerHTML=`${EPS.length}개 에피소드 · head=<b>${DATA.head_file}</b> · Grad-CAM 공유 ResNet34(base vs residual)`;
function mv(p){return FRAME==='world'?[RW[0][0]*p[0]+RW[0][1]*p[1]+RW[0][2]*p[2],RW[1][0]*p[0]+RW[1][1]*p[1]+RW[1][2]*p[2],RW[2][0]*p[0]+RW[2][1]*p[1]+RW[2][2]*p[2]]:p;}

const tabs=document.getElementById('tabs');
EPS.forEach((e,i)=>{const b=document.createElement('div');b.className='tab'+(i?'':' on');
  b.innerHTML=e.name+(e.n_int?` <span class=b>개입${e.n_int}</span>`:'');
  b.onclick=()=>{[...tabs.children].forEach((c,j)=>c.className='tab'+(j==i?' on':''));selEp(i);};tabs.appendChild(b);});

const b3=(rng)=>({paper_bgcolor:paper,plot_bgcolor:paper,font:{color:fg},showlegend:false,margin:{l:0,r:0,t:4,b:0},
  scene:{aspectmode:rng?'cube':'data',uirevision:'k',
   xaxis:{gridcolor:grid,color:fg,title:'x',range:rng&&rng[0]},yaxis:{gridcolor:grid,color:fg,title:'y',range:rng&&rng[1]},zaxis:{gridcolor:grid,color:fg,title:'z',range:rng&&rng[2]}}});
const cfg={responsive:true,displaylogo:false};
const L=(P,c,w)=>({type:'scatter3d',mode:'lines',x:P.map(p=>p[0]),y:P.map(p=>p[1]),z:P.map(p=>p[2]),line:{color:c,width:w},opacity:.5,hoverinfo:'skip'});
const E=(c,w)=>({type:'scatter3d',mode:'lines',x:[],y:[],z:[],line:{color:c,width:w},hoverinfo:'skip'});
const M=(c,s)=>({type:'scatter3d',mode:'markers',x:[],y:[],z:[],marker:{color:c,size:s},hoverinfo:'skip'});

function selEp(i){ep=i;cur=0;const e=EPS[i];document.getElementById('sl').max=e.n-1;document.getElementById('sl').value=0;
  const B=e.frames.map(f=>mv(f.base)),R=e.frames.map(f=>mv(f.resid)),Ac=e.frames.map(f=>mv(f.achieved));
  Plotly.react('p3d',[L(B,C.base,3),L(R,C.resid,3),L(Ac,C.achv,2),
    E(C.base,5),E(C.resid,5),M(C.base,4),M(C.resid,5),M(C.achv,5),
    {type:'scatter3d',mode:'lines',x:[],y:[],z:[],line:{color:C.resid,width:7},hoverinfo:'skip'}],b3(null),cfg);
  // 타임라인
  const x=e.frames.map((f,k)=>k),shp=[{type:'line',x0:0,x1:0,yref:'paper',y0:0,y1:1,line:{color:'#fff',width:1.5}}];
  let a=0,run=e.frames.map(f=>f.is_int);
  for(let k=0;k<run.length;k++){if(run[k]&&(k==0||!run[k-1])){a=k;} if(run[k]&&(k==run.length-1||!run[k+1]))shp.push({type:'rect',x0:a-0.5,x1:k+0.5,yref:'paper',y0:0,y1:1,fillcolor:'rgba(255,93,93,.14)',line:{width:0},layer:'below'});}
  const tr=(key,col,w)=>({type:'scatter',mode:'lines',x,y:e.frames.map(f=>key=='e'?f.e_mm:f.abl[key]),line:{color:col,width:w},name:key});
  Plotly.react('tl',[tr('e',C.mut,1.2),tr('low',C.low,1.2),tr('base_action',C.resid,1.2),tr('force',C.force,2),tr('vision',C.vision,2)],
   {paper_bgcolor:paper,plot_bgcolor:paper,font:{color:fg},margin:{l:40,r:8,t:6,b:26},showlegend:false,
    xaxis:{gridcolor:grid,color:fg,title:'frame(subsample)'},yaxis:{gridcolor:grid,color:fg,title:'Δ mm'},shapes:shp},cfg);
  upd(0);
}
function upd(k){cur=k;const e=EPS[ep],f=e.frames[k];document.getElementById('sl').value=k;
  document.getElementById('flab').textContent=`f${f.t} · ‖e‖ ${f.e_mm}mm · Δcmd ${f.d_mm}mm · ${f.is_int?'개입':'nominal'}`;
  const B=e.frames.slice(0,k+1).map(x=>mv(x.base)),R=e.frames.slice(0,k+1).map(x=>mv(x.resid)),Ac=e.frames.slice(0,k+1).map(x=>mv(x.achieved));
  const st=(P,i)=>Plotly.restyle('p3d',{x:[P.map(p=>p[0])],y:[P.map(p=>p[1])],z:[P.map(p=>p[2])]},[i]);
  st(B,3);st(R,4);
  const pt=(p,i)=>Plotly.restyle('p3d',{x:[[p[0]]],y:[[p[1]]],z:[[p[2]]]},[i]);
  const b=mv(f.base),r=mv(f.resid),a=mv(f.achieved);pt(b,5);pt(r,6);pt(a,7);
  Plotly.restyle('p3d',{x:[[b[0],r[0]]],y:[[b[1],r[1]]],z:[[b[2],r[2]]]},[8]);  // residual delta arrow
  // B: 국소 (base=원점, cm)
  const o=b,cm=p=>[(p[0]-o[0])*100,(p[1]-o[1])*100,(p[2]-o[2])*100];
  const rd=cm(mv(f.resid)),hd=cm(mv(f.human)),Rr=Math.max(1,Math.abs(rd[0]),Math.abs(rd[1]),Math.abs(rd[2]),Math.abs(hd[0]),Math.abs(hd[1]),Math.abs(hd[2]))*1.25;
  const seg=(v,c,w)=>({type:'scatter3d',mode:'lines+markers',x:[0,v[0]],y:[0,v[1]],z:[0,v[2]],line:{color:c,width:w},marker:{color:c,size:[3,7]},hoverinfo:'skip'});
  Plotly.react('ploc',[{type:'scatter3d',mode:'markers',x:[0],y:[0],z:[0],marker:{color:C.base,size:6},hoverinfo:'skip'},seg(hd,C.human,4),seg(rd,C.resid,7)],b3([[-Rr,Rr],[-Rr,Rr],[-Rr,Rr]]),cfg);
  Plotly.relayout('tl',{'shapes[0].x0':k,'shapes[0].x1':k});
  // saliency: 가장 가까운 계산 프레임
  let best=-1,bd=1e9;e.frames.forEach((x,j)=>{if(x.img){const d=Math.abs(j-k);if(d<bd){bd=d;best=j;}}});
  const sal=document.getElementById('sal');if(best>=0){sal.style.display='';sal.src=e.frames[best].img;}else sal.style.display='none';
}
document.getElementById('sl').oninput=()=>{stop();upd(+document.getElementById('sl').value);};
document.getElementById('fr').onchange=e=>{FRAME=e.target.value;selEp(ep);upd(cur);};
document.getElementById('tl').addEventListener('plotly_click',ev=>{stop();upd(Math.round(ev.points[0].x));});
const pb=document.getElementById('play');
function stepf(){const e=EPS[ep];if(cur>=e.n-1){stop();return;}upd(cur+1);}
function play(){if(cur>=EPS[ep].n-1)upd(0);timer=setInterval(stepf,300);pb.textContent='⏸';}
function stop(){if(timer){clearInterval(timer);timer=null;}pb.textContent='▶';}
pb.onclick=()=>timer?stop():play();
selEp(0);
</script>
</div></body></html>"""


if __name__ == "__main__":
    main()
