#!/usr/bin/env python
"""에피소드 attribution을 브라우저에서 인터랙티브하게 보는 자립형 HTML 뷰어 생성기.

프레임(추론)을 슬라이더/버튼으로 넘기며:
  - vision occlusion saliency 오버레이 (어디를 보나)
  - force 6축(Fx/Fy/Fz/Tx/Ty/Tz) 막대 (어떤 축이 중요)
  - modality 지배도(Δvision vs Δwrench) 수치 + VISION/WRENCH 배지
  - 하단 타임라인(클릭해서 프레임 점프)
를 함께 본다. 모든 데이터/이미지를 HTML 안에 embed → 서버 없이 파일만 열면 됨(외부 전송 없음).

사용 예:
  python -m analysis.modality_attribution.build_attribution_viewer \
      -i data/outputs/260710_insert_box_hand_wrench_abs/epoch=0900-train_loss=0.001.ckpt \
      --obs data/online_runs/run_hand/actor_episodes/eval_debug/episode_000013_infer_obs.hdf5 \
      -o   data/online_runs/run_hand/actor_episodes/attribution_ep013/viewer.html
"""
from __future__ import annotations

import base64
import io
import json
import pathlib

import click
import numpy as np
import torch

from analysis.modality_attribution import attribution as attr
from analysis.modality_attribution.replay_offline import load_policy
from analysis.modality_attribution.record_infer_obs import load_inference_obs
from analysis.modality_attribution.visualize_attribution import (
    occlusion_saliency, force_axis_attribution, _rgb_last_frame, _upsample,
    WRENCH_AXIS_LABELS,
)


def _overlay_b64(rgb, sal):
    """RGB + saliency 오버레이를 작은 PNG(base64)로."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    H, W = rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(rgb)
    ax.imshow(_upsample(sal, H, W), cmap="jet", alpha=0.5)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0; background:#111; color:#eee; }
  header { padding:12px 18px; background:#000; }
  header h1 { font-size:16px; margin:0; font-weight:600; }
  header .sub { font-size:12px; color:#aaa; margin-top:4px; }
  .wrap { display:flex; gap:16px; padding:16px; flex-wrap:wrap; }
  .imgcard { background:#1b1b1b; border-radius:10px; padding:10px; }
  .imgcard img { width:440px; max-width:90vw; border-radius:6px; display:block; }
  .side { flex:1; min-width:320px; display:flex; flex-direction:column; gap:14px; }
  .panel { background:#1b1b1b; border-radius:10px; padding:14px 16px; }
  .panel h2 { font-size:13px; margin:0 0 10px; color:#bbb; font-weight:600; }
  .badge { display:inline-block; padding:3px 10px; border-radius:6px; font-weight:700; font-size:13px; }
  .badge.vision { background:#1f77b4; } .badge.wrench { background:#d62728; } .badge.joint { background:#2ca02c; }
  .kv { font-size:13px; color:#ddd; margin:6px 0; }
  .kv b { color:#fff; }
  .bar-row { display:flex; align-items:center; gap:8px; margin:5px 0; font-size:12px; }
  .bar-row .lbl { width:26px; text-align:right; color:#ccc; }
  .bar-track { flex:1; background:#333; border-radius:4px; height:16px; overflow:hidden; }
  .bar-fill { height:100%; border-radius:4px; }
  .bar-row .val { width:60px; color:#aaa; font-variant-numeric:tabular-nums; }
  .controls { display:flex; align-items:center; gap:10px; padding:0 18px 8px; flex-wrap:wrap; }
  .controls button { background:#333; color:#eee; border:none; padding:7px 12px; border-radius:6px; cursor:pointer; font-size:14px; }
  .controls button:hover { background:#444; }
  .controls input[type=range] { flex:1; min-width:200px; }
  #frameLabel { font-variant-numeric:tabular-nums; font-size:13px; color:#ccc; min-width:130px; }
  svg { width:100%; height:180px; display:block; background:#1b1b1b; border-radius:10px; }
  .legend { font-size:11px; color:#aaa; padding:2px 18px 16px; }
  .legend span { margin-right:14px; }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:4px; vertical-align:middle;}
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="sub">occlusion vision saliency · force per-axis · modality dominance — 프레임을 넘기며 보세요 (외부 전송 없음, 로컬 파일)</div>
</header>

<div class="controls">
  <button id="prev">◀ 이전</button>
  <button id="play">▶ 재생</button>
  <button id="next">다음 ▶</button>
  <input type="range" id="slider" min="0" value="0">
  <span id="frameLabel"></span>
</div>

<div class="wrap">
  <div class="imgcard">
    <img id="frameimg" alt="vision saliency overlay">
    <div class="kv" style="margin-top:8px; text-align:center; color:#999;">Vision occlusion saliency (밝을수록 그 영역을 가리면 action이 크게 바뀜)</div>
  </div>
  <div class="side">
    <div class="panel">
      <h2>Modality dominance</h2>
      <div id="badge"></div>
      <div class="kv">Δ vision = <b id="dv"></b></div>
      <div class="kv">Δ wrench = <b id="dw"></b></div>
      <div class="kv">Δ joint&nbsp; = <b id="dj"></b> <span style="color:#888">(robot pose/quat + hand)</span></div>
    </div>
    <div class="panel">
      <h2>Force per-axis (축을 0으로 껐을 때 Δaction)</h2>
      <div id="forcebars"></div>
    </div>
  </div>
</div>

<svg id="timeline" viewBox="0 0 1000 180" preserveAspectRatio="none"></svg>
<div class="legend">
  <span><span class="dot" style="background:#1f77b4"></span>Δ vision</span>
  <span><span class="dot" style="background:#d62728"></span>Δ wrench</span>
  <span><span class="dot" style="background:#2ca02c"></span>Δ joint</span>
  <span>· 타임라인 클릭 = 그 프레임으로 점프</span>
</div>

<script>
const DATA = __DATA__;
const F = DATA.frames, AX = DATA.axis_labels;
const AXCOL = ["#d62728","#d62728","#d62728","#1f77b4","#1f77b4","#1f77b4"];
let cur = 0, playing = false, timer = null;

const $ = id => document.getElementById(id);
const slider = $("slider"); slider.max = F.length - 1;

const maxForce = Math.max(1e-9, ...F.flatMap(f => f.force));
const maxDelta = Math.max(1e-9, ...F.flatMap(f => [f.dv, f.dw, f.dj].filter(x => !isNaN(x))));

function fmt(x){ return (x==null||isNaN(x)) ? "–" : x.toFixed(4); }

function renderForce(f){
  let html = "";
  for(let i=0;i<AX.length;i++){
    const w = Math.max(1, 100*f.force[i]/maxForce);
    html += `<div class="bar-row"><span class="lbl">${AX[i]}</span>`
         +  `<span class="bar-track"><span class="bar-fill" style="width:${w}%;background:${AXCOL[i]}"></span></span>`
         +  `<span class="val">${fmt(f.force[i])}</span></div>`;
  }
  $("forcebars").innerHTML = html;
}

function renderTimeline(){
  const N=F.length, W=1000, H=180, pad=24;
  const x = i => pad + (W-2*pad)*(N<2?0.5:i/(N-1));
  const y = v => H-pad - (H-2*pad)*(v/maxDelta);
  let svg = "";
  // grid baseline
  svg += `<line x1="${pad}" y1="${H-pad}" x2="${W-pad}" y2="${H-pad}" stroke="#444"/>`;
  const line = (key,col) => {
    let d = F.map((f,i)=>`${i===0?'M':'L'}${x(i).toFixed(1)},${y(f[key]).toFixed(1)}`).join(' ');
    return `<path d="${d}" fill="none" stroke="${col}" stroke-width="2"/>`
         + F.map((f,i)=>`<circle cx="${x(i).toFixed(1)}" cy="${y(f[key]).toFixed(1)}" r="3" fill="${col}"/>`).join('');
  };
  svg += line('dv',"#1f77b4") + line('dw',"#d62728") + line('dj',"#2ca02c");
  svg += `<line id="marker" x1="${x(cur)}" y1="6" x2="${x(cur)}" y2="${H-6}" stroke="#fff" stroke-width="1.5" stroke-dasharray="4 3"/>`;
  // clickable overlay rects
  for(let i=0;i<N;i++){
    const cx=x(i), w=(W-2*pad)/Math.max(1,N-1);
    svg += `<rect x="${(cx-w/2).toFixed(1)}" y="0" width="${w.toFixed(1)}" height="${H}" fill="transparent" style="cursor:pointer" onclick="go(${i})"/>`;
  }
  $("timeline").innerHTML = svg;
}

function render(){
  const f = F[cur];
  $("frameimg").src = f.img;
  $("dv").textContent = fmt(f.dv);
  $("dw").textContent = fmt(f.dw);
  $("dj").textContent = fmt(f.dj);
  const cand = [["vision",f.dv],["wrench",f.dw],["joint",f.dj]].filter(c => !isNaN(c[1]));
  cand.sort((a,b) => b[1]-a[1]);
  const dom = cand.length ? cand[0][0] : "vision";
  $("badge").innerHTML = `<span class="badge ${dom}">${dom.toUpperCase()} dominant</span>`;
  $("frameLabel").textContent = `inf ${f.idx} · t=${f.t.toFixed(2)}s · ${cur+1}/${F.length}`;
  slider.value = cur;
  renderForce(f);
  renderTimeline();
}
function go(i){ cur = Math.max(0, Math.min(F.length-1, i)); render(); }
$("prev").onclick = ()=>go(cur-1);
$("next").onclick = ()=>go(cur+1);
slider.oninput = e => go(parseInt(e.target.value));
$("play").onclick = function(){
  playing = !playing; this.textContent = playing ? "⏸ 정지" : "▶ 재생";
  if(playing){ timer = setInterval(()=>{ go(cur>=F.length-1?0:cur+1); }, 700); }
  else clearInterval(timer);
};
window.go = go;
render();
</script>
</body>
</html>
"""


@click.command()
@click.option("--input", "-i", required=True, help="Path to checkpoint")
@click.option("--obs", required=True, help="episode_XXXXXX_infer_obs.hdf5")
@click.option("--output", "-o", required=True, help="출력 HTML 경로")
@click.option("--num_inference_steps", "-n", default=16, type=int, show_default=True)
@click.option("--seeds", default="0,1", help="Comma-separated seeds.")
@click.option("--grid", default=8, type=int, show_default=True)
@click.option("--occ_chunk", default=16, type=int, show_default=True)
@click.option("--frames", default="all", help="처리할 프레임 수(정수) 또는 'all'.")
@click.option("--vision_baseline", type=click.Choice(["mean", "self", "start"]), default="mean",
              show_default=True,
              help="modality dominance용 vision baseline. mean=중립이미지(공정, zero-wrench와 대칭), "
                   "start=시작프레임고정(로봇이동에 과대), self=직전프레임고정.")
@click.option("--device", default="cuda", show_default=True)
def main(input, obs, output, num_inference_steps, seeds, grid, occ_chunk, frames, vision_baseline, device):
    seeds = [int(s) for s in str(seeds).split(",") if s.strip() != ""]
    out = pathlib.Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading policy from {input}")
    policy, cfg = load_policy(input, num_inference_steps=num_inference_steps, device=device)
    rgb_key = policy.rgb_keys[0] if policy.rgb_keys else None
    wrench_key = policy.wrench_keys[0] if policy.wrench_keys else None

    data = load_inference_obs(obs)
    obs_by_inference = data["obs_by_inference"]
    elapsed_s = np.asarray(data["elapsed_s"], dtype=np.float64)
    inference_index = np.asarray(data["inference_index"])
    N = len(obs_by_inference)
    if str(frames).lower() == "all":
        sel = list(range(N))
    else:
        k = min(int(frames), N)
        sel = sorted(set(np.linspace(0, N - 1, k).astype(int).tolist()))
    print(f"  {N} inferences, viewer 프레임 {len(sel)}개 (grid={grid}, seeds={seeds})")

    device_t = policy.device
    start_obs = attr.obs_np_to_tensor(obs_by_inference[0], device_t)

    # joint(proprioception=low_dim) baseline용 에피소드 평균 low_dim (공정 = 평균값으로 교체)
    ld_ref = {}
    for k in policy.low_dim_keys:
        arrs = [np.asarray(o[k]) for o in obs_by_inference if k in o]
        if arrs:
            m = np.concatenate([a.reshape(-1, a.shape[-1]) for a in arrs], 0).mean(0)
            ld_ref[k] = torch.from_numpy(m.astype(np.float32)).view(1, 1, -1).to(device_t)

    frames_data = []
    for j, i in enumerate(sel):
        obs_dict = attr.obs_np_to_tensor(obs_by_inference[i], device_t)
        baselines = {}
        if rgb_key is not None:
            if vision_baseline == "mean":
                baselines["vision"] = attr.make_blank_vision(policy)          # 공정(기본)
            elif vision_baseline == "self":
                baselines["vision"] = attr.make_freeze_vision(policy, obs_dict)
            else:  # start
                baselines["vision"] = attr.make_freeze_vision(policy, start_obs)
        if wrench_key is not None:
            baselines["wrench"] = attr.make_zero_wrench(policy)
        if policy.low_dim_keys and ld_ref:
            baselines["joint"] = attr.make_replace_low_dim(policy, ld_ref)
        res = attr.ablation_deltas(policy, obs_dict, baselines, seeds=seeds)
        dv = res.deltas["vision"].total if "vision" in res.deltas else float("nan")
        dw = res.deltas["wrench"].total if "wrench" in res.deltas else float("nan")
        dj = res.deltas["joint"].total if "joint" in res.deltas else float("nan")

        force = ([float(x) for x in force_axis_attribution(policy, obs_dict, wrench_key, seeds=seeds)]
                 if wrench_key is not None else [0.0] * len(WRENCH_AXIS_LABELS))

        sal = (occlusion_saliency(policy, obs_dict, rgb_key, grid=grid, seeds=seeds, chunk=occ_chunk)
               if rgb_key is not None else np.zeros((1, 1)))
        rgb = _rgb_last_frame(obs_by_inference[i], rgb_key) if rgb_key is not None else np.zeros((8, 8, 3))
        img_b64 = _overlay_b64(rgb, sal)

        t = float(elapsed_s[i]) if np.isfinite(elapsed_s[i]) else float(i)
        frames_data.append({"idx": int(inference_index[i]), "t": t,
                            "dv": dv, "dw": dw, "dj": dj, "force": force, "img": img_b64})
        print(f"  [{j+1}/{len(sel)}] inf {int(inference_index[i])}: dom_v={dv:.4f} dom_w={dw:.4f} "
              f"dom_j={dj:.4f}  topforce={WRENCH_AXIS_LABELS[int(np.argmax(force))] if wrench_key else '-'}")

    payload = {"axis_labels": WRENCH_AXIS_LABELS, "frames": frames_data}
    title = f"Attribution viewer — {pathlib.Path(obs).name}"
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(payload)).replace("__TITLE__", title)
    out.write_text(html, encoding="utf-8")
    size_mb = out.stat().st_size / 1e6
    print(f"\nViewer saved: {out}  ({size_mb:.1f} MB, {len(frames_data)} frames)")
    print(f"열기: 브라우저에서 file://{out.resolve()} 또는 `xdg-open {out}`")


if __name__ == "__main__":
    main()
