#!/usr/bin/env python
"""wrench 영향이 왜 작은지 '메커니즘' 분해 (modality-attention fusion 정책 전용).

이 정책 fusion:
  tokens = [vis_t0, vis_t1, wrench]  (각 vision_feature_dim=512, 동일 크기)
    → +learnable pos-emb → TransformerEncoderLayer(self-attn) → out(3,512)
    → linear_projection(1536→512) → +raw low_dim(28) = global_cond(540) → U-Net.

force는 '3개 토큰 중 하나'라 concat 크기 문제(작아서)는 아니다. 저조한 영향의 원인을 세 곳에서 실측:
  [B] wrench 토큰 feature 변동성 — obs가 바뀔 때 wrench 토큰이 변하나(죽었나).
  [C] attention — 다른 토큰(및 자기)이 wrench 토큰을 얼마나 attend 하나(받는 attention).
  [D] linear_projection — 최종 512 feature가 wrench 토큰 슬라이스를 얼마나 쓰나(가중치 크기).
"""
from __future__ import annotations

import click
import numpy as np
import torch

from analysis.modality_attribution.replay_offline import load_policy
from analysis.modality_attribution.record_infer_obs import load_inference_obs


@click.command()
@click.option("--input", "-i", required=True)
@click.option("--obs", required=True)
@click.option("--device", default="cuda")
def main(input, obs, device):
    policy, cfg = load_policy(input, device=device)
    dev = policy.device
    print(f"fuse_mode={policy.fuse_mode}  vision_feature_dim={policy.vision_feature_dim}  "
          f"force_feature_dim={policy.force_feature_dim}  n_obs_steps={policy.n_obs_steps}")
    if policy.fuse_mode != "modality-attention":
        print("이 스크립트는 modality-attention 전용입니다."); return

    Vf = policy.vision_feature_dim
    n_vis_tok = len(policy.rgb_keys) * policy.n_obs_steps
    n_tok = n_vis_tok + policy.force_obs_steps
    wrench_idx = list(range(n_vis_tok, n_tok))   # wrench 토큰 인덱스들
    tok_labels = [f"vis#{t}" for t in range(n_vis_tok)] + ["WRENCH"] * policy.force_obs_steps
    print(f"tokens({n_tok}) = {tok_labels}, 각 {Vf}차원.  wrench token idx={wrench_idx}")

    # ── obs 배치 준비 ──
    data = load_inference_obs(obs)
    frames = data["obs_by_inference"]
    keys = data["obs_keys"]
    obs_batch = {k: torch.from_numpy(np.stack([np.asarray(o[k]) for o in frames]).astype(np.float32)).to(dev)
                 for k in keys}
    N = len(frames)

    # ── in_embeds(토큰) + attention 캡처 ──
    cap = {}
    te = policy.transformer_encoder
    orig_te = te.forward
    def te_wrap(src, *a, **kw):
        cap["in_embeds"] = src.detach().float().cpu()   # (N, n_tok, Vf)  (pos-emb 더해진 입력)
        out = orig_te(src, *a, **kw)
        cap["out_embeds"] = out.detach().float().cpu()  # (N, n_tok, Vf)  (projection이 실제 보는 것)
        return out
    te.forward = te_wrap

    sa = te.self_attn
    orig_sa = sa.forward
    def sa_wrap(q, k, v, **kw):
        kw["need_weights"] = True
        kw["average_attn_weights"] = True
        out, w = orig_sa(q, k, v, **kw)
        cap["attn"] = w.detach().float().cpu()          # (N, n_tok, n_tok)  attn[b,q,k]
        return out, w
    sa.forward = sa_wrap

    try:
        with torch.no_grad():
            policy.predict_action(obs_batch)
    finally:
        te.forward = orig_te
        sa.forward = orig_sa

    in_embeds = cap["in_embeds"]   # (N, n_tok, Vf)
    attn = cap.get("attn", None)

    # ── [B] 토큰 feature 변동성 ──
    print("\n[B] 토큰 feature 변동성 (obs간): 토큰별 |per-dim std| 평균 + 토큰벡터 norm")
    std_per_tok = in_embeds.std(dim=0)          # (n_tok, Vf)
    norm_per_tok = in_embeds.norm(dim=-1).mean(0)  # (n_tok,)
    for t in range(n_tok):
        s = std_per_tok[t].numpy()
        tag = " ← wrench" if t in wrench_idx else ""
        print(f"    {tok_labels[t]:8s}: mean std={s.mean():.4f}  max std={s.max():.4f}  "
              f"|token|={norm_per_tok[t].item():.3f}{tag}")

    # ── [C] attention: 각 토큰이 '받는' 평균 attention ──
    if attn is not None:
        print("\n[C] attention — 각 토큰이 받는 평균 attention (query 평균, 클수록 fusion이 그 토큰에 의존):")
        recv = attn.mean(dim=(0, 1)).numpy()    # (n_tok,)  key 방향 = 받는 attention
        for t in range(n_tok):
            tag = " ← wrench" if t in wrench_idx else ""
            print(f"    받는 attn[{tok_labels[t]:8s}] = {recv[t]:.4f}{tag}")
        # wrench가 다른 토큰들로부터 받는 attention (자기 제외)
        wq = [q for q in range(n_tok) if q not in wrench_idx]
        for wt in wrench_idx:
            cross = attn[:, wq, wt].mean().item()
            print(f"    vision→wrench(token {wt}) cross-attn = {cross:.4f}")

    # ── [D] linear_projection 가중치: 토큰 슬라이스별 크기 ──
    W = policy.linear_projection.weight.detach().float().cpu().numpy()  # (Vf, Vf*n_tok)
    print("\n[D] linear_projection(1536→512) 토큰 슬라이스별 weight 크기 (최종 feature가 그 토큰을 얼마나 쓰나):")
    slice_fro = []
    for t in range(n_tok):
        Wt = W[:, t * Vf:(t + 1) * Vf]
        fro = np.linalg.norm(Wt)
        slice_fro.append(fro)
        tag = " ← wrench" if t in wrench_idx else ""
        print(f"    {tok_labels[t]:8s}: ||W_slice||_F={fro:.3f}  mean|w|={np.abs(Wt).mean():.5f}{tag}")

    # ── [종합] 유효 기여 = ||W_slice · diag(out_std_slice)||_F  (projection이 보는 out_embeds 기준) ──
    out_embeds = cap.get("out_embeds", in_embeds)
    out_std = out_embeds.std(dim=0).numpy()   # (n_tok, Vf)
    print("\n[B2] transformer '출력' 토큰 변동성 (projection이 실제 보는 것): 토큰별 mean std")
    for t in range(n_tok):
        tag = " ← wrench" if t in wrench_idx else ""
        print(f"    {tok_labels[t]:8s}: out mean std={out_std[t].mean():.4f}{tag}")
    print("\n[종합] 토큰별 유효기여 = ||W_slice · out_std_slice||_F (conditioning에 주는 RMS):")
    eff = []
    for t in range(n_tok):
        Wt = W[:, t * Vf:(t + 1) * Vf] * out_std[t][None, :]
        eff.append(float(np.linalg.norm(Wt)))
    tot = sum(eff)
    for t in range(n_tok):
        tag = " ← wrench" if t in wrench_idx else ""
        print(f"    {tok_labels[t]:8s}: {eff[t]:.4f}  ({100.0*eff[t]/tot:5.1f}%){tag}")
    w_share = sum(eff[t] for t in wrench_idx) / tot * 100
    print(f"\n  → wrench 토큰(들) 유효기여 합계 = {w_share:.1f}%")
    print("  해석: [B] wrench std가 작으면 인코더가 힘을 뭉갬(데이터에 force→action 신호 부족).")
    print("        [C] wrench 받는 attn이 작으면 attention이 힘을 무시하게 학습됨.")
    print("        [D] wrench W_slice가 작으면 projection이 힘 토큰을 버림.")


if __name__ == "__main__":
    main()
