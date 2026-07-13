#!/usr/bin/env python
"""왜 wrench(force) attribution이 작게 나오는지 진단.

세 가지를 본다:
  [1] wrench 값 크기 — 실제 롤아웃의 손목 힘이 얼마나 컸나 (작으면 zero-ablation 효과도 작음).
  [2] baseline 비대칭 — vision 'freeze-to-start'는 시간이 갈수록 매우 큰 perturbation이 되고,
      wrench 'zero'는 (원래 힘이 작으면) 작은 perturbation. 그래서 Δvision이 과대평가될 수 있다.
      vision 'self'(자기 직전 프레임 고정, 작은 perturbation)와, wrench x5(힘 증폭)로 대조.
  [3] wrench 민감도 스윕 — 힘을 인위로 키워 넣었을 때 action이 바뀌나?
      바뀌면 = 정책이 force를 '쓸 능력은 있다'(작게 나온 건 실제 힘이 작아서).
      안 바뀌면 = 정책이 force를 거의 '무시'하도록 학습됨.
"""
from __future__ import annotations

import click
import numpy as np
import torch

from analysis.modality_attribution import attribution as attr
from analysis.modality_attribution.replay_offline import load_policy
from analysis.modality_attribution.record_infer_obs import load_inference_obs

AX = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]


def make_scale_wrench(policy, factor):
    def fn(od):
        out = attr.clone_obs(od)
        for k in policy.wrench_keys:
            if k in out:
                out[k] = out[k] * factor
        return out
    return fn


def make_add_wrench_axis(policy, key, axis, value):
    def fn(od):
        out = attr.clone_obs(od)
        out[key][..., axis, :] = out[key][..., axis, :] + value
        return out
    return fn


@click.command()
@click.option("--input", "-i", required=True)
@click.option("--obs", required=True)
@click.option("--seeds", default="0,1,2")
@click.option("--device", default="cuda")
def main(input, obs, seeds, device):
    seeds = [int(s) for s in str(seeds).split(",") if s.strip() != ""]
    policy, cfg = load_policy(input, device=device)
    dev = policy.device
    wk = policy.wrench_keys[0] if policy.wrench_keys else None
    print(f"rgb_keys={policy.rgb_keys}  wrench_keys={policy.wrench_keys}")
    if wk is None:
        print("이 정책엔 wrench modality가 없습니다."); return

    # normalizer에 wrench가 포함되는지 (정규화되면 스케일 확인)
    try:
        pk = list(policy.normalizer.params_dict.keys())
        print("normalizer keys:", pk)
        if wk in pk:
            p = policy.normalizer.params_dict[wk]
            print(f"  '{wk}' normalizer scale mean={p['scale'].mean().item():.4g} "
                  f"offset mean={p['offset'].mean().item():.4g}")
        else:
            print(f"  '{wk}' 는 normalizer에 없음 → force encoder가 raw wrench를 직접 받음")
    except Exception as e:
        print("normalizer introspect 실패:", e)

    data = load_inference_obs(obs)
    obs_list = data["obs_by_inference"]
    N = len(obs_list)
    start_obs = attr.obs_np_to_tensor(obs_list[0], dev)

    # ── [1] wrench 값 크기 ──
    wr = np.stack([o[wk] for o in obs_list])  # (N,1,6,32)
    print("\n[1] 실제 롤아웃 wrench 크기 (축별 |mean|, max|.|):")
    for c, a in enumerate(AX):
        v = wr[:, 0, c, :]
        print(f"    {a}: |mean|={np.abs(v).mean():.3f}   max|.|={np.abs(v).max():.3f}")
    print(f"    전체 |mean|={np.abs(wr).mean():.3f}   (calibrated: 무접촉≈0, 접촉 시 커짐)")

    frames = sorted(set(np.linspace(0, N - 1, min(6, N)).astype(int).tolist()))

    # ── [2] baseline 비대칭 ──
    print("\n[2] frame별 Δaction:  vis_blank(공정)  vis_start  vis_self  |  wrench_zero(공정)  wrench_x5")
    print("    ※ 공정 비교 = vis_blank(화면 통째 제거) ↔ wrench_zero(힘 통째 제거)")
    for i in frames:
        od = attr.obs_np_to_tensor(obs_list[i], dev)
        b = {
            "vb": attr.make_blank_vision(policy),               # ★ 공정: 화면 전체 제거
            "vs": attr.make_freeze_vision(policy, start_obs),   # 시작 고정(이동에 과대)
            "vf": attr.make_freeze_vision(policy, od),          # 자기 고정(광류만, 과소)
            "wz": attr.make_zero_wrench(policy),                # ★ 공정: 힘 제거
            "w5": make_scale_wrench(policy, 5.0),               # 힘 5배 증폭
        }
        r = attr.ablation_deltas(policy, od, b, seeds=seeds).deltas
        print(f"    f{i:2d}:  vis_blank={r['vb'].total:.4f}   vis_start={r['vs'].total:.4f}   "
              f"vis_self={r['vf'].total:.4f}  |  wrench_zero={r['wz'].total:.4f}   wrench_x5={r['w5'].total:.4f}")

    # ── [3] wrench 민감도 스윕 ──
    i = frames[len(frames) // 2]
    od = attr.obs_np_to_tensor(obs_list[i], dev)
    base = {s: attr.predict_action(policy, od, s)["action"] for s in seeds}
    print(f"\n[3] wrench 민감도 스윕 (frame {i}):")
    print("    (a) Fx에 상수 힘 주입 → Δaction")
    for off in [1, 2, 5, 10, 20, 50]:
        bld = make_add_wrench_axis(policy, wk, 0, float(off))
        ab = bld(od)
        ds = [attr.action_delta(policy, base[s], attr.predict_action(policy, ab, s)["action"]).total for s in seeds]
        print(f"      Fx += {off:5.1f}  → Δaction={np.mean(ds):.4f}")
    print("    (b) 전체 wrench 스케일 → Δaction")
    for sc in [0.0, 2.0, 5.0, 10.0, 20.0]:
        bld = make_scale_wrench(policy, sc)
        ab = bld(od)
        ds = [attr.action_delta(policy, base[s], attr.predict_action(policy, ab, s)["action"]).total for s in seeds]
        print(f"      wrench x{sc:4.1f}  → Δaction={np.mean(ds):.4f}")

    print("\n해석 가이드:")
    print("  · 공정 비교는 [2]의 vis_blank ↔ wrench_zero (둘 다 '해당 modality를 통째로 제거').")
    print("    vis_blank >> wrench_zero 면 → vision이 실제로 더 지배적(baseline artifact 아님).")
    print("  · [3]에서 큰 힘엔 Δ가 커지면 → 정책은 force를 '쓸 능력은 있다'. 낮은 attribution은")
    print("    실제 힘(~1N)이 moderate라 제거 효과가 작은 것(정책은 큰 힘에 크게 반응).")
    print("  · vis_self는 '2프레임 광류'만 재므로 vision 의존도의 척도가 아님(과소평가) — 쓰지 말 것.")


if __name__ == "__main__":
    main()
