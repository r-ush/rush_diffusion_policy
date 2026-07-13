#!/usr/bin/env python
"""mailbox의 weights_vN.pt(온라인 학습 가중치, state_dict만) + base ckpt(cfg/구조) →
   완전한 재사용 가능 체크포인트(.ckpt)로 '굽는다'.

online_learner가 발행하는 weights_vN.pt는 state_dict만 들어있어(cfg 없음) 그 자체로는
actor/eval의 -i 로 못 쓴다. 이 스크립트로 base ckpt의 cfg + 새 가중치를 합쳐 저장하면
그대로 --input <baked.ckpt> 로 롤아웃/평가하거나, 새 온라인 학습의 base로 쓸 수 있다.

  python online_learning/bake_weights_to_ckpt.py \
      -b data/outputs/260710_insert_box_hand_wrench_abs/epoch=0900-train_loss=0.001.ckpt \
      -w data/online_runs/run_hand/weights/weights_v7.pt \
      -o data/outputs/260713_insert_box_hand_online_v7/epoch=online-v7.ckpt
"""
import os
import pathlib

import click
import dill
import torch


@click.command()
@click.option("--base", "-b", required=True, help="base ckpt(.ckpt) — cfg/architecture 제공")
@click.option("--weights", "-w", required=True, help="mailbox weights_vN.pt (state_dict)")
@click.option("--output", "-o", required=True, help="저장할 완전한 .ckpt 경로")
@click.option("--drop-optimizer/--keep-optimizer", default=True,
              help="optimizer state 제거(배포용, 기본) / 유지(fine-tune 이어하기).")
def main(base, weights, output, drop_optimizer):
    print(f"[bake] base ckpt 로드: {base}")
    payload = torch.load(open(base, "rb"), pickle_module=dill, weights_only=False)
    print(f"[bake] weights 로드: {weights}")
    wb = torch.load(weights, map_location="cpu", weights_only=False)
    sd = wb["state_dict"]

    ema = payload["state_dicts"].get("ema_model", {})
    # 키 일치 검증 (같은 아키텍처여야 함)
    missing = set(ema.keys()) - set(sd.keys())
    extra = set(sd.keys()) - set(ema.keys())
    if missing or extra:
        print(f"[bake][WARN] 키 불일치 missing={len(missing)} extra={len(extra)} — 그래도 주입 진행")

    payload["state_dicts"]["ema_model"] = {k: v for k, v in sd.items()}
    payload["state_dicts"]["model"] = {k: v for k, v in sd.items()}
    if drop_optimizer and "optimizer" in payload["state_dicts"]:
        del payload["state_dicts"]["optimizer"]
        print("[bake] optimizer state 제거(배포용).")

    out = pathlib.Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        torch.save(payload, f, pickle_module=dill)
    size_gb = out.stat().st_size / 1e9
    print(f"[bake] 완료: v{wb.get('version')} (num_demos={wb.get('num_demos')}) "
          f"주입 {len(sd)}개 → {out}  ({size_gb:.2f} GB)")
    print(f"[bake] 사용:  --input {out}   (actor/eval/새 온라인학습 base로 그대로 사용)")


if __name__ == "__main__":
    main()
