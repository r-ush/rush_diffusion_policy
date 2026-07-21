#!/usr/bin/env python
"""
온라인 residual DAgger Learner — INTERVENTION(physical-push) 판.

residual_teleop_learner.ResidualOnlineLearner 를 상속해 두 가지만 바꾼다:
  1. config 모듈을 config_residual_intervention 으로(WORKDIR/config_name/가중치 등 개입 전용).
  2. `_make_sampler` 를 override — 데이터셋 replay_buffer 의 per-step is_intervention 플래그로
     **개입 프레임을 INTERVENTION_SAMPLE_WEIGHT 배 가중**하는 WeightedRandomSampler 를 만든다.
     (cr-dagger DynamicDataset 의 correction 업샘플링에 대응.)

나머지(정책 인스턴스화 / warm-continue 학습 / head+normalizer 발행 / hot-swap 프로토콜)는
teleop 판과 완전히 동일하다 — 개입형은 actor 의 제어 방식만 다르고 relabel/학습 포맷은 같기
때문이다.

가중 근거:
  개입형 actor 는 에피소드 전체 프레임을 보낸다. 안 민 프레임은 achieved≈base → residual≈0
  (nominal 상태에서 "교정 안 함" 을 가르치는 negative), 민 프레임만 residual=사람 밀림.
  전체를 균등 학습하면 residual≈0 이 지배해 신호가 희석되므로, 개입 프레임을 가중해 뽑되
  negative 도 남겨 과도교정을 막는다.

실행 (env 는 timm 있는 bae_robodiff):
  export RESIDUAL_INTERVENTION_WORKDIR=data/online_runs/run_hand_intervention
  /home/rush/anaconda3/envs/bae_robodiff/bin/python online_learning/residual_intervention_learner.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from online_learning import config_residual_intervention as IC
from online_learning.residual_teleop_learner import ResidualOnlineLearner


def intervention_sample_weights(indices, is_intervention, weight):
    """각 sample 윈도우의 "현재 스텝"(pad_after=0 이므로 buf_end-1)이 개입이면 weight, 아니면 1.

    indices          : dataset.sampler.indices (각 원소 (buf_start, buf_end, samp_start, samp_end))
    is_intervention  : replay_buffer per-step 개입 플래그(모든 에피소드 concat), (총스텝,) 또는 (총스텝,1)
    weight           : 개입 프레임 가중치(>1)
    반환             : (len(indices),) float64 가중치 (정책 없이도 계산 가능 = 테스트용).
    """
    isint = np.asarray(is_intervention, dtype=np.float64).reshape(-1)
    w = np.ones(len(indices), dtype=np.float64)
    for i, idx in enumerate(indices):
        step = int(idx[1]) - 1
        if 0 <= step < isint.shape[0] and isint[step] > 0.5:
            w[i] = float(weight)
    return w


class ResidualInterventionLearner(ResidualOnlineLearner):
    tag = "[I-Learner]"

    def __init__(self, cfg=None):
        super().__init__(cfg=cfg if cfg is not None else IC)

    def _make_sampler(self, dataset):
        """개입 프레임 가중 WeightedRandomSampler. is_intervention 없으면 기본 샘플러로 폴백."""
        rb = dataset.replay_buffer
        if "is_intervention" not in rb:
            print(f"{self.tag} is_intervention 키 없음 → 기본(가중 없음) 샘플러")
            return super()._make_sampler(dataset)

        # replay_buffer 의 per-step 개입 플래그(모든 에피소드 concat).
        isint = np.asarray(rb["is_intervention"]).reshape(-1)
        indices = dataset.sampler.indices  # 각 sample: (buf_start, buf_end, samp_start, samp_end)
        W = float(self.C.INTERVENTION_SAMPLE_WEIGHT)
        weights = intervention_sample_weights(indices, isint, W)

        n_int = int((weights > 1.0).sum())
        n_steps_int = int((isint > 0.5).sum())
        n_samples = (self.C.MAX_SAMPLES_PER_EPOCH
                     if self.C.MAX_SAMPLES_PER_EPOCH > 0 else len(indices))
        print(f"{self.tag} 가중 샘플러: 개입 윈도우 {n_int}/{len(indices)} "
              f"(개입 스텝 {n_steps_int}/{isint.shape[0]}), W={W}, num_samples={n_samples}")
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=int(n_samples), replacement=True)
        return sampler, False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="온라인 residual DAgger learner (intervention 판).")
    parser.add_argument("--slow_ckpt", default=None, help="frozen slow base ckpt. 미지정 시 config 값.")
    parser.add_argument("--config_name", default=None, help="residual hydra config 이름.")
    args = parser.parse_args()
    if args.slow_ckpt is not None:
        IC.SLOW_CKPT = args.slow_ckpt
    if args.config_name is not None:
        IC.RESIDUAL_CONFIG_NAME = args.config_name
    print(f"[I-Learner] SLOW_CKPT={IC.SLOW_CKPT}")
    print(f"[I-Learner] CONFIG={IC.RESIDUAL_CONFIG_NAME}  WORKDIR={IC.ONLINE_WORKDIR}")
    print(f"[I-Learner] INTERVENTION_SAMPLE_WEIGHT={IC.INTERVENTION_SAMPLE_WEIGHT}")
    ResidualInterventionLearner().run()
