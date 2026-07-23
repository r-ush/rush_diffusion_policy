#!/usr/bin/env bash
# 터미널 2 — INTERVENTION residual actor.
#   팔: base(+residual)를 임피던스로 계속 실행 + 사람이 물리적으로 밀어 교정.
#   손: 개입 중에도 base(slow) 자율 제어.  servo/manus 불필요(핸드오프 안 함).
#   페달/키(이 터미널 포커스): a=개입시작(밀기)  b=개입종료  s 또는 c=유지+전송  d=폐기  q=종료.
#   learner 가 head 발행 전에는 slow-only 로 돌며 교정만 수집(정상).
#   추가 인자는 그대로 전달된다. 예: 2_actor.sh --residual_gain_scale 0.8
set -euo pipefail
export RESIDUAL_INTERVENTION_SLOW_CKPT=/home/vision/diffusion-policy/data/outputs/260710_insert_box_hand_wrench_abs/epoch=0500-train_loss=0.003.ckpt
export RESIDUAL_INTERVENTION_WORKDIR=/home/vision/rush_diffusion_policy/data/online_runs/run_hand_intervention
cd /home/vision/rush_diffusion_policy
exec /home/vision/venv_diffusion/bin/python online_learning/residual_intervention_actor_env_runner.py \
  -i /home/vision/diffusion-policy/data/outputs/260710_insert_box_hand_wrench_abs/epoch=0500-train_loss=0.003.ckpt \
  --use_hand --steps_per_inference 6 --frequency 10 --num_inference_steps 12 \
  --max_duration 86400 \
  "$@"
