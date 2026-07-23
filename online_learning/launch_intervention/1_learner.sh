#!/usr/bin/env bash
# 터미널 1 — INTERVENTION residual head learner (개입 프레임 가중 학습 후 head 발행).
#   servo/manus 불필요. teleop 판과 분리된 전용 workdir(run_hand_intervention).
#   abs slow base (teleop 판 launch_abs 와 동일 base).
set -euo pipefail
export RESIDUAL_INTERVENTION_SLOW_CKPT=/home/vision/diffusion-policy/data/outputs/260710_insert_box_hand_wrench_abs/epoch=0500-train_loss=0.003.ckpt
export RESIDUAL_INTERVENTION_WORKDIR=/home/vision/rush_diffusion_policy/data/online_runs/run_hand_intervention
# (선택) 개입 프레임 가중 배수. 기본 5.0.
# export INTERVENTION_SAMPLE_WEIGHT=5.0
cd /home/vision/rush_diffusion_policy
exec /home/vision/venv_diffusion/bin/python online_learning/residual_intervention_learner.py "$@"
