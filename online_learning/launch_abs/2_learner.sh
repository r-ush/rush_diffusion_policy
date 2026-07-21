#!/usr/bin/env bash
# 터미널 2 — residual head learner (교정 데이터로 학습 후 head 발행).
#   abs slow base + 전용 workdir(run_hand_residual_abs).
set -euo pipefail
export RESIDUAL_SLOW_CKPT=/home/vision/diffusion-policy/data/outputs/260710_insert_box_hand_wrench_abs/epoch=0500-train_loss=0.003.ckpt
export RESIDUAL_ONLINE_WORKDIR=/home/vision/rush_diffusion_policy/data/online_runs/run_hand_residual_abs
cd /home/vision/rush_diffusion_policy
exec /home/vision/venv_diffusion/bin/python online_learning/residual_teleop_learner.py
