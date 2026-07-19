#!/usr/bin/env bash
# 터미널 4 — residual online actor (slow abs base + fast residual + 교정수집 + hot-swap + 전송).
#   learner가 head 발행 전에는 slow-only로 돌며 교정만 수집(정상).
set -euo pipefail
export RESIDUAL_SLOW_CKPT=/home/vision/rush_diffusion_policy/data/outputs/260710_insert_box_hand_wrench_abs/epoch=0900-train_loss=0.001.ckpt
export RESIDUAL_ONLINE_WORKDIR=/home/vision/rush_diffusion_policy/data/online_runs/run_hand_residual_abs
cd /home/vision/rush_diffusion_policy
exec /home/vision/venv_diffusion/bin/python online_learning/residual_online_actor_env_runner.py \
  -i /home/vision/rush_diffusion_policy/data/outputs/260710_insert_box_hand_wrench_abs/epoch=0900-train_loss=0.001.ckpt \
  --use_hand --steps_per_inference 6 --frequency 10 --num_inference_steps 12
