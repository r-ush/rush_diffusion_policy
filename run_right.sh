#!/usr/bin/env bash
# 오른팔 임피던스 eval 실행 (logistics_abs_10 체크포인트)
# 사용법:  bash run_right.sh
set -e

python rush_eval_real_robot_imp_right.py \
    --input "data/outputs/logistics_abs_10/epoch=0300-train_loss=0.002.ckpt" \
    --output data/results \
    --steps_per_inference 12 \
    --frequency 10 \
    --num_inference_steps 60
