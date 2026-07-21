#!/usr/bin/env bash
# 온라인 DAgger - Learner (GPU) 실행
# 사용법:  bash run_online_learner.sh
# 로그: online_learning/logs/learner.log (Claude가 읽어서 점검)
set -o pipefail

cd "$(dirname "$0")"

# 오른팔 base ckpt (logistics_abs_10, epoch=0200)
export ONLINE_BASE_CKPT="/home/vision/rush_diffusion_policy/data/outputs/logistics_abs_10/epoch=0300-train_loss=0.002.ckpt"
# actor와 반드시 동일해야 함 (파일 통신 폴더). 깨끗한 새 폴더로 시작.
export ONLINE_WORKDIR="/home/vision/rush_diffusion_policy/data/online_runs/run2"

LOG="online_learning/logs/learner.log"
echo "===== Learner start $(date '+%F %T') =====" | tee "$LOG"
echo "ONLINE_BASE_CKPT=$ONLINE_BASE_CKPT" | tee -a "$LOG"

# -u / PYTHONUNBUFFERED: Python print가 tee 파이프에서 버퍼에 갇히지 않게 (대기 메시지 즉시 표시)
export PYTHONUNBUFFERED=1
python -u online_learning/finetune_teleop_learner.py 2>&1 | tee -a "$LOG"
