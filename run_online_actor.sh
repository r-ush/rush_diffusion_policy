#!/usr/bin/env bash
# 온라인 DAgger - Actor (실로봇/카메라) 실행
# 사용법:  bash run_online_actor.sh
# 로그: online_learning/logs/actor.log (Claude가 읽어서 점검)
# 페달: 왼(a)=correction 토글, 가운데(b)=유지(S), 오른(c)=폐기(D)
set -o pipefail

cd "$(dirname "$0")"

# 오른팔 base ckpt (learner와 동일하게)
export ONLINE_BASE_CKPT="/home/vision/rush_diffusion_policy/data/outputs/logistics_abs_10/epoch=0300-train_loss=0.002.ckpt"
# learner와 반드시 동일해야 함 (파일 통신 폴더). 깨끗한 새 폴더로 시작.
export ONLINE_WORKDIR="/home/vision/rush_diffusion_policy/data/online_runs/run2"

LOG="online_learning/logs/actor.log"
echo "===== Actor start $(date '+%F %T') =====" | tee "$LOG"
echo "ONLINE_BASE_CKPT=$ONLINE_BASE_CKPT" | tee -a "$LOG"

# -u / PYTHONUNBUFFERED: Python print가 tee 파이프에서 버퍼에 갇히지 않게 (진행 메시지 즉시 표시)
export PYTHONUNBUFFERED=1
python -u online_learning/online_actor_env_runner.py \
    --num_inference_steps 60 2>&1 | tee -a "$LOG"
