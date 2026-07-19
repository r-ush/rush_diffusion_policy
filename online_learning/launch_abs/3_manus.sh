#!/usr/bin/env bash
# 터미널 3 — manus 손 (교정 중에만 발행). absolute 매핑(--relative 없음, 잘 되던 버전).
#   force 캘리브레이션 + gate-teleop(정책 중 침묵). 평소 ROS2 python 사용.
set -euo pipefail
cd /home/vision/manus_ws/src/ROS2
exec python manus_to_aidin_force.py --gate-teleop -r
