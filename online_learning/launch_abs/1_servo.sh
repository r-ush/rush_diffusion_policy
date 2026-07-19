#!/usr/bin/env bash
# 터미널 1 — 팔 servo (VR Vive + teleop). RESIDUAL env 불필요.
set -euo pipefail
cd /home/vision/rush_diffusion_policy
exec /home/vision/venv_diffusion/bin/python online_learning/servo_rightarm_imp_online.py
