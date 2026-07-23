#!/usr/bin/env bash
# 개입(intervention) 수집 리포트 갱신 — 에피소드가 늘 때마다 이거 한 번만 돌리면 된다.
#
#   1) intervention_report.html : 에피소드/구간 품질 판정(병진·회전 신호 비, coherence, 3D 궤적)
#   2) residual_playback.html   : 프레임 단위 재생(base 가 가려던 곳 vs 사람이 고친 곳)
#
# 사용법
#   online_learning/refresh_intervention_report.sh            # 1회 갱신
#   online_learning/refresh_intervention_report.sh --watch    # 새 에피소드 들어올 때마다 자동 갱신
#
# env 로 조절 (기본값은 이 워크스페이스 기준)
#   TRANSITIONS  : 에피소드 폴더
#   OUT          : HTML 출력 폴더
#   WORLD_ROT    : base->world X축 회전각(도)
#   SPI          : actor 의 --steps_per_inference (nominal chunk 위상 계산용)
#   FREQ         : actor 의 --frequency (Hz)
#   PY           : python 실행기
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/vision/venv_diffusion/bin/python}"
TRANSITIONS="${TRANSITIONS:-$ROOT/data/online_runs/run_hand_intervention/transitions}"
OUT="${OUT:-$ROOT/data/verify_intervention}"
WORLD_ROT="${WORLD_ROT:-135}"   # 부호가 반대로 보이면 WORLD_ROT=-135 로 실행
SPI="${SPI:-6}"
FREQ="${FREQ:-10}"

refresh() {
  local n
  n=$(find "$TRANSITIONS" -maxdepth 1 -name 'ep_*.hdf5' | wc -l)
  echo "── [$(date '+%H:%M:%S')] 에피소드 ${n}개로 리포트 갱신 ──"

  "$PY" "$ROOT/online_learning/export_intervention_report_html.py" \
    --episodes "$TRANSITIONS" --out "$OUT" \
    --steps_per_inference "$SPI" --frequency "$FREQ" --world_rot_x_deg "$WORLD_ROT"

  "$PY" "$ROOT/online_learning/export_traj3d_playback_html.py" \
    --episodes "$TRANSITIONS" --head none \
    --world_rot_x_deg "$WORLD_ROT" --out "$OUT" >/dev/null

  echo "   $OUT/intervention_report.html"
  echo "   $OUT/residual_playback.html"
}

if [[ "${1:-}" != "--watch" ]]; then
  refresh
  exit 0
fi

# ── watch 모드: 새 에피소드가 '완전히 쓰인 뒤'(mailbox 의 .ready 마커 기준) 갱신 ──
echo "[watch] $TRANSITIONS 감시 중 — 새 에피소드가 들어오면 자동 갱신. 종료는 Ctrl-C."
last=""
while true; do
  cur=$(find "$TRANSITIONS" -maxdepth 1 -name 'ep_*.ready' | sort | tr '\n' ' ')
  if [[ "$cur" != "$last" && -n "$cur" ]]; then
    # actor 가 hdf5 를 다 쓰고 .ready 를 남기지만, 파일시스템 flush 여유를 조금 준다.
    sleep 2
    if refresh; then last="$cur"; else echo "[watch] 갱신 실패 — 다음 주기에 재시도"; fi
  fi
  sleep 5
done
