# 세션 요약 & 남은 일 (2026-07-13)

손(hand) 온라인 학습 + modality attribution 분석 도구 구축 세션. 나중에 이어서 하려고 정리.
상세 개선 계획은 `online_learning/IMPROVEMENT_PLAN.md` 참고.

---

## 0. 핵심 상수 (자주 씀)
```bash
PY=/home/vision/venv_diffusion/bin/python                 # timm+imagecodecs 있는 env
CKPT=data/outputs/260710_insert_box_hand_wrench_abs/epoch=0900-train_loss=0.001.ckpt   # base(16D 손)
V7=data/outputs/260713_insert_box_hand_online_v7/epoch=online-v7.ckpt                  # 온라인 v7 구운 것
export ONLINE_WORKDIR=data/online_runs/run_hand           # ★ 손 온라인학습 workdir (actor·learner 공통)
RUN=data/online_runs/run_hand/actor_episodes
```

---

## 1. 완료된 것

### A. 손(hand) 지원 병합 (online_learning + env/controller)
diffusion-policy 포크의 `--use_hand`(16D=팔9+손7) + wrench 노이즈 + timeseries HDF5를 **rush 포크에 병합**
(rush의 PAUSE/RESUME·record_only 온라인 기능 유지).
- `diffusion_policy/real_world/rightarm_hand_insert_plug_interpolation_controller.py` — 손 subscriber/publisher/`hand_command_publish`, 13D `rightarm_hand` 보간, `wrench_wrist_R_current`. 손 발행은 `if not paused`(교정 시 manus에 양보).
- `diffusion_policy/real_world/bae_real_env_rightarm_hand_insert_plug.py` — `use_hand`/`record_wrist_wrench`/timeseries HDF5.
- `diffusion_policy/real_world/real_inference_util.py` — `add_wrench_obs_noise` 이식(rush엔 없던 함수).
- `online_learning/online_actor_env_runner.py` — `--use_hand`, wrench 노이즈 옵션, hand 검증, teleop 시 achieved 손자세로 16D correction, `InferenceObsRecorder` 내장(자동 infer_obs 저장), hot-swap shape-mismatch 방어.
- `online_learning/relabel_utils.py` — 16D 손 relabel(pose9+hand7).
- `online_learning/online_learner.py` — 변경 없음(차원 무관).
- `online_learning/legacy/` — 손 추가 이전 팔 전용 스냅샷(자립형).

### B. manus 손 제어
- `manus_to_aidin_rush.py` / `manus_to_aidin_force.py`:
  - `--gate-teleop` — `/teleop_control ∈ (1,2)`(교정) 일 때만 손 발행 → 정책 손과 안 싸움.
  - `--relative --hand-gain 0.5 --hand-rate 0.03` (rush만 구현) — 교정 진입 grasp 유지 + glove 델타×gain + rate-limit → **튐 방지**(#3).

### C. 온라인 학습 실행/디버깅
- venv_diffusion에 **imagecodecs 설치**(dataset zarr 변환에 필요했음).
- **workdir는 run_hand로 새로** (기존 run1은 9D라 충돌).
- learner v0→v7 학습 확인(loss 0.35→0.05).
- `online_learning/bake_weights_to_ckpt.py` — mailbox weights_vN(state_dict) + base cfg → **완전한 재사용 .ckpt**. v7을 `260713_insert_box_hand_online_v7/`로 구워둠.

### D. Modality attribution 도구 (`analysis/modality_attribution/`)
- `record_infer_obs.py`(recorder), `replay_offline.py`(시간축 Δ), `visualize_attribution.py`(saliency+force축 PNG),
  `build_attribution_viewer.py`(★인터랙티브 HTML, **3-way vision/wrench/joint**), `batch_build_viewers.py`(새 에피소드 자동),
  `combine_pngs.py`, `analyze_force_influence.py`, `analyze_wrench_bottleneck.py`. README 갱신됨.
- **공정 baseline**: vision=`make_blank_vision`(중립이미지), wrench=zero, joint=평균pose. (freeze-to-start는 vision 과대평가라 폐기.)

---

## 2. 핵심 발견 (다시 볼 것)

- **modality 우세**: 공정 baseline 기준 대체로 **joint(proprioception) ≳ vision ≫ wrench**.
  abs-action이라 "현재 pose"가 action을 가장 좌우 → joint 최대. vision 2위, wrench 최소.
- **wrench가 낮은 진짜 원인**(`analyze_wrench_bottleneck.py` 실측): 이 정책은 **modality-attention** fusion,
  wrench는 vision과 동일한 512차원 토큰이고 projection 가중치도 공정함. 문제는 **wrench 토큰 출력 변동성이
  vision의 ~1/7**(유효기여 8.6%) = **인코더가 force를 상수로 뭉갬**(virtual-target action이 pose+vision으로 결정돼
  force→action 신호가 약함). → concat/가중치 문제 아님, **데이터·학습신호 문제**.
- **online 학습 검증**: train loss↓ ≠ 태스크 성공↑. **base demo mix=0 → catastrophic forgetting**(새 위치 되고 옛 위치 안 됨).

---

## 3. 실행 커맨드 레퍼런스

```bash
# 온라인 학습 (4 터미널) — export ONLINE_WORKDIR 먼저!
$PY online_learning/servo_rightarm_imp_online.py                        # 팔 servo(VR 트래커+/teleop_control=1 필요)
$PY online_learning/online_learner.py -i $CKPT                          # learner
(manus_ws)$ python manus_to_aidin_rush.py --gate-teleop -r --relative   # 손 manus(교정만, relative)
$PY online_learning/online_actor_env_runner.py -i $CKPT \
    --steps_per_inference 12 --frequency 10 --num_inference_steps 12 --use_hand   # actor

# 분석: 새 에피소드 뷰어 자동 생성 → 열기
$PY -m analysis.modality_attribution.batch_build_viewers -i $CKPT       # (--loop 30 로 감시)
xdg-open $RUN/attribution_ep000042/viewer.html

# wrench 병목 / force 영향 진단
$PY -m analysis.modality_attribution.analyze_wrench_bottleneck -i $CKPT --obs $RUN/eval_debug/episode_000042_infer_obs.hdf5

# 온라인 가중치 → 재사용 ckpt 굽기
$PY online_learning/bake_weights_to_ckpt.py -b $CKPT -w $ONLINE_WORKDIR/weights/weights_vN.pt -o <out.ckpt>
```

---

## 4. 남은 일 (TODO, 우선순위) — 상세는 IMPROVEMENT_PLAN.md

1. **[#3] manus relative 로봇 테스트 & 튜닝** — 구현됨, 하드웨어에서 gain/rate 조정만.
2. **[#2] catastrophic forgetting** — `ONLINE_NUM_BASE_DEMOS=10~30` + 옛 위치 포함 손 데이터셋(`ONLINE_BASE_DATASET`),
   또는 옛 성공 에피소드 rehearsal. (근본은 아래 #4)
3. **[#1] switch 떨림** — actor는 이미 fresh inference 함(확인). 떨림 주범은 **chunk 경계 재계획 + 손(#3)**.
   → RTC chunk 블렌딩(`residual_policy/pigdm_realtime_chunking.py`) 도입 검토.
4. **[#4] residual online (CR-DAgger식, GRU)** — #2·wrench 근본 해결. **현재 residual은 offline학습+online추론만**
   존재(online 학습 없음). 두 경로:
   - Path 1(먼저): 손+wrench용 residual **task config 신규**(기존은 팔 9D) + run_hand 데이터 fast 변환
     (`create_slow_pred_fast_dataset.py`) + `gru.yaml` 오프라인 학습 + residual eval 배포 → 효과 검증.
   - Path 2: online residual actor/learner 신규 구축(head가 tiny라 <1s/round, 온라인에 이상적).
5. **[아키텍처] wrench 인코더 collapse 개선**(선택, 근본): vision **modality dropout**(학습 중 vision 마스킹 →
   force 의존 강제) / 토큰 LayerNorm·gain / dilated causal conv. 개선 후 attribution으로 Δwrench 재측정.

---

## 5. 변경/생성 파일 인벤토리

**rush_diffusion_policy** (수정): 위 1-A, 1-D의 파일들 + `analysis/modality_attribution/README.md`.
**rush_diffusion_policy** (신규):
- `online_learning/bake_weights_to_ckpt.py`, `online_learning/legacy/*`, `online_learning/IMPROVEMENT_PLAN.md`, 이 문서.
- `analysis/modality_attribution/{visualize_attribution,build_attribution_viewer,batch_build_viewers,combine_pngs,analyze_force_influence,analyze_wrench_bottleneck}.py`
- `data/outputs/260713_insert_box_hand_online_v7/epoch=online-v7.ckpt` (v7 구운 것)
**manus_ws/src/ROS2** (수정): `manus_to_aidin_rush.py`(gate-teleop+relative), `manus_to_aidin_force.py`(gate-teleop).

---

## 6. 주의사항 (gotchas)
- 분석/학습은 **`venv_diffusion`** 에서 (timm·imagecodecs 있음). robodiff_rush엔 timm 없음.
- 손 온라인학습은 **반드시 `ONLINE_WORKDIR=run_hand`** (run1은 9D라 크래시).
- learner **재시작 시 accumulated 초기화 + weights version 0부터 덮어씀**(keep_last=2). 보존하려면 `bake_weights_to_ckpt.py`로 구울 것.
- infer_obs는 **정책 추론이 있는(유지 종료) 에피소드**만 저장됨. 교정만 한 에피소드는 없음.
- attribution 뷰어 폴더는 6자리(`attribution_ep000042`).
- servo는 **VR(Vive) 트래커 기반** — 스페이스마우스 아님. `/dsr01/joint_states` + `/teleop_control=1` + Vive `/tf` 필요.
