# 온라인 학습 개선 계획 (검토 + 로드맵)

작성: 2026-07-13. 대상: `online_learning/` (손 16D, base=`260710_insert_box_hand_wrench_abs`).
4개 이슈를 코드/데이터로 검토한 결과와 우선순위별 실행 계획.

---

## 핵심 통찰
**#2(망각)와 #4(residual)는 같은 해법을 가리킨다.** frozen base + 작게 학습하는 residual head는
(a) base 능력을 안 잊고, (b) 라운드가 <1s로 빨라지며, (c) force→Δ를 명시해 wrench 병목까지 완화한다.
→ **단기(Phase 1)**는 현 파이프라인에서 즉효 패치, **중기(Phase 2)**는 residual-online 전환.

---

## 이슈별 진단 (근거)

### #1 교정→정책 switch 떨림 / "stale inference"
- **코드 확인**: `process_key('b')` → `teleop.publish(4); env.resume_robot(); mode='policy'`. 컨트롤러 `RESUME`은
  `pose_interp`를 **현재 팔(+손) 포즈 단일 waypoint로 리셋**(스테일 waypoint 제거). teleop 중엔
  `exec_actions(record_only=True)`라 waypoint를 아예 안 쌓음 → **팔에 남은 옛 chunk는 없음**.
- 전환 후 다음 loop에서 **fresh inference가 실제로 돈다**(스테일 재사용 아님).
- **데이터(ep39)**: 교정→정책 경계에서 팔 target 점프 ~12mm, 스텝간 변화 최대 ~14mm → 치명적 stale 아님.
- **결론**: 떨림 주범은 (a) **diffusion chunk 경계 재계획 불연속**(매 12스텝 새 궤적), (b) **손 mismatch(#3)**,
  (c) 전환 직후 ~0.1–0.2s inference 지연 동안의 hold→급가속. "옛 inference가 남아 튄다"는 아님.

### #2 Catastrophic forgetting (새 위치 되고 옛 위치 안 됨)
- `config_online.py`: `NUM_BASE_DEMOS_TO_MIX=0` → **base 데이터 mix 없음**. correction만으로 **diffusion 전체**를
  LR 1e-5로 fine-tune → 새 분포로 이동하며 옛 위치 능력 소실. 전형적 망각.

### #3 손 튐 (manus ↔ 로봇손 mismatch)
- `manus_to_aidin_rush.py`: glove ergonomics를 **절대 관절각**으로 매핑해 발행 → 교정 진입 순간 로봇의
  현재 grasp와 무관하게 glove 절대 자세로 점프. "grasp 유지 + 필요시 수정" 이 안 됨.

### #4 CR-DAgger식 residual online
- `diffusion_policy/residual_policy/`: `FastResidualTemporalPolicy`(GRU) / `FastResidualContextStepPolicy`(MLP)
  = **frozen slow(diffusion) + 작은 head**. `workspace.py`는 **offline** 학습 루프(DataLoader/EMA/ckpt).
  최종 action = base + residualΔ (`predict_slow_fast_residual_action`). **online learner는 아직 없음** → 신규 필요.

---

## 실행 계획

### Phase 1 — 즉효 패치 (현 파이프라인 유지, 1~2일)

**P1-a. #2 망각 완화 (코드 변경 최소, 최우선)**
- `ONLINE_NUM_BASE_DEMOS=10~30` + `ONLINE_BASE_DATASET`에 **원래 박스 위치 포함 손(16D) 데이터셋** 지정.
- 옛 성공 에피소드를 accumulated에 상시 rehearsal.
- `ONLINE_LR`↓ / `ONLINE_EPOCHS_PER_ROUND`↓ 로 급이동 억제.
- (선택) `compute_loss`에 **L2-to-base(EWC lite)** 앵커: `+λ‖θ−θ_base‖²`.

**P1-b. #1 switch 떨림 완화**
- `process_key('b')`에서 mode 전환 직후 **즉시 1회 inference+exec** + iter_idx/타이밍 재동기화(대기 갭 제거).
- chunk 경계 블렌딩: `residual_policy/pigdm_realtime_chunking.py`(RTC) 도입 또는 연속 chunk overlap-blend
  → 재계획 불연속 완화(떨림의 실제 주범).
- 전환 시 손도 현재 grasp로 resync 확인(이미 RESUME이 curr_hand 포함).

**P1-c. #3 손 relative 교정 모드**
- `manus_to_aidin_rush.py`: 교정 진입(`/teleop_control==1`) 순간 **glove 기준값 캡처** → 발행 =
  `현재 로봇 grasp + gain×(glove − glove_ref)`. grasp 유지 + glove 델타만 반영.
- rate-limit / low-pass 로 튐 억제. (선택) 특정 손가락만 수정, 나머지 hold.

### Phase 2 — 전략 전환: residual online (중기, #2·wrench 근본 해결)
- **online residual actor/learner 신규 구축**:
  - actor: slow+residual(`predict_slow_fast_residual_action`)로 롤아웃, 교정 시 `base_action` + achieved 기록
    → residual target = (achieved − base).
  - learner: **작은 head만** 학습(<1s/round). frozen base → **망각 원천 차단(#2 해결)**.
  - residual에 `train_force_encoder=True` → **force→Δ 명시** → wrench 병목 개선.
- 데이터: 온라인 relabel을 residual 포맷(`base_action_rel`, `residual_delta6`)으로 변환하는 online 버전 필요.
- 검증: 우리가 만든 attribution 툴로 **Δwrench 상승** + 옛/새 위치 성공률 동시 확인.

---

## 우선순위 제안
1. **P1-a (망각)** — 지금 가장 아픈 것, 환경변수+데이터로 즉시.
2. **P1-c (손 relative)** — 교정 품질에 직결, manus 소폭 수정.
3. **P1-b (switch/RTC)** — 떨림.
4. **Phase 2 (residual online)** — 근본 해결, 신규 구현(가장 큼).

## 검증 루프 (공통)
매 개선 후: v_N 굽기(`bake_weights_to_ckpt.py`) → 롤아웃 → `batch_build_viewers` attribution →
**옛/새 위치 성공률 + Δwrench** 비교.
