# CR-DAgger식 Residual-Online DAgger — 인수인계

> 2026-07-15. `online_learning/` 의 full-finetune 온라인 DAgger를 **진짜 cr-dagger식
> residual-online** 으로 옮긴 작업. frozen slow base + tiny residual head 만 온라인 학습.
> 망각(#2) 원천 차단, 라운드 <수초, force→Δ 명시로 wrench 병목 완화 여지.
>
> **로봇에서 완성·실행할 때는 `RESIDUAL_ONLINE_ROBOT_RUNBOOK.md`(actor 완성 훅3 + 4터미널
> 실행 + 튜닝/검증/트러블슈팅 체크리스트) 를 보라.** 이 문서는 설계/배경, 런북은 실전 절차.

## 0. 한 문단 요약
Base(slow) diffusion 을 **동결**하고, 작은 residual head 만 사람 교정 데이터로 온라인 학습한다.
actor 는 매 스텝 slow chunk 를 뽑고 그 위에 head 의 잔차(Δ6D pose)를 얹어 실행한다. 사람이
임피던스/servo 로 팔을 밀어 교정하면, 그 결과 achieved pose 가 곧 **residual target = achieved −
slow_pred** 가 된다. learner 가 head 를 warm-continue 로 갱신해 발행하고, actor 가 에피소드
경계에서 **head 만 hot-swap** 한다. full state_dict 가 아니라 head(+normalizer) 만 오가므로
가볍고 빠르다.

## 1. 결정된 설계(이번 세션)
- **경로**: Path 2 (오프라인 검증 건너뛰고 residual-online 직행).
- **base**: hand 16D (`260714_insert_box_hand_rel/epoch=0500-train_loss=0.002.ckpt`, SSD).
  `bae_diffusion_unet_hybrid_image_wrench_encoder_policy` = residual-호환(vision_encoder/
  force_encoder/wrench_keys 등 노출 확인). obs=image0+robot_pose_R+robot_quat_R+hand_pose_R(7)+
  wrench_wrist_R[6,32], action 16D, `action_pose_repr=relative`, n_obs=2/horizon=16.
- **residual base 기준**: **slow 예측**(predicted-base). residual = 사람교정 − slow예측.
- **residual 범위**: **pose-only 6D**(v1). 손 명령은 slow 그대로. 손 교정 학습은 v2.
- **정렬**: actor 가 실시간 slow 를 돌리므로 same-row 페어링 → task config `action_target_shift=0`
  (relabel 이 "한 스텝 뒤 achieved = virtual" 정렬을 내부에서 끝냄).
- **env**: `bae_robodiff` (timm 있음). `robodiff` 엔 timm 없음. torch2.8/cuda.

## 2. 만든 것 (검증됨 ✅ = 로봇 없이 통과)
| 파일 | 역할 | 상태 |
|------|------|------|
| `diffusion_policy/config/residual_policy/task/hand_online.yaml` | hand 16D · pose-only 6D · slow-pred 키 · shift=0 task | ✅ |
| `diffusion_policy/config/residual_policy/hand_online_mlp.yaml` | top-level residual config(context-step MLP head) | ✅ |
| `online_learning/config_residual_online.py` | 공유 설정(SLOW_CKPT, config_name, LR/epochs/workdir, 환경변수 override) | ✅ |
| `online_learning/residual_relabel_utils.py` | 에피소드→residual 포맷 HDF5(slow_pred_target_abs, virtual=다음스텝 achieved, residual_delta6) | ✅ |
| `online_learning/residual_online_learner.py` | learner: FastResidualContextStepPolicy 인스턴스화 → accumulated → head warm-continue 학습 → **head+normalizer 만 발행** | ✅ |
| `online_learning/smoke_test_residual_no_robot.py` | 로봇 없이 A 호환/B 학습/C hot-swap 전 루프 검증 | ✅ 통과 |
| `online_learning/residual_online_actor_env_runner.py` | actor: slow chunk+per-step fast residual, one-step-per-tick, head hot-swap, slow_pred 로깅, residual 전송, Δ캡, 교정 핸드오프 | ⚠️ 초안(py_compile OK), 로봇 VERIFY 남음 |

**smoke 결과**: 합성 3에피소드(69샘플) → head loss 0.095→0.041, 라운드 8.9s, hot-swap max|Δ|=0.

## 3. 실행 (learner 쪽, 로봇 없이 가능)
```bash
PY=/home/rush/anaconda3/envs/bae_robodiff/bin/python
export RESIDUAL_ONLINE_WORKDIR=data/online_runs/run_hand_residual
# (선택) slow ckpt / config override:
#   export RESIDUAL_SLOW_CKPT=/media/rush/.../260714_insert_box_hand_rel/epoch=0500-train_loss=0.002.ckpt
$PY online_learning/smoke_test_residual_no_robot.py          # 먼저 sanity check
$PY online_learning/residual_online_learner.py               # learner 상시 구동(actor 대기)
```
발행 payload: `{version, num_demos, head_state, normalizer_state, force_encoder_state}`.

## 4. 남은 일 — actor 로봇 VERIFY + 전체 검증 ★
actor 는 **작성 완료**(`online_learning/residual_online_actor_env_runner.py`, py_compile OK).
one-step-per-tick 설계로 아래 훅 3개가 이미 구현됨. **남은 건 로봇에서 실행 검증뿐**
(dev 머신은 rclpy/realsense 없어 import 불가 = 정상). 실측 확인할 **[VERIFY] 5개는
`RESIDUAL_ONLINE_ROBOT_RUNBOOK.md` §1** 에 정리(이미지 스케일 / Δ캡 / 교정 slow주기 /
exec 타이밍 / servo 핸드오프).

구현된 훅(참고):
1. **head hot-swap** (`maybe_hotswap_residual_head`) — head+normalizer(+force_encoder)만, slow 불변.
2. **slow_pred 로깅** — 매 tick live obs 프레임에 그 스텝 base pose9(`slow_pred_target_abs`) 기록
   → one-step-per-tick 이라 obs 와 1:1 정렬 보장.
3. **relabel+전송** — live 프레임 dict → `write_residual_episode_hdf5` → `mailbox.send_episode`.
   relabel 이 virtual(다음스텝 achieved) 과 residual_delta6 를 계산.

### 정렬(구현됨, 참고): relabel 은 `virtual[t]=achieved(t+1)`, 마지막 스텝 버림(T→T-1),
`slow_pred_target_abs[t]`=그 tick 실행 base. actor 가 one-step-per-tick 이라 자동 정렬. task
config `action_target_shift=0`.

### 안전(구현됨): Δ캡 `--residual_translation_cap 0.05 --residual_rotation_cap 0.4`(방향 유지,
크기 상한). 임피던스 stiffness 낮게 + 손 gate-teleop relative 는 기존 manus 설정 유지.

## 5. 검증 루프 (actor 완성 후)
1. smoke(§3) → learner 상시.
2. actor 로 몇 에피소드 교정 → learner loss↓ + head vN 발행 로그 확인.
3. hot-swap 후 정책 변화 관찰. `analysis/modality_attribution` 으로 Δwrench 재측정(v2 대비).
4. 옛/새 위치 성공률 동시 확인(frozen base 라 망각 없어야 함 — full-finetune 대비 핵심 이점).

## 6. v2 확장 포인트
- **손 residual**(pose+hand 13D): shape_meta.action=[13], relabel 에 hand delta 추가,
  `delta6_from_base_to_target` 를 손까지 확장.
- **correction 가중 샘플링**: 실제 민 스텝에 WeightedRandomSampler(cr-dagger DynamicDataset).
- **train_force_encoder=True**: force→Δ 명시로 wrench 병목 완화(hand_online.yaml
  `fast_force_encoder` 블록 토글).
- **RTC**: `pigdm_realtime_chunking` 로 chunk 경계 떨림 완화.
