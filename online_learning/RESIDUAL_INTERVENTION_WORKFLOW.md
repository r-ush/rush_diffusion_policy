# Residual Intervention — 실행 워크플로우 (대배치 DAgger, restart-safe)

원본 CR-DAgger 정합(대배치 업데이트 + 시작구간 집중 샘플링) + **learner restart-safe**로,
수집과 학습을 **분리**해서(중노동 없이 나눠서) 돌릴 수 있다.

---

## TL;DR
- **actor와 learner는 디스크(`transitions/`)로 통신하는 별개 프로세스** → 동시에 안 붙어 있어도 됨.
- **50개를 한 번에 안 모아도 됨.** 여러 번 나눠 쌓아도 파일은 그대로 남는다.
- learner는 시작할 때 **`transitions/`의 모든 에피소드로 버퍼를 복원**(`.done` 포함)하므로,
  **껐다 켜도 이전 데이터를 잃지 않는다.** 재시작해도 기존(학습된) head는 유지된다.

---

## 설정값 (원본 논문 기준, env로 조절)
| 항목 | 기본 | env | 의미 |
|---|---|---|---|
| 첫 학습 임계 | **50** | `RESIDUAL_INTERVENTION_FIRST_TRAIN_EPISODES` | 이만큼 모여야 첫 학습(논문 `num_episodes_before_first_training`) |
| 갱신 주기 | **10** | `RESIDUAL_INTERVENTION_UPDATE_EVERY_N` | learner를 계속 켜둔 경우, 이만큼 새로 쌓일 때마다 갱신 |
| 개입 가중 | **5.0** | `INTERVENTION_SAMPLE_WEIGHT` | 개입 프레임 오버샘플 배수 |
| 시작구간 | **8** | `INTERVENTION_CORRECTION_START_HORIZON` | 개입 onset 이후 이 스텝만 가중(dense-after). 0=개입 전체 균등 |

> ⚠️ 손 task는 50이 클 수 있음 → 예: `FIRST_TRAIN_EPISODES=20`, `UPDATE_EVERY_N=5`로 낮춰 시작.

---

## (B) 분리 워크플로우 — 수집 먼저, 학습 나중 (권장, 편함)

### 흐름
```
[반복]
 1. actor 만 켜서 교정 수집   → transitions/ep_*.hdf5 (디스크에 쌓임, 여러 번 나눠도 OK)
 2. 목표 개수 모이면 learner 실행 → 디스크 전체 복원 → 게이트 충족 시 즉시 학습 → head 발행 → 종료
 3. actor 가 새 head hot-swap (또는 actor 재시작)
```

### 1. 수집 (actor만)
learner 없이 actor만 실행. head가 아직 없으면 **slow-only로 돌며 교정만 수집**(정상).
```bash
# (로봇 머신) 개입 actor. 페달/키: a=개입시작(밀기) b=개입종료 s=유지+전송 d=폐기 q=종료
online_learning/launch_intervention/2_actor.sh
```
- 팔을 밀어 교정할 구간에 `a`(시작)~`b`(종료), 좋으면 `s`로 전송. 그 에피소드가 `transitions/`에 쌓임.
- **쉬었다 다시 해도 됨.** actor를 껐다 켜도 이전 `transitions/` 파일은 그대로.
- 첫 프레임에서 [VERIFY-1] 이미지 스케일(0~255 uint8) 한 번 확인.

### 2. 학습 (모이면 learner 한 번)
```bash
# FIRST_TRAIN 을 손 task에 맞게 조절해서 실행 (예: 20)
RESIDUAL_INTERVENTION_FIRST_TRAIN_EPISODES=20 \
online_learning/launch_intervention/1_learner.sh
```
learner가 시작하면서:
- `transitions/`의 **모든** 에피소드를 버퍼로 복원(restart-safe) → `num_demos`=총개수.
- `num_demos ≥ FIRST_TRAIN`이면 **즉시 학습**(FIRST_EPOCHS, ~수 분) → `weights/weights_vN.pt` 발행.
- 안 되면 부족분만큼 더 모으라고 대기 로그.
- 학습 끝나면 Ctrl-C로 learner 종료해도 됨(발행된 head는 `weights/`에 남음).

### 3. 사용 + 다음 배치
- actor가 켜져 있으면 자동 hot-swap. 꺼져 있었으면 `2_actor.sh` 다시 실행 → 최신 head 로드.
- 다음 배치: **1~2 반복**. learner를 다시 켜면 `transitions/` 전체(이전+신규)를 복원해 재학습하므로
  이전 데이터를 잃지 않는다. (기존 발행 head는 새 학습 완료 전까지 유지 → actor 퇴보 없음.)

---

## (A) 온라인 워크플로우 — 둘 다 켜둠 (논문 방식)
learner+actor 동시 실행. 첫 `FIRST_TRAIN`개 동안 learner는 **누적만**(학습 X, 부하 ≈0), 도달 시 1회 학습,
이후 `UPDATE_EVERY_N`개마다 warm-continue 갱신·발행·hot-swap.
```bash
online_learning/launch_intervention/1_learner.sh   # 터미널1 (계속 켜둠)
online_learning/launch_intervention/2_actor.sh     # 터미널2
```
> ⚠️ (A)에서는 **learner를 세션 중간에 껐다 켜지 말 것**은 이제 완화됨(restart-safe). 다만 껐다 켜면
> head가 새로 초기화되어 재학습되므로, 굳이 필요 없으면 계속 켜두는 편이 warm-continue에 유리.

---

## 동작 근거 (왜 되는가)
- **디스크 큐**: `mailbox.send_episode`가 `transitions/ep_N.hdf5`+`.ready`를 남김 → 영속. actor/learner 비동기.
- **restart-safe**: `learner._rebuild_from_disk()`가 시작 시 `mailbox.list_all_episodes()`(`.ready`/`.done` 무관)로
  `accumulated`+`num_demos` 재구성 → 재시작해도 버퍼 유지.
- **head 유지**: 재시작 시 기존 `weights/latest.txt` 있으면 빈 v0 재발행 안 하고 버전만 이어받음 → 재학습 전까지 기존 head 사용.
- **대배치 게이트**: `_should_train()` — 첫 `FIRST_TRAIN`, 이후 `UPDATE_EVERY_N`. 논문: 소배치 잦은 갱신은 불안정.
- **시작구간 집중**: `intervention_sample_weights(start_horizon)` — 개입 onset 이후 `CORRECTION_START_HORIZON`
  스텝만 가중(시작 직전 실패징후 제외). 논문 dense-after.

## 분석 (수집한 데이터로)
```bash
BAE=/home/rush/anaconda3/envs/bae_robodiff/bin/python
# logged 재생(의존성 없음): 밀림/nominal 분할 + 3D 재생 HTML
$BAE online_learning/export_traj3d_playback_html.py \
  --episodes data/online_runs/run_hand_intervention/transitions \
  --head none --world_rot_x_deg 135 --out data/verify_intervention
```
(head 오버레이는 `--head <weights_vN.pt>` + `RESIDUAL_CONFIG_NAME=residual_policy/hand_intervention_mlp`
+ 동일 base 필요. → `RESIDUAL_ONLINE_NEXT.md` / playback 도구 참고.)

## 관련 파일
- learner/actor: `residual_intervention_learner.py`, `residual_intervention_actor_env_runner.py`,
  공유 `residual_teleop_learner.py`(게이팅·restart-safe), `mailbox.py`(`list_all_episodes`)
- config: `config_residual_intervention.py`, task `hand_intervention[_mlp].yaml`
- 런처: `launch_intervention/{1_learner,2_actor}.sh`
- 검증: `smoke_test_residual_intervention_no_robot.py`
