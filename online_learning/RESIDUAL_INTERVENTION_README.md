# Residual-Online DAgger — INTERVENTION (physical-push) 판

> frozen slow base + tiny residual head 를 **온라인**으로 학습하는 cr-dagger식 경로.
> teleop 판(`residual_teleop_*`)과 학습 코어는 같고 **개입 방식만 다르다**.

## teleop 판 vs intervention 판

|  | teleop 판 (`residual_teleop_*`) | **intervention 판 (`residual_intervention_*`)** |
|---|---|---|
| 개입 시 팔 | 페달로 servo 에 넘김 → 사람이 VR 트래커로 몬다 | 넘기지 않음 → base 가 임피던스로 계속 실행, 사람이 **물리적으로 민다** |
| 개입 시 손 | manus 글러브로 사람이 몬다 | **항상 base(slow) 자율** (사람 안 건드림) |
| 페달 역할 | 제어 핸드오프(servo/manus on) | **label 만** — 어느 프레임이 개입인지 표시 |
| 필요한 외부 노드 | servo + manus (+gate) | **없음** (learner + actor 2 터미널) |
| relabel 수식 | `residual = achieved(t+1) − slow_pred(t)` | **동일** |
| learner | 균등 샘플 | 개입 프레임 **가중 샘플**(`INTERVENTION_SAMPLE_WEIGHT`) |

핵심: 두 판 모두 achieved pose 로 residual 을 만든다. 개입형은 제어를 안 넘기고 사람이 임피던스
팔을 밀어 achieved 를 바꾼다. base 가 매 tick forward 를 이미 명령하므로 **안 밀면 residual≈0,
밀면 residual=밀림** 으로 자연히 분리된다. 이 정렬을 위해 **개입 중에는 매 tick slow 를 새로
예측**한다(actor 가 자동으로 함).

## 파일

| 파일 | 역할 |
|------|------|
| `config_residual_intervention.py` | 공유 설정(WORKDIR/slow_ckpt/config_name/가중치). teleop config 와 속성명 동일 |
| `residual_intervention_learner.py` | learner: `residual_teleop_learner` 상속 + 개입 프레임 가중 `WeightedRandomSampler` |
| `residual_intervention_actor_env_runner.py` | actor: 항상 base 실행(팔+손) + 페달 토글 label + fresh-slow + relabel/전송 |
| `residual_relabel_utils.py` (공유) | episode 에 `is_intervention` 있으면 통과 → `obs/is_intervention` |
| `diffusion_policy/config/residual_policy/task/hand_intervention.yaml` | hand_online + `intervention_key: obs/is_intervention` |
| `diffusion_policy/config/residual_policy/hand_intervention_mlp.yaml` | top-level(위 task 바인딩) |
| `smoke_test_residual_intervention_no_robot.py` | Part 0(배선, ckpt 불필요) + A/B/C(ckpt 필요) |
| `launch_intervention/{1_learner,2_actor}.sh` | 2 터미널 실행 |

`step_dataset.py` 의 `FastResidualContextStepDataset` 에 `intervention_key` 파라미터가 추가됐다
(있을 때만 replay_buffer 에 `is_intervention` 적재; policy 입력·정규화엔 영향 없음).

## 실행

```bash
# (권장) 먼저 로봇 없이 배선/학습 검증
/home/rush/anaconda3/envs/bae_robodiff/bin/python \
  online_learning/smoke_test_residual_intervention_no_robot.py

# 실운영 — 2 터미널
bash online_learning/launch_intervention/1_learner.sh   # 터미널1 (GPU)
bash online_learning/launch_intervention/2_actor.sh     # 터미널2 (로봇 PC)
```

actor 터미널에 포커스를 준 뒤: **a**=개입 시작(밀기) / **b**=개입 종료 / **s**=유지+전송 /
**d**=폐기 / **q**=종료. learner 가 head 를 발행하기 전(교정 `MIN_EPISODES` 미만)에는
slow-only 로 돌며 교정만 모은다(정상).

## 튜닝 (env override)

- `INTERVENTION_SAMPLE_WEIGHT`(기본 5.0): 개입 프레임 학습 가중. 교정이 잘 안 배면 ↑.
- `--residual_translation_cap`(0.05 m) / `--residual_rotation_cap`(0.4 rad): head 출력 캡
  (사람 밀림은 물리라 이 캡과 무관).
- `RESIDUAL_INTERVENTION_FIRST_EPOCHS`(120) / `_EPOCHS_PER_ROUND`(40) / `_LR`(1e-4).

## 로봇에서 실측할 [VERIFY] (하드웨어 없이는 검증 불가)

1. **임피던스 stiffness**: 사람이 손으로 팔을 밀어 눈에 띄게 움직일 만큼 낮은지. 너무 높으면
   교정 신호(residual)가 작아지고, 너무 낮으면 base 추종이 나빠진다.
2. **이미지 스케일** (`_to_uint8_image`): env(obs_float32=True) 이미지가 0~1 인지 0~255 인지
   실측. relabel 데이터셋은 uint8 HWC 기대.
3. **개입 중 base 전진 vs 밀림**: 개입 시 팔이 계속 forward 하며 사람이 밀림을 얹는 구조가
   맞는지 체감 확인(밀 때만 residual 이 남아야 함). 손이 개입 중에도 base 로 잘 도는지.
4. **exec 타이밍**: `--frequency` / `--steps_per_inference` 에서 제어가 끊기지 않는지.
5. **Δ 캡**: head 발산 시 캡이 안전하게 거는지(초기 라운드).

## v2

- 손 residual(pose+hand 13D), correction 세부 가중 스케줄, RTC(chunk 경계 떨림 완화),
  `train_force_encoder=True`(force→Δ). teleop 판 `RESIDUAL_ONLINE_HANDOFF.md` §6 참고.

---

## 변경 기록 (2026-07-22 구현)

### 1. 기존 두 경로를 `_teleop` 으로 리네임 (git mv, 히스토리 보존)
| 이전 | 이후 |
|---|---|
| `online_learner.py` | `finetune_teleop_learner.py` |
| `online_actor_env_runner.py` | `finetune_teleop_actor_env_runner.py` |
| `residual_online_learner.py` | `residual_teleop_learner.py` |
| `residual_online_actor_env_runner.py` | `residual_teleop_actor_env_runner.py` |

- 클래스명(`OnlineLearner`, `ResidualOnlineLearner`)·config 모듈명·env-var 는 **유지**.
- 모든 import / 런처(`run_*.sh`, `launch_abs/2·4.sh`) / 라이브 문서(README·QUICKSTART·RUNBOOK·HANDOFF) 갱신.
- 과거 스냅샷 문서(`SESSION_SUMMARY_*`, `ONLINE_DAGGER_*PLAN/HANDOFF`)는 시점 기록이라 보존.

### 2. 새 `residual_intervention` 경로 추가 (위 "파일" 표)
- actor: 핸드오프 제거, 팔·손 항상 base 실행, 페달 토글 label + fresh-slow, `is_intervention` 기록·전송.
- learner: `ResidualOnlineLearner` 상속 + `_make_sampler` override(개입 프레임 `WeightedRandomSampler`).
- config/task: teleop 와 분리(`run_hand_intervention`, `hand_intervention[_mlp].yaml`).

### 3. 공유 유틸 확장 (backward-compatible — teleop 경로 동작 불변)
- `residual_teleop_learner.py`: `__init__(cfg=None)`+`self.C`+`_make_sampler` 훅으로 리팩터.
- `residual_relabel_utils.py`: episode 에 `is_intervention` 있으면 통과.
- `step_dataset.py`: `FastResidualContextStepDataset(intervention_key=None)` — 있을 때만
  replay_buffer 에 실림(정책 입력·정규화 불변). 가중은 sample 윈도우 현재스텝 `is_intervention[buf_end-1]`.

### 4. 검증 (bae_robodiff, 2026-07-22)
- `smoke_test_residual_intervention_no_robot.py` **전부 통과**: Part0 배선(relabel→dataset(intervention_key)→
  가중 24/69 정확) + A 인스턴스화 + B 가중학습(loss 0.218→0.049) + C hot-swap(Δ=0).
- 회귀: `smoke_test_residual_no_robot.py`(teleop) 통과(self.C 리팩터 무해). 전 파일 py_compile OK.
- **actor 실로봇 제어 루프는 하드웨어 필요 → 미실행**. 위 [VERIFY] 5개를 로봇에서 확인할 것.
