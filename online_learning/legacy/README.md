# [LEGACY · 팔 전용] Online DAgger Learning (actor–learner, hot weight-swap)

> ⚠️ **이 폴더는 손(hand) 지원 이전의 "팔만 재학습" 버전 보관본**입니다.
> `online_learning/online_actor_env_runner.py`·`relabel_utils.py`에 `--use_hand`(16D,
> 마누스) 지원을 넣기 전 스냅샷(HEAD 기준)을 그대로 담아 자립형 서브패키지로 만든 것.
> 내부 import는 `online_learning.legacy.*` 로, ROOT는 리포 루트로 재조정되어 있어
> 이 위치에서 그대로 실행된다. 팔 전용(9D) 워크플로가 필요하면 여기 코드를 쓰면 된다.
>
> 실행 (팔 전용):
> ```
> python online_learning/legacy/servo_rightarm_imp_online.py       # 터미널1
> python online_learning/legacy/online_learner.py -i <box_arm_ckpt> # 터미널2
> python online_learning/legacy/online_actor_env_runner.py \        # 터미널3
>     -i <box_arm_ckpt> --steps_per_inference 12 --frequency 10 --num_inference_steps 12
> ```
> 상위 `online_learning/` 의 현재본은 `--use_hand` 로 팔·손 겸용이며, 플래그를 안 주면
> 동일하게 팔 전용으로도 동작한다(하위호환).

---

## (원문) Online DAgger Learning (actor–learner, hot weight-swap)

CR-DAgger의 온라인 구조(actor–learner 비동기 + 실시간 가중치 교체)를 이 스택
(Doosan M0609, ROS2 임피던스, F/T 센서 없음)에 이식한 것.

CR-DAgger와의 차이:
- CR-DAgger는 frozen base 위에 **작은 residual MLP**를 온라인 학습 → 수초 만에 갱신.
- 이 스택엔 residual 인프라가 없어서, **base diffusion policy 자체를 낮은 LR로 online
  fine-tune**하고 EMA 가중치를 hot-swap. 구조는 동일하게 online이지만 한 라운드가
  수초~수분 걸린다(diffusion이 무겁기 때문). 더 빠른 갱신을 원하면 residual 변형 필요(아래).
- F/T 센서가 없으므로 correction 타깃은 **achieved-pose relabel**(사람이 민 결과 pose)로 만든다.

## 구성 요소
| 파일 | 역할 |
|------|------|
| `config_online.py` | actor·learner 공유 설정 (경로, LR, epoch, mailbox 등) |
| `mailbox.py` | 파일시스템 기반 통신 (robotmq 대체). cross-machine은 ZMQ로 교체 |
| `online_learner.py` | Learner 프로세스 (에피소드 수신 → fine-tune → 가중치 발행) |
| `online_actor_env_runner.py` | Actor (실로봇): inference + correction 수집 + hot-swap + 전송 |
| `relabel_utils.py` | replay_buffer 에피소드 → achieved-pose relabel HDF5 |
| `smoke_test_no_robot.py` | 로봇 없이 루프 전체 검증 (통과 확인됨) |

## 데이터 흐름
```
 [Actor: 실로봇]                         [Learner: GPU]
  base policy inference
  ├─ 사람이 페달/‘C’로 correction
  ├─ 'S'로 에피소드 유지
  │     └─ relabel → episode HDF5 ──(mailbox/transitions)──▶ 누적 HDF5에 append
  │                                                          └─ fine-tune (EPOCHS_PER_ROUND)
  └─ 다음 에피소드 시작 시 ◀──(mailbox/weights)── EMA state_dict 발행
       weights 버전 오르면 policy.load_state_dict 로 hot-swap
```

## 실행
```bash
conda activate robodiff
cd /home/rush/rush_diffusion_policy

# 먼저 로봇 없이 검증 (권장):
python online_learning/smoke_test_no_robot.py

# 실제 온라인 운영:
# 터미널 1 (GPU):
python online_learning/online_learner.py
# 터미널 2 (로봇 PC):
python online_learning/online_actor_env_runner.py
```
actor OpenCV 창을 클릭해 포커스를 준 뒤 `C`(correction 토글, 페달 매핑 가능) / `S`(유지) / `D`(폐기).

## 설정 (config_online.py)
- `LR=1e-5`, `EPOCHS_PER_ROUND=30` : fine-tune 강도. 망각 심하면 LR↓ / epoch↓.
- `MIN_EPISODES_BEFORE_TRAIN=2` : 첫 학습 전 최소 에피소드 수.
- `NUM_BASE_DEMOS_TO_MIX` : forgetting 완화용으로 base 데이터 N개를 누적셋에 섞음(0=안 섞음).
  섞으면 라운드마다 그만큼 학습이 무거워짐.
- `SEND_TRANSITIONS=False` : 순수 평가 모드(데이터 안 보냄).

## 검증 상태
- ✅ `smoke_test_no_robot.py` : 실제 `epoch=0700` 체크포인트로 에피소드 전송 → 학습
  (loss 감소) → v1 발행 → hot-swap(state_dict 완전 일치) → predict_action 까지 통과.
- ⚠️ `online_actor_env_runner.py` : 실제 로봇/카메라(LeftarmRealEnvImp)가 있어야 실행
  가능 → 하드웨어 없이는 미검증. 제어 루프는 검증된 `rush_eval_real_robot_imp.py`와
  동일하고, 추가한 훅(hot-swap, 전송)만 얹었다. 첫 실행 시 `stage` 기록·전송·swap 로그를
  꼭 확인할 것.
- 참고: 스모크 테스트 중 tiny 데이터셋 normalizer 계산에서 numpy cast 경고가 뜨지만
  학습/추론에 영향 없음(우리는 base normalizer를 그대로 사용).

## 한계 / 주의
- **갱신 주기**: diffusion fine-tune은 라운드당 수초~수분. 로봇이 도는 중 실시간으로
  매 스텝 갱신되는 게 아니라, "에피소드 단위"로 갱신된다(actor는 에피소드 시작 시 swap).
- **망각(forgetting)**: 소량 correction 데이터에만 fine-tune하면 다른 상황 성능이 나빠질
  수 있음 → `NUM_BASE_DEMOS_TO_MIX`로 base 데이터를 섞거나 LR/epoch를 줄여 완화.
- **가중치 파일 크기**: EMA state_dict ~300MB가 라운드마다 디스크에 써짐(같은 머신 가정).
  cross-machine이면 `mailbox.py`를 ZMQ 전송으로 바꾸고 압축 고려.

## 더 빠른 온라인(residual 변형)으로 가려면
frozen base의 vision feature + base action을 입력받는 작은 MLP가 Δpose를 회귀하도록
학습하고(train 수초), inference에서 `pose_base ⊕ Δpose`로 합성하면 CR-DAgger처럼
빠른 갱신이 된다. 다만 이 스택의 image encoder(robomimic VisualCore)에서 feature를
빼내는 작업이 필요 → 별도 설계 요망.
```
```
