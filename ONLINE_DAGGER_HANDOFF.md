# CR-DAgger식 Human-Correction 온라인 학습 — 인수인계 문서

> **2026-07-03 업데이트**: 실로봇(오른팔) 코드가 리포에 들어옴. 실로봇 적용 전 마무리해야
> 할 구현(영상 디코드 relabel, actor 오른팔 전환 등)은 **`ONLINE_DAGGER_REALROBOT_PLAN.md`**
> 참고 — 그 문서가 최신 구현 지시서다. 특히 이 문서 §3의 "actor가 왼팔 env 그대로 동작"
> 가정은 더 이상 유효하지 않음(오른팔 env `RightarmRealEnvImp` 사용, 이미지가 zarr가 아닌
> mp4에 저장됨).

이 문서 하나로 지금까지 만든 것 전부와, 나중에 **실제 로봇에 적용할 때 무엇을 가져가고
무엇을 실행하는지**를 정리한다.

기반 체크포인트(base policy):
`data/outputs/logistic_box_unet_abs/checkpoints/epoch=0700-train_loss=0.001.ckpt`
(image-conditioned diffusion policy, obs=image0+robot_pose_L+robot_quat_L, action=9D abs pos+rot6d)

---

## 0. 핵심 아이디어 (한 문단)
Base diffusion policy를 그대로 두고(또는 낮은 LR로 fine-tune), 로봇이 **임피던스(compliant)
모드**로 도는 동안 사람이 팔을 물리적으로 밀어 교정한다. F/T 센서가 없어도, 사람이 민 결과가
**실제 도달한 pose(achieved pose)**에 그대로 나타나므로, "action 라벨 = 한 스텝 뒤 achieved
pose"로 **relabel**하면 사람의 교정이 곧 지도학습 타깃이 된다(= `f/K`가 결과 pose에 녹아있음).
이 correction 데이터를 learner가 학습해 새 가중치를 만들고, actor(로봇)가 에피소드 경계에서
**hot-swap**한다. CR-DAgger의 actor–learner 비동기 구조와 동일하되, residual MLP 대신
full policy fine-tune, robotmq 대신 파일시스템 통신을 쓴다.

---

## 1. 지금까지 만든 것 (파일 인벤토리)

### A. 오프라인 correction 파이프라인 (수동 반복 방식, 가장 안전한 시작점)
| 파일 | 역할 | 상태 |
|------|------|------|
| `rush_eval_real_robot_imp.py` (수정) | 실로봇 inference + correction 수집. `C`(교정 토글)/`S`(유지)/`D`(폐기) 키, `stage` 필드 기록 | 로봇 필요 |
| `data_process/rush_replay_buffer_to_correction_hdf5.py` | replay_buffer.zarr → 학습용 HDF5 (achieved-pose relabel, `--oversample`) | ✅ 문법검증 |
| `data_process/rush_merge_hdf5_datasets.py` | base HDF5 + correction HDF5 병합 | ✅ 문법검증 |
| `correction_retraining_plan.md` | 오프라인 fine-tune Step 1~7 가이드 | 문서 |

### B. Push 시뮬레이터 (학습 없음, 상호작용만)
| 파일 | 역할 | 상태 |
|------|------|------|
| `temp_project_sim_pushtest/sim_push_interactive.py` | 추론 + 화살표 push 시각화 | ✅ 헤드리스 파이프라인 검증 |
| `temp_project_sim_pushtest/README.md` | 사용법 | 문서 |

### C. 온라인 학습 시스템 (actor–learner, hot-swap) — **실로봇 적용 대상**
| 파일 | 역할 | 상태 |
|------|------|------|
| `online_learning/config_online.py` | 공유 설정 (경로/LR/epoch/샘플상한 등, 환경변수 override) | ✅ |
| `online_learning/mailbox.py` | 파일시스템 통신 (에피소드/가중치/status). robotmq 대체 | ✅ 검증 |
| `online_learning/online_learner.py` | Learner: 에피소드 수신 → fine-tune → EMA 가중치 발행 | ✅ 검증 |
| `online_learning/online_actor_env_runner.py` | **Actor(실로봇)**: inference + correction 수집 + hot-swap + 전송 | ⚠️ 로봇에서 미검증 |
| `online_learning/relabel_utils.py` | 에피소드 relabel 헬퍼 | ✅ 검증 |
| `online_learning/smoke_test_no_robot.py` | 로봇 없이 루프 전체 검증 | ✅ 통과 |
| `online_learning/README.md` | 사용법/한계 | 문서 |

### D. 온라인 루프 GUI 데모 (로봇 없이 전 과정 시각화)
| 파일 | 역할 | 상태 |
|------|------|------|
| `temp_project_sim_pushtest/online_loop_demo.py` | 밀기→전송→학습→hot-swap→policy변화 GUI. 백그라운드 learner 자동 실행 | ✅ 플러밍 검증, 실제 학습 성공 확인 |

---

## 2. 검증 상태 (무엇이 확인됐나)
- ✅ **온라인 루프 전 과정**: 에피소드 전송 → learner 학습(loss 감소) → 가중치 v0→v1→...→v4
  발행 → hot-swap(state_dict 완전 일치) → predict_action. GUI 데모에서 실제로 9개 에피소드로
  4+ 라운드 완료, 가중치가 base에서 ‖Δ‖≈11로 변하고 라운드마다 커짐, policy 출력 평균 113mm
  변화 확인.
- ✅ **relabel 정상**: achieved-pose 기준, obs/action 범위 일치, 팔 이동 정상.
- ⚠️ **미검증(하드웨어 필요)**: `online_actor_env_runner.py`의 실제 로봇 실행. 제어 루프는
  검증된 `rush_eval_real_robot_imp.py`와 동일하고, 추가한 훅(hot-swap/전송)만 얹음.

---

## 3. 실제 로봇에 적용하기 — 무엇을 가져가고 무엇을 실행하나

### 3-1. 가져갈 것 (실로봇에 필요한 최소 세트)
```
online_learning/               ← 통째로 (핵심)
rush_eval_real_robot_imp.py    ← 참고용(액터가 이 제어 루프를 그대로 씀)
data_process/rush_replay_buffer_to_correction_hdf5.py  ← 오프라인 병행 시
base 체크포인트(epoch=0700...ckpt)
```
`temp_project_sim_pushtest/`(B, D)는 데모용이라 실로봇엔 불필요.

### 3-2. 사전 준비 (실로봇 고유)
1. **임피던스 모드 + 낮은 stiffness**: 사람이 손으로 밀 수 있을 만큼. (C++ 임피던스 컨트롤러가
   너무 뻣뻣하면 correction 자체가 안 됨. CR-DAgger 기준 병진 ~1000 N/m.)
2. **카메라/로봇이 base 학습과 동일 세팅**: 같은 카메라 위치/해상도여야 policy가 유효.
   (데모와 달리 실로봇은 진짜 이미지가 상태와 연결되므로 policy가 push에 실제로 반응함 — 즉
   실로봇 버전이 데모보다 오히려 더 유효하다.)
3. **사람 손이 카메라 시야 밖**에서 밀기 (시야 안이면 policy가 "손 보이면 이렇게"를 잘못 학습).
4. **페달**: 키보드 에뮬레이션 HID면 `C`(교정 토글)/`S`(유지)/`D`(폐기)에 매핑. (actor는
   OpenCV 창 포커스 상태에서 키 입력을 받음.)

### 3-3. 설정 (`online_learning/config_online.py`)
```python
BASE_CKPT   = ".../epoch=0700-train_loss=0.001.ckpt"   # 실제 base 경로
ONLINE_WORKDIR = ".../data/online_runs/robot1"          # actor·learner 공유 폴더
BASE_DATASET_PATH = ".../..._diffusion_des.hdf5"        # 망각 완화용(선택)
NUM_BASE_DEMOS_TO_MIX = 10        # ★ 실로봇 권장: base 데이터 섞어 과적합/망각 완화
LR = 1e-5                          # fine-tune LR (낮게)
EPOCHS_PER_ROUND = 30
MIN_EPISODES_BEFORE_TRAIN = 3~5   # 처음 몇 개 모은 뒤 학습 시작
MAX_SAMPLES_PER_EPOCH = 128~256   # 라운드 속도 상한 (0=전체)
DEVICE = "cuda:0"
```

### 3-4. 실행
```bash
conda activate robodiff
cd /home/rush/rush_diffusion_policy

# (선택) 먼저 로봇 없이 한번 더 sanity check
python online_learning/smoke_test_no_robot.py

# 터미널 1 — GPU 머신(러너):
python online_learning/online_learner.py

# 터미널 2 — 로봇 PC(액터):
python online_learning/online_actor_env_runner.py
```
- actor/learner가 **같은 머신 또는 공유 파일시스템**이면 그대로 동작(mailbox가 파일 통신).
- **다른 머신 + 공유 FS 없음**이면 `mailbox.py`를 ZMQ(설치돼 있음) 또는 robotmq로 교체해야 함.
  현재는 파일 기반. (가중치 파일 ~350MB/라운드가 오가므로 네트워크면 압축 고려.)

### 3-5. 운영 루프 (사람이 하는 일)
1. actor 창 포커스 → policy가 task 수행하는 걸 지켜봄
2. 실패/어긋나면 페달로 correction ON → 팔을 물리적으로 밀어 교정 → OFF
3. `S`로 에피소드 유지(자동 relabel 후 learner 전송) 또는 `D`로 폐기
4. learner가 학습 후 새 가중치 발행 → **다음 에피소드 시작 시 actor가 hot-swap**
5. 반복하며 policy가 점점 사람 교정을 반영

---

## 4. 실로봇에서 데모와 달라지는 점 / 주의
- **이미지 조건부 한계 사라짐**: 데모는 demo 이미지 재생이라 policy가 push를 못 봤지만,
  실로봇은 실제 카메라라 policy가 교정 결과를 보고 반응 → 진짜 학습 효과 기대 가능.
- **과적합/망각 위험**: 소량 correction만으로 full policy fine-tune 시 다른 상황 성능 저하 가능.
  → `NUM_BASE_DEMOS_TO_MIX`↑, `LR`↓, `EPOCHS_PER_ROUND`↓로 완화. (데모에서 z가 7cm 튄 것도
  이 현상. 실로봇에선 base 데이터 섞기 권장.)
- **갱신 주기**: diffusion fine-tune은 라운드당 수십초~수분. 매 스텝 실시간이 아니라 **에피소드
  단위** 갱신. (더 빠른 갱신이 필요하면 아래 residual 변형.)
- **에피소드 길이**: 너무 길면 라운드가 느려짐. correction은 짧고 명확하게.
- **안전**: 필요 시 relabel/실행 단계에서 pose delta 상한(캡)을 걸어 과도한 교정 방지
  (CR-DAgger의 `scale_and_cap_residual_action`: 병진 5cm, 회전 0.4rad 상당).

---

## 5. 튜닝 파라미터 요약
| 파라미터 | 위치 | 의미 | 실로봇 권장 |
|----------|------|------|-------------|
| `LR` | config_online | fine-tune 강도 | 1e-5 (망각 심하면 ↓) |
| `EPOCHS_PER_ROUND` | config_online | 라운드당 epoch | 20~30 |
| `MAX_SAMPLES_PER_EPOCH` | config_online | 라운드 속도 상한 | 128~256 |
| `NUM_BASE_DEMOS_TO_MIX` | config_online | 망각 완화 | 10~30 |
| `MIN_EPISODES_BEFORE_TRAIN` | config_online | 첫 학습 시점 | 3~5 |
| stiffness | 임피던스 컨트롤러 | 사람이 밀 수 있는 정도 | 낮게 |

---

## 6. 다음 개선 방향
- **Residual policy 변형(진짜 CR-DAgger식, 빠른 온라인)**: base를 동결하고, 작은 MLP가
  (frozen vision feature + base action + 현재 pose) → Δpose를 회귀. 학습이 수초라 갱신이 빠름.
  이 스택의 robomimic VisualCore에서 feature를 빼내는 작업 필요 → 별도 설계.
- **cross-machine 통신**: `mailbox.py`를 ZMQ/robotmq로 교체.
- **안전 캡**: relabel/실행에 pose delta 상한 추가.
- **correction 가중 샘플링**: 실제 push한 프레임에 학습 가중치를 더 주기(현재는 에피소드 단위
  oversample만).

---

## 7. 로봇 없이 지금 다시 보고 싶으면
```bash
python temp_project_sim_pushtest/online_loop_demo.py   # 온라인 루프 GUI 데모
python temp_project_sim_pushtest/sim_push_interactive.py  # push 상호작용만
```
데모 작업폴더 `data/online_runs/gui_demo/`는 실행 시마다 초기화됨.
