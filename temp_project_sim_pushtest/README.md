# [TEMP] Interactive Push Simulation

실제 로봇 없이, 학습된 diffusion policy로 실시간 추론을 돌리면서
키보드 화살표로 "사람이 로봇을 미는 것"을 흉내내보는 임시 시뮬레이터.
**마음에 안 들면 이 폴더(`temp_project_sim_pushtest/`)만 통째로 지우면 됨.**

이 폴더엔 두 개의 스크립트가 있음:
- `sim_push_interactive.py` : 추론 + push 시각화만 (학습 없음). 아래 대부분의 설명이 이것.
- `online_loop_demo.py`     : **로봇 없이 온라인 학습 루프 전체를 GUI로 보는 데모**
  (밀기 → correction 전송 → 백그라운드 learner 학습 → 가중치 hot-swap → policy 변화).
  맨 아래 "온라인 루프 GUI 데모" 절 참고.

## 실행
```bash
conda activate robodiff
cd /home/rush/rush_diffusion_policy
python temp_project_sim_pushtest/sim_push_interactive.py \
    --ckpt data/outputs/logistic_box_unet_abs/checkpoints/epoch=0700-train_loss=0.001.ckpt \
    --dataset /home/rush/Desktop/Datasets/20260630_195919_diffusion_des.hdf5 \
    --demo 0
```
창을 클릭해 포커스를 준 뒤 화살표로 조작.

## 조작
| 키 | 동작 |
|----|------|
| ↑ | x+ (앞으로) 미는 것처럼 |
| ↓ | x- |
| ← | y- |
| → | y+ |
| W / S | z+ / z- |
| R | demo 시작 위치로 리셋 |
| SPACE | 궤적 trace 지우기 |
| ESC / Q | 종료 |

화면: 왼쪽 = policy에 들어가는 입력 이미지(demo 재생), 오른쪽 = top-down x-y 뷰
(초록=policy target, 파랑=실제 EE, 빨강 화살표=push 방향).

## ⚠️ 이 시뮬레이터의 근본적 한계 (꼭 읽을 것)
이 policy는 **이미지 조건부**다. 시뮬레이션 EE 상태에 맞는 렌더링 화면이 없으므로,
여기서는 **학습 데이터셋 demo의 카메라 이미지를 그대로 재생**해서 policy 입력으로 넣는다.

결과적으로:
- policy는 demo 궤적을 따라가려 하고, 화살표 push는 그 위에서 **실제 EE 위치만** 밀어낸다.
- policy는 push를 **보지 못한다** (이미지가 고정 재생이므로). 즉 push했다 놓으면
  policy가 다시 자기 target으로 끌어당기는 모습(=compliant 상황에서 사람이 밀었다 놓기)은
  보이지만, "사람 개입에 policy가 시각적으로 반응"하는 것까지는 재현되지 않는다.

따라서 이건 **correction 루프의 상호작용 메커니즘을 눈으로 보는 데모**이지,
닫힌 루프 task 성공을 재현하는 물리 시뮬레이션이 아니다.

## 파라미터 튜닝
- `--follow_gain` (기본 0.35): policy target으로 EE가 끌려가는 속도. 크면 뻣뻣, 작으면 물렁.
- `--push_gain` (기본 0.008): 화살표 한 tick당 밀리는 거리[m]. push가 약하면 키워라.
- `--img_advance` (기본 1): tick당 재생 이미지 전진량. demo가 빨리/느리게 흐름.
- `--demo N`: 다른 demo(0~99)의 이미지/시작pose 사용.

## 만약 "진짜 닫힌 루프"가 필요하면
이 policy를 시뮬레이션에서 제대로 돌리려면 EE/물체 상태로부터 학습 분포와 비슷한
카메라 이미지를 **렌더링**하는 시뮬레이터(MuJoCo/Isaac 등에 동일 카메라 세팅)가 필요하다.
그건 이 temp 스크립트의 범위를 크게 벗어나므로 별도 작업으로 논의.

---

## 온라인 루프 GUI 데모 (`online_loop_demo.py`)

로봇 없이 **온라인 학습 루프 전체를 한 화면에서** 지켜보는 데모.

```bash
conda activate robodiff
cd /home/rush/rush_diffusion_policy
python temp_project_sim_pushtest/online_loop_demo.py
```
실행하면 자동으로 백그라운드 learner 프로세스가 함께 뜬다(로딩 10~20초). 창을 클릭해
포커스를 준 뒤 조작:

| 키 | 동작 |
|----|------|
| ↑x+ ↓x- ←y- →y+ / W,S:z | 로봇 밀기 (누르는 동안 correction 으로 기록됨) |
| ENTER | 지금까지 기록한 구간을 correction 에피소드로 learner에 전송 (최소 25스텝) |
| BACKSPACE | 기록 버퍼 비우기(전송 안 함) |
| R / SPACE / ESC(Q) | pose 리셋 / 궤적 지우기 / 종료(learner도 함께 종료) |

화면 하단 패널에서 흐름을 볼 수 있음:
- `[ACTOR/GUI]` : 현재 policy 버전, 전송한 에피소드 수, 기록버퍼 길이, 추론 지연
- `[LEARNER]` : 대기/수신/**학습중(epoch·loss)**/발행완료, 누적 demo 수, 최신 가중치 버전
- 새 가중치가 나오면 `⚡ policy SWAPPED -> vN` 플래시가 뜨고 ACTOR policy 버전이 올라감

동작 순서 예: 화살표로 좀 민다 → ENTER(전송) → LEARNER가 "학습중"으로 바뀌고 loss가 도는
게 보임 → 몇 초 뒤 "발행완료 vN" → 화면에 SWAPPED 플래시 → policy=vN 으로 갱신.

### 데모 설정 (필요시 환경변수로 조절)
`online_loop_demo.py` 상단에서 learner를 가볍게 돌리도록 자동 설정함
(`ONLINE_EPOCHS_PER_ROUND=12`, `ONLINE_MIN_EPISODES=1`, `ONLINE_BATCH_SIZE=8`).
작업폴더는 `data/online_runs/gui_demo/` (실행 시마다 초기화). learner 로그는 그 안
`learner.log`.

### 한계 (중요)
`sim_push_interactive.py`와 동일하게 **이미지 조건부 한계**가 있다. 여기서 확실히 보이는 건
"밀기 → 데이터 전송 → 학습 → 가중치 버전 상승 → policy hot-swap"이라는 **온라인 루프의
흐름**이다. 학습 후 policy 출력이 "민 쪽으로 확실히 이동"하는 건(적은 데이터 + 이미지
고정재생 때문에) 보장되지 않는다. task 학습 성공 재현이 아니라 파이프라인 시각화 데모다.

### 검증 상태
- ✅ 서브프로세스 learner 실행 → v0 발행/`idle` → 에피소드 전송 → 상태전이
  (received→training(epoch,loss)→published) → 가중치 v0→v1 상승 → hot-swap 로드까지
  로봇/디스플레이 없이 헤드리스로 검증됨.
- ⚠️ pygame 창 렌더링 자체는 디스플레이가 있는 환경에서 직접 실행해 확인 필요
  (코드 구조는 검증된 `sim_push_interactive.py`와 동일).
- GPU: GUI(policy 추론) + learner(model+ema) 프로세스가 각각 GPU를 쓴다. 작은 GPU면
  메모리 부족 가능 → learner를 CPU로 돌리려면 `ONLINE_DEVICE=cpu`.
