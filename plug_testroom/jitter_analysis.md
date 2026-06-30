# 실제 로봇 Jitter 원인 분석

## 핵심 요약

`desired_pose`가 이동 방향 기준으로 **현재 위치보다 뒤(과거)**에 생성되는 현상의 원인은 
**3개의 레이어로 중첩**되어 있습니다.

---

## 데이터 분석 결과 (impedance_control_data_vr_20260519_160627.csv)

| 항목 | 값 |
|------|-----|
| 총 샘플 수 | 15,574 |
| 데이터 주파수 | 100 Hz |
| 총 기간 | ~155초 |
| Action 시작 시점 | ~20.5초 |
| desired_pose가 current보다 **앞에** 있는 비율 | **64.6%** |
| desired_pose가 current보다 **뒤에** 있는 비율 | **35.4%** (→ jitter 원인) |
| 뒤처진 구간의 평균 오차 | 6.23 mm |
| 앞서가는 구간의 평균 앞섬 | 11.30 mm |

---

## 원인 1 (가장 심각): `action_timestamps` 시작점 오류

### 문제 코드 (line 208~212)
```python
action_timestamps = (
    np.arange(len(action), dtype=np.float64)
) * dt + obs_timestamps[-1]
```

### 문제 설명

```
t=0:    obs 수집 완료 (obs_timestamps[-1])
t=0:    action[0] timestamp = obs[-1] + 0*dt = t=0  ← 이미 현재 시각!
t=0.1s: action[1] timestamp = obs[-1] + 1*dt = t=0.1s
...
t=~150ms: 추론 완료 (DDIM 12 steps latency)

→ curr_time = obs[-1] + 0.15s
→ is_new = action_timestamps > (curr_time + 0.01)
→ action[0] (t=0.0s) → 탈락 (이미 지남)
→ action[1] (t=0.1s) → 탈락 (0.15+0.01 < 0.1이므로 탈락)
→ action[2]부터 실행
```

**`action[2]`의 의미: obs[-1] 기준 +200ms 후의 목표 포즈**

그런데 policy가 학습에서 `action[0]`을 **"현재 obs 시점과 동시인 포즈"**로 학습했기 때문에,
action[2]는 "200ms 후의 미래 포즈"를 예측한 값이 맞습니다.

> [!IMPORTANT]
> 문제는 이 자체가 아닙니다. 하지만 추론 latency가 불규칙하거나 길어질 때
> `is_new` 필터가 제거하는 인덱스가 달라지면서 **일관성 없는 action 시작점**이 발생합니다.

---

## 원인 2 (구조적): `n_action_steps` 오버라이드

### 문제 코드 (line 115)
```python
policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
```

### 문제 설명

```
horizon = 16
n_obs_steps = 2
→ n_action_steps = 16 - 2 + 1 = 15

steps_per_inference = 12
```

Policy는 15개의 action을 생성하지만, 실행에는 `steps_per_inference=12`개만 사용합니다.

**하지만 이것이 직접적인 jitter 원인은 아닙니다.**
진짜 문제는 `n_action_steps`를 이렇게 직접 오버라이드하면 
**학습 시 설계된 action window 의미**와 달라질 수 있다는 점입니다.

---

## 원인 3 (핵심): Policy가 현재 위치를 중심으로 oscillate

### 실제 데이터에서 확인된 패턴

```
t=27.2s: curr_y=22.8,  desired_y=3.5    → diff = -19.3 (정상: desired가 앞)
t=28.0s: curr_y=-14.7, desired_y=-28.6  → diff = -13.8 (정상: desired가 앞)
t=32.0s: curr_y=-120.9, desired_y=-132.5 → diff = -11.5 (정상: desired가 앞)
t=37.4s: curr_y=-134.1, desired_y=-134.1 → diff = +0.1 (역전! desired가 뒤)
```

로봇이 빠르게 이동하는 구간에서는 desired가 앞에 있으나 (정상),
**방향 전환 지점이나 감속 구간**에서 desired가 current보다 뒤로 역전됩니다.

### 가장 의심되는 근본 원인: **절대 좌표 action + 학습 데이터 특성**

Action이 **절대 좌표(absolute pose)**로 학습되어 있는데, policy가 obs 시점 기준으로
"이미 지나온 경로"를 action으로 예측하는 경우 발생합니다.

이는 다음 두 가지 경우에 발생합니다:

**Case A. Train/Test Distribution mismatch**
- 학습 데이터에서의 로봇 이동 속도 vs. 실제 실행 시 이동 속도가 다름
- Policy가 학습 데이터 속도보다 느리게 실행되면 action이 항상 앞에 있어야 하지만,
  실제 로봇이 더 빠르게 움직이면 action이 뒤처짐

**Case B. Action horizon의 절대 좌표 문제**
```
학습:  obs[T-1], obs[T] → predict: action[T], action[T+1], ..., action[T+15]
실행:  obs 수집 후 추론 (latency ~150ms)
      → action[T]은 이미 지난 위치
      → action[T+2]부터 실행
      → action[T+2]가 실제 현재 위치보다 뒤에 있으면 → 뒤로 당김 → jitter
```

---

## 원인 4 (데이터 특성): Action Block 길이 분포

```
action block 길이 분포:
  1 샘플 (10ms): 11,351개  ← 99%
  2 샘플 (20ms): 71개
```

**이론적으로**: `steps_per_inference=12`, `dt=0.1s` → 각 action이 100ms 동안 유지되어야 함

**실제**: desired_pose가 **100Hz(10ms)마다 매번 바뀌고 있음**

이는 `LeftarmRealEnv`가 전달받은 action sequence를 100Hz로 보간/전달하고 있어서,
action[0]~action[11]을 10ms 간격으로 순서대로 내보내는 구조입니다.

즉, steps_per_inference=12의 12개 action이 **120ms** 동안 소비됩니다 (1.2초가 아니라).
→ **실제 control frequency가 설정과 다름**

---

## 진단 정리

```
[문제]  desired_pose가 현재 위치보다 뒤에서 생성됨 → jitter

[원인1] action_timestamps 시작점이 obs[-1] (현재 시각)
        → 추론 latency(~150ms) 동안 action[0~1]은 이미 과거가 됨
        → is_new 필터로 앞쪽 action 제거 → 시작점이 불규칙해짐

[원인2] n_action_steps = 15로 오버라이드
        → Policy가 15개 action 생성, 앞부분 2개 버려지고 나머지 12개 사용
        → action[2]부터 시작 = obs[-1] + 0.2s 후 목표

[원인3] LeftarmRealEnv가 action을 100Hz로 내보냄
        → steps_per_inference=12 action이 1.2s 아닌 120ms만에 소비됨
        → inference cycle (steps_per_inference * dt = 1.2s)과 불일치
        → 다음 inference 전까지 ~1.08s 동안 동일 action 반복 or idle

[원인4] Policy의 절대좌표 action이 현재보다 뒤를 가리키는 경우
        → 방향 전환 구간, 감속 구간에서 발생
        → 이 경우 로봇이 실제로 뒤로 움직이려 함 → oscillation
```

---

## 권장 해결책

### 1. `action_timestamps` offset 보정

```python
# 현재 (문제)
action_timestamps = np.arange(len(action)) * dt + obs_timestamps[-1]

# 수정: 추론 latency를 고려한 offset 추가
inf_latency_est = time.time() - t_inf  # 실제 latency
action_timestamps = np.arange(len(action)) * dt + obs_timestamps[-1] + inf_latency_est
```

또는 더 단순하게: **항상 action[n_obs_steps]부터 시작**하도록 고정 offset 부여

```python
# 예: n_obs_steps=2 → action[2]부터 항상 실행 (is_new 필터 불필요)
action_offset = n_obs_steps  # 2
this_target_poses = action[action_offset:action_offset + steps_per_inference]
```

### 2. `steps_per_inference`를 실제 실행 주기에 맞게 조정

현재 100Hz 환경에서 `steps_per_inference=12`, `dt=0.1s`이면:
- 이론: 12 * 0.1s = 1.2s마다 inference
- 실제: LeftarmRealEnv가 100Hz로 action 전송 → 12개가 120ms 안에 소비

→ `steps_per_inference`를 **120** (= 1.2s * 100Hz)으로 설정하거나,
   LeftarmRealEnv의 action 전송 방식을 확인해야 함

### 3. `n_action_steps` 오버라이드 제거

```python
# 현재 (비표준)
policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

# 수정: 학습 시 설정된 값 그대로 사용
# policy.n_action_steps는 건드리지 않음
```

### 4. 학습 데이터 재확인

- `epoch=0900, train_loss=0.000` → **train loss가 0으로 과적합 가능성**
- 더 낮은 epoch 체크포인트 (예: epoch=0600~0700)로 테스트 권장
- Validation loss 곡선 확인 필요

---

## 타임라인 시각화

```
t=0      t=0.1    t=0.2    t=0.3  ...  t=1.2    t=1.3  ...
|--------|--------|--------|-----------|---------|---------|
 obs[-2]  obs[-1]  act[0]   act[1]      act[10]  act[11]   (학습 기준)

t=0: obs 수집 완료
t=150ms: 추론 완료
  → action[0](t=0.0), action[1](t=0.1) 는 이미 지남 → 버려짐
  → action[2](t=0.2s) 부터 실행
  
그런데 action[2]의 절대 좌표값이 현재 robot 위치보다 뒤에 있으면:
  → 로봇이 뒤로 이동하려 함 → jitter!
```
