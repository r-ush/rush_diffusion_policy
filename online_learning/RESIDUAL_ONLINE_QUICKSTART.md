# Residual-Online DAgger — 이 머신(vision) 기동 퀵스타트

> 로봇 앞에서 **뭐 켜야 하는지**만 정리. 설계는 `RESIDUAL_ONLINE_HANDOFF.md`,
> 상세 절차/튜닝은 `RESIDUAL_ONLINE_ROBOT_RUNBOOK.md`.
> 모든 명령은 export 없이 **한 줄 인라인**(변수를 명령 앞에 붙임). 붙여넣기만 하면 됨.

## ⚠️ 환경 (제일 중요)
- **learner·actor 는 `venv_diffusion` 만** — residual policy 가 `timm`(vision encoder) 필요.
- **`robodiff_rush`(conda) 쓰지 말 것** — timm 없음 → import 단계에서 죽음.
- servo·manus 는 timm 불필요(rclpy 만). manus 는 평소 ROS2 python 그대로.
- 고정 경로:
  - slow ckpt: `/home/vision/diffusion-policy/data/outputs/260714_insert_box_hand_rel/epoch=0500-train_loss=0.002.ckpt`
  - workdir: `/home/vision/rush_diffusion_policy/data/online_runs/run_hand_residual`
  - python: `/home/vision/venv_diffusion/bin/python`

## 켜는 순서
0. (한 번) workdir 비우기 → 1 servo → 2 learner → 3 manus → 4 actor
   (learner 를 actor 보다 먼저 — v0 head 를 미리 발행해 actor 가 즉시 hot-swap)

### 0. (최초 1회) workdir 초기화 — 낡은 폴더 재사용 금지
```bash
rm -rf /home/vision/rush_diffusion_policy/data/online_runs/run_hand_residual
```

### 터미널 1 — servo (팔 servo, VR Vive + teleop=1). RESIDUAL env 불필요
```bash
cd /home/vision/rush_diffusion_policy && /home/vision/venv_diffusion/bin/python online_learning/servo_rightarm_imp_online.py
```

### 터미널 2 — learner (head warm-continue 학습 + 발행)
```bash
cd /home/vision/rush_diffusion_policy && RESIDUAL_SLOW_CKPT=/home/vision/diffusion-policy/data/outputs/260714_insert_box_hand_rel/epoch=0500-train_loss=0.002.ckpt RESIDUAL_ONLINE_WORKDIR=/home/vision/rush_diffusion_policy/data/online_runs/run_hand_residual /home/vision/venv_diffusion/bin/python online_learning/residual_online_learner.py
```

### 터미널 3 — manus (손 교정, 교정 중에만 발행). 평소 ROS2 python
```bash
cd /home/vision/manus_ws/src/ROS2 && python manus_to_aidin_rush.py --gate-teleop -r --relative
```

### 터미널 4 — actor (slow+fast 추론 + 교정 수집 + hot-swap + 전송)
```bash
cd /home/vision/rush_diffusion_policy && RESIDUAL_SLOW_CKPT=/home/vision/diffusion-policy/data/outputs/260714_insert_box_hand_rel/epoch=0500-train_loss=0.002.ckpt RESIDUAL_ONLINE_WORKDIR=/home/vision/rush_diffusion_policy/data/online_runs/run_hand_residual /home/vision/venv_diffusion/bin/python online_learning/residual_online_actor_env_runner.py -i /home/vision/diffusion-policy/data/outputs/260714_insert_box_hand_rel/epoch=0500-train_loss=0.002.ckpt --use_hand --steps_per_inference 6 --frequency 10 --num_inference_steps 12
```

## actor 키 (터미널 4에 포커스)
`a`=servo 핸드오프(교정 시작) · `b`=정책 복귀 · `c`=홈 · `s`=에피소드 유지+전송 · `d`=폐기 · `q`=종료

## 운영 루프
1. actor 가 slow+residual 로 task 수행하는 걸 지켜봄.
2. 어긋나면 `a` → 팔을 물리적으로 밀어 교정(+필요시 manus 로 손) → `b` 로 복귀.
3. `s` 로 에피소드 유지(자동 relabel 후 learner 전송) 또는 `d` 로 폐기.
4. learner 가 head 갱신·발행(라운드 수초) → 다음 에피소드 시작 시 actor 가 head hot-swap.
5. 반복하며 head 가 사람 교정을 점점 반영. **frozen base 라 옛 위치도 계속 성공(망각 없음)**.

## 첫 실행이라 눈으로 볼 것 (VERIFY, 코딩 아님)
- **Δ캡 작게 시작**: 기본 `--residual_translation_cap 0.05 --residual_rotation_cap 0.4`. 로봇이 튀면 더 작게.
- **이미지 스케일**: 정책이 엉뚱하게 움직이면 `_to_uint8_image` 의 0~1↔0~255 추정 확인.
- **교정 중 slow 주기**: teleop 모드는 매 tick slow 재추론이라 느릴 수 있음. 느리면 `--num_inference_steps` ↓.
- **손은 카메라 시야 밖**에서 밀 것(시야 안이면 "손 보이면" 오학습).
- **임피던스 stiffness 낮게**(사람이 손으로 밀 수 있게).

## 트러블슈팅
| 증상 | 원인/해결 |
|------|-----------|
| `ModuleNotFoundError: timm` | `robodiff_rush` 로 돌림 → `venv_diffusion` 사용. |
| slow ckpt 없음 | 경로 오타/미존재. 위 고정 경로 확인. |
| hot-swap "shape mismatch"/스킵 | 낡은 workdir 재사용. 0단계로 비우고 learner·actor 재시작. |
| actor 로봇이 튐 | Δ캡 미적용 또는 stiffness 과다. 캡 낮추고 stiffness ↓. |
| learner 재시작 시 head v0 부터 | 정상(누적 초기화). 보존하려면 실행 전 workdir 백업. |
