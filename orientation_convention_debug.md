# Orientation Convention 디버깅 메모 (deploy 시 자세 ~150° 오차)

작성일: 2026-07-03 / 상태: **데이터 원인 확정 + 수정 완료 (zarr/logistic_box 파이프라인)**

---

## ★ 확정된 근본 원인 (데이터) + 수정 — 2026-07-03 추가

**zarr 데이터셋(`20260630_195919`)의 `command_pose_se3` 회전이 오염되어 있었음.**

측정 결과:
- `command` vs `ee` 자세: **stored 상태 ~166° 고정 오차**(위치는 ~8~10mm로 정상), 전 에피소드 일관(std ~2°).
- **stored 회전을 `ZYZ` order로 각도 추출 → `ZYX` order로 재구성**하면 `ee` 추종(잔차 ~3° = 실제 추종오차)으로 완벽히 붙음.
- 즉 수집 시 command SE3를 **ZYZ euler로 만들었으나 원시 command 각도는 실제 ZYX** 였음. (meta.json의 `command_float64_euler_order: zyz` 가 오히려 틀림. 사용자가 "실 로봇은 zyx"라 한 것과 일치.)
- `ee_pose_se3`(FK)는 정상 → **command 회전만 오염**. 학습 action = command 이므로 action label 이 통째로 틀어져 있었음.

**수정 (rot6d 학습 방식 유지, 데이터만 보정):**
- `data_process/zarr_episodes_to_diffusion_hdf5.py` 에 `fix_command_orientation_zyz_to_zyx()` 추가.
  command 회전을 `R.from_euler('ZYX', R.from_matrix(stored).as_euler('ZYZ'))` 로 재구성 후 rot6d 추출.
- 플래그 `FIX_COMMAND_ZYZ_TO_ZYX=True` (기본 on).
- 검증: 보정 후 action(command) 회전 vs obs(ee) 잔차 mean 2.37° / max 4.5°. rot6d 포맷 불변.

**다음 단계:** hdf5 재생성(`_diffusion.hdf5` 새로 뽑기, 캐시 `.zarr.zip` 삭제 후) → 기존 `rush_logistic_box_pose_only` config 그대로 재학습.

**스코프 주의:** 아래 배포 로그는 **우완 insert_plug** 모델(action=FK TCP pose, command 아님)이라 이 command 오염과는 **다른 파이프라인**. 다만 동일한 "로봇 ZYX vs 데이터 ZYZ" 혼동이 공통 뿌리이므로, 배포측 obs/command euler order도 ZYX 기준으로 점검할 것.

---

## (이하) 최초 배포 로그 기반 후보 정리 (deploy-side, insert_plug)

작성일: 2026-07-03 / 상태: 후보 (insert_plug 배포측)

관련 실행:
```
python bae_eval_real_robot_rightarm_insert_plug.py \
  --input data/outputs/260703_insert_box_wrench/epoch=0800-train_loss=0.001.ckpt \
  --output data/results
# action_rotation_send_mode: zyz, Pose representation: obs abs / action relative
```

---

## 1. 증상

- 크래시 아님. 로그의 traceback은 전부 `KeyboardInterrupt`(직접 Ctrl-C). → **동작 이상**(로봇이 엉뚱한 자세로 감).
- 실 로봇은 (사용자 인식상) zyx인데 안 돼서 zyz로 바꿔봄 → **여전히 이상**.

## 2. 결정적 수치 증거 (로그에서 직접 추출)

"obs abs / action **relative**" 정책의 **첫 inference 첫 스텝** → target은 현재 pose와 거의 같아야 함.

| 채널 | 시작 curr_pose | 첫 target_pose | 판정 |
|------|----------------|----------------|------|
| position | `[0.519, 0.006, 0.101]` | `[0.524, -0.004, 0.087]` | ~2cm, **정상** (relative 잘 붙음) |
| orientation | `[-0.01, 1.10, -0.83]` | `[-0.73, -2.38, -0.99]` | **~150° 어긋남** |

`curr`↔`target`을 **ZYX/ZYZ/XYZ 어떤 조합으로 해석해도 두 회전 사이 각도 140~170°** (≈0°이 되는 convention 짝이 하나도 없음).

→ **결론: send convention(zyx↔zyz) 문제가 아니다.** send mode는 파이프라인 맨 끝(회전행렬→euler)만 바꾸는데, 어떤 euler 해석으로도 안 맞으므로 오류는 **euler 변환 이전 단계**에 있음. 그래서 zyx도 zyz도 둘 다 실패.

## 3. 원인 후보 (deploy 시점, 둘 중 하나 또는 둘 다)

### (A) 관측 자세 `robot_quat_R` 생성 불일치
- 학습 시 `robot_quat_R` = `R.from_matrix(FK회전행렬).as_quat()` → **convention-free, 물리적으로 정확**.
- eval에서 로봇 현재 pose(두산=euler)로 quat을 만들 때 **euler order를 로봇 실제(ZYZ)와 다르게** 쓰면, 같은 물리 자세인데 다른 quat → 정책 입력 자세 오염 → 출력 통째로 오염.
- **확인 지점**: `rightarm_hand_insert_plug_interpolation_controller.py`에서 `robot_quat_R` 만드는 라인.
  - FK 행렬 기반(`from_matrix(...).as_quat()`)이면 → obs는 정상, (A) 배제 → (B)로.
  - 로봇 euler 기반(`from_euler(ORDER, ...)`)이면 → ORDER가 로봇 실제 convention과 학습과 일치하는지 확인.

### (B) relative 자세 합성 오류
- action이 relative라 모델 delta 회전을 현재 자세에 곱해 절대 target 생성.
- **곱 순서(pre/post-multiply) 또는 기준 프레임(base vs body)** 이 학습/의도와 다르면, 입력이 맞아도 절대 자세가 **일정하게** 틀어짐 → 관측된 "일정한 ~150° offset"과 일치.
- 참고: 이 repo의 analog 컨트롤러 `diffusion_policy/real_world/rightarm_hand_with_wrench_interpolation_controller.py`는 전 구간 **rotvec/FK 행렬**만 쓰고 euler send가 없음 → 이 문제 없음. insert_plug의 **relative + euler-send** 로직이 새로 들어온 의심 지점.

## 4. 검증 기준 (참/거짓 판정)

수정 후 다시 돌렸을 때 **첫 스텝 target 자세가 curr와 거의 같아지는지**(≈150° → ≈0°) 확인. 이게 유일한 판정 기준.

부수 확인:
- euler 배열 순서가 `[a,b,c]`인지 `[rz,ry,rx]`처럼 뒤집혀 들어가는지
- **deg/rad 단위** (두산 posx는 deg)

## 5. 매우 중요 — 재학습으로는 이 문제를 못 고침

학습 데이터 orientation은 **quat(obs) / rot6d(action)** 이고, 둘 다 **FK 회전행렬에서 파생 → euler convention 없음**.
- 근거: `data_process/common_to_diffusion_hand_R_raw_wrench_plug.py` (obs line 278 `from_matrix(...).as_quat()`, action line 314-316 `quat_to_6d`), `data_process/zarr_episodes_to_diffusion_hdf5.py` 동일.
- 따라서 **"데이터셋을 zyx로 재변환 + 재학습"은 orientation 관점 no-op** (물리 회전 동일). ~150° 문제는 순수 배포측(로봇 I/O) 이슈.

## 6. 데이터 수집측 별도 점검거리 (선택)

- `command_pose_se3.zarr`는 command 토픽(`/bae_r/desired_pose`, meta.json상 euler order = **zyz**)을 수집 시 SE3로 변환한 것.
- 수집기가 이 zyz euler를 SE3로 바꿀 때 order를 틀렸다면 `command_pose_se3` 자체가 물리적으로 틀림 → 이 경우엔 데이터가 오염된 것.
- 단, 위 plug 파이프라인의 action은 **command가 아니라 FK TCP pose**에서 나오므로 이 오염과 무관. (command 기반 파이프라인을 썼을 때만 관련.)
