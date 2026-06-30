# Relative Action Diffusion Policy 재학습 Plan

## 문제 원인

이전 학습에서 task config의 `dataset._target_`이 `RobomimicReplayImageDataset`으로 설정됨.
이 클래스는 `pose_repr` 파라미터를 **지원하지 않아** config에 `action_pose_repr: relative`라 적혀 있어도 **절대(abs) action으로 학습**됨.

올바른 dataset 클래스는 `BaeRobomimicReplayDataset`이며, 이 클래스만이 `pose_repr`을 받아 학습 시 relative 변환을 수행함.

---

## Step 1: Task YAML 생성

`diffusion_policy/config/task/` 디렉토리에 새 task yaml 파일을 만들어야 함.

기존 체크포인트에서 추출한 설정 기반 + `BaeRobomimicReplayDataset` 사용:

```yaml
# diffusion_policy/config/task/rush_leftarm_desired_pose_relative.yaml

name: rush_leftarm_desired_pose_relative

image_shape: [3, 240, 320]
dataset_path: <HDF5_PATH>   # ← 실제 HDF5 경로로 교체

shape_meta: &shape_meta
  obs:
    image0:
      shape: [3, 240, 320]
      type: rgb
    robot_pose_L:
      shape: [3]
      type: low_dim
    robot_quat_L:
      shape: [4]
      type: low_dim
      rotation_rep: rotation_6d    # ← BaeRobomimicReplayDataset에서 사용
  action:
    shape: [9]   # pos_L(3) + rot6d_L(6)
    rotation_rep: rotation_6d      # ← 필수: normalizer에서 사용

pose_repr: &pose_repr
  obs_pose_repr: abs               # obs는 절대값 유지
  action_pose_repr: relative       # action만 relative

env_runner:
  _target_: diffusion_policy.env_runner.real_pusht_image_runner.RealPushTImageRunner

dataset:
  #############################################################
  # 핵심 변경: BaeRobomimicReplayDataset 사용
  #############################################################
  _target_: diffusion_policy.dataset.bae_robomimic_replay_image_dataset.BaeRobomimicReplayDataset
  shape_meta: *shape_meta
  dataset_path: ${task.dataset_path}
  horizon: ${horizon}
  pad_before: ${eval:'${n_obs_steps}-1'}
  pad_after: ${eval:'${n_action_steps}-1'}
  n_obs_steps: ${dataset_obs_steps}
  use_cache: True
  seed: 42
  val_ratio: 0.0
  pose_repr: *pose_repr            # ← 핵심: pose_repr 전달
  # abs_action: True               # BaeRobomimicReplayDataset에서는 불필요
```

> [!IMPORTANT]
> - `dataset._target_`이 반드시 `bae_robomimic_replay_image_dataset.BaeRobomimicReplayDataset`이어야 함
> - `dataset.pose_repr: *pose_repr`이 반드시 포함되어야 함
> - `action.rotation_rep: rotation_6d`가 반드시 있어야 함 (normalizer 계산에 사용)
> - `robot_quat_L`에 `rotation_rep: rotation_6d`가 있어야 함

---

## Step 2: 기존 캐시 삭제

이전 학습에서 `use_cache: True`로 생성된 `.zarr.zip` 캐시가 남아 있으면 abs 데이터가 재사용됩니다.

```bash
# HDF5 파일 경로 옆에 있는 캐시 파일 삭제
rm -f <HDF5_PATH>.zarr.zip
rm -f <HDF5_PATH>.zarr.zip.lock
```

> [!CAUTION]
> 캐시를 삭제하지 않으면 이전 abs 데이터로 학습될 수 있음!

---

## Step 3: HDF5 데이터 확인

HDF5 파일의 action 데이터가 **절대 좌표 (pos + rot6d, 9차원)**으로 저장되어 있어야 함.
`BaeRobomimicReplayDataset`이 학습 시 자동으로 relative 변환을 수행하므로, HDF5에는 **절대값**이 있어야 정상.

```python
import h5py
import numpy as np

with h5py.File("<HDF5_PATH>", 'r') as f:
    demo = f['data/demo_0']
    actions = demo['actions'][:]
    print("Action shape:", actions.shape)         # (T, 9) 예상
    print("Action[0]:", actions[0])               # 절대 pos + rot6d 값
    print("Action range:", actions.min(0), actions.max(0))

    # obs도 확인
    pose_L = demo['obs/robot_pose_L'][:]
    quat_L = demo['obs/robot_quat_L'][:]
    print("robot_pose_L shape:", pose_L.shape)    # (T, 3)
    print("robot_quat_L shape:", quat_L.shape)    # (T, 4) - quaternion xyzw
```

> [!NOTE]
> - `actions` shape이 `(T, 9)`여야 함
> - `robot_quat_L` shape이 `(T, 4)`여야 함 (quaternion)
> - action의 position 값이 절대 좌표 범위(예: -0.5 ~ 0.5m)에 있어야 함

---

## Step 4: 학습 실행

```bash
python train.py \
    --config-name=train_diffusion_unet_hybrid_workspace \
    task=rush_leftarm_desired_pose_relative \
    training.num_epochs=1000 \
    training.device=cuda:0 \
    hydra.run.dir='data/outputs/${now:%y%m%d}_des_relative'
```

---

## Step 5: 학습 검증 (체크포인트 로드 후)

학습 완료 후 체크포인트를 로드하여 설정이 올바른지 반드시 확인:

```python
import torch, dill
from omegaconf import OmegaConf

p = torch.load(open("<NEW_CKPT_PATH>", 'rb'), pickle_module=dill)
cfg = p['cfg']

# 1. Dataset class 확인
print("dataset._target_:", cfg.task.dataset._target_)
# 기대값: diffusion_policy.dataset.bae_robomimic_replay_image_dataset.BaeRobomimicReplayDataset

# 2. pose_repr 확인
print("pose_repr:", cfg.task.get('pose_repr', 'NOT SET'))
# 기대값: {'obs_pose_repr': 'abs', 'action_pose_repr': 'relative'}

# 3. dataset에 pose_repr이 전달되었는지 확인
print("dataset.pose_repr:", cfg.task.dataset.get('pose_repr', 'NOT SET'))
# 기대값: {'obs_pose_repr': 'abs', 'action_pose_repr': 'relative'}
# ⚠️ 'NOT SET'이면 다시 학습해야 함!
```

---

## Step 6: 추론 실행

```bash
python rush_eval_real_robot_imp.py \
    --input <NEW_CKPT_PATH> \
    --output data/results \
    --steps_per_inference 12 \
    --frequency 10 \
    --num_inference_steps 12
```

추론 시 로그에서 확인할 것:
- `[DEBUG] cfg.task.pose_repr: {'obs_pose_repr': 'abs', 'action_pose_repr': 'relative'}` ✓
- `Generated Action (Relative)` position이 **[0, 0, 0] 근처** (±0.05 이내) ✓
- `Converted Absolute Target Pose` position이 **현재 로봇 위치 근처** ✓

---

## 요약 체크리스트

| # | 항목 | 확인 |
|---|---|---|
| 1 | Task YAML의 `dataset._target_`이 `BaeRobomimicReplayDataset`인지 | ☐ |
| 2 | Task YAML에 `pose_repr` anchor와 `dataset.pose_repr` 참조가 있는지 | ☐ |
| 3 | `action`에 `rotation_rep: rotation_6d`가 있는지 | ☐ |
| 4 | 기존 `.zarr.zip` 캐시 삭제했는지 | ☐ |
| 5 | HDF5의 action이 절대 좌표(9D)인지 | ☐ |
| 6 | 학습 후 체크포인트에 `dataset.pose_repr`이 올바르게 저장되었는지 | ☐ |
| 7 | 추론 시 relative action 값이 0 근처인지 | ☐ |
