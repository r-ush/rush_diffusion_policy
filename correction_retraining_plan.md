# Human-Correction 기반 재학습 Plan (base ckpt: logistic_box_unet_abs / epoch=0800)

CR-DAgger 아이디어를 이 리포의 스택(Doosan M0609, ROS2 임피던스 컨트롤러, F/T 센서 없음)에 맞춰
"오프라인 반복" 버전으로 적용한다. Base policy는 그대로 두고, 사람이 물리적으로 교정한 데이터를
모아서 같은 policy를 낮은 LR로 파인튜닝하는 방식 (residual policy 방식(B안)은 별도 논의 필요).

## 0. 사전 조건
- `rush_eval_real_robot_imp.py`에 'C'(correction 토글) / 'S'(stop&keep) / 'D'(stop&discard) 키가 추가됨.
- 임피던스 stiffness가 사람이 밀 수 있을 만큼 낮게 설정되어 있어야 함 (너무 뻣뻣하면 교정 자체가 안됨).
- 사람 손/팔이 카메라 시야에 들어가지 않는 위치에서 로봇을 잡고 교정할 것 (안 그러면 정책이
  "사람 손이 보이면 이렇게 움직인다"는 잘못된 상관관계를 학습할 수 있음).

## Step 1: 교정 데이터 수집
```bash
conda activate robodiff
python rush_eval_real_robot_imp.py \
    --input data/outputs/logistic_box_unet_abs/checkpoints/epoch=0800-train_loss=0.000.ckpt \
    --output data/results/correction_session1
```
- 평소처럼 policy가 동작하게 두고, 실패하거나 궤적이 안 좋아 보이면 **'C'를 눌러 correction 모드를 켠 뒤**
  로봇을 물리적으로 밀어서 원하는 동작으로 유도. 끝나면 다시 'C'를 눌러 끔.
- 에피소드가 끝나면 'S'(저장) 또는 'D'(폐기)로 마무리.
- 결과: `data/results/correction_session1/replay_buffer.zarr`

## Step 2: Relabel + HDF5 변환
```bash
python data_process/rush_replay_buffer_to_correction_hdf5.py \
    --input data/results/correction_session1/replay_buffer.zarr \
    --output /home/rush/Desktop/Datasets/correction_batch1.hdf5 \
    --oversample 3
```
- action 라벨이 "정책이 낸 명령"이 아니라 "한 스텝 뒤 실제 도달한 pose"로 재작성됨 (relabel 원리는
  스크립트 상단 docstring 참고).
- `--oversample 3`: correction이 포함된 에피소드를 3배 복제해서 학습 시 더 자주 샘플링되게 함.
  데이터가 적을 때(수십 에피소드 이하)는 3~5, 많을 때는 1~2 정도로 시작해보고 과적합 여부를 보면서 조정.

## Step 3: 기존 데이터와 병합 (선택, 권장)
Base policy 학습에 쓴 원본 HDF5와 합쳐서 파인튜닝하면 base 성능을 덜 잃는다 (교정 데이터만으로
파인튜닝하면 특정 실패 케이스에 과적합되어 다른 상황 성능이 나빠질 수 있음, catastrophic forgetting).
```bash
python data_process/rush_merge_hdf5_datasets.py \
    --inputs /home/rush/Desktop/Datasets/20260630_195919_diffusion_des.hdf5 \
             /home/rush/Desktop/Datasets/correction_batch1.hdf5 \
    --output /home/rush/Desktop/Datasets/logistic_box_finetune_v1.hdf5
```
(base HDF5 경로는 `epoch=0800...ckpt`의 `cfg.task.dataset_path`에서 확인한 값)

## Step 4: Task YAML 만들기
`diffusion_policy/config/task/rush_logistic_box_finetune_v1.yaml` (기존
`rush_logistic_box_pose_only.yaml` 복사 후 `dataset_path`만 교체):
```yaml
name: rush_logistic_box_finetune_v1
dataset_path: /home/rush/Desktop/Datasets/logistic_box_finetune_v1.hdf5
image_shape: [3, 240, 320]

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
      rotation_rep: rotation_6d
  action:
    shape: [9]
    rotation_rep: rotation_6d

pose_repr: &pose_repr
  obs_pose_repr: abs
  action_pose_repr: abs

env_runner:
  _target_: diffusion_policy.env_runner.real_pusht_image_runner.RealPushTImageRunner

dataset:
  _target_: diffusion_policy.dataset.bae_robomimic_replay_image_dataset.BaeRobomimicReplayDataset
  shape_meta: *shape_meta
  dataset_path: ${task.dataset_path}
  horizon: ${horizon}
  pad_before: ${eval:'${n_obs_steps}-1'}
  pad_after: ${eval:'${n_action_steps}-1'}
  n_obs_steps: ${dataset_obs_steps}
  use_cache: True
  seed: 42
  val_ratio: 0.02
  pose_repr: *pose_repr
```
> [!CAUTION]
> `use_cache: True`이므로 `<dataset_path>.zarr.zip` 캐시가 새로 생성됨. 같은 파일명을 재사용하며
> 내용만 바꿨다면 캐시를 지우고 다시 실행해야 함 (`rm -f <dataset_path>.zarr.zip*`).

## Step 5: 파인튜닝 실행
이미 `checkpoint.resume_path` 필드가 지원되므로 (`base_workspace.py`, `train_diffusion_unet_hybrid_workspace.py`
의 `_get_resume_checkpoint_path`), 새 코드 없이 기존 `train.py`로 바로 이어받아 학습 가능:
```bash
HYDRA_FULL_ERROR=1 python train.py \
    --config-name=bae_train_diffusion_unet_real_hybrid_workspace \
    task=rush_logistic_box_finetune_v1 \
    checkpoint.resume_path=data/outputs/logistic_box_unet_abs/checkpoints/epoch=0800-train_loss=0.000.ckpt \
    training.resume=True \
    training.num_epochs=200 \
    optimizer.lr=1.0e-5 \
    hydra.run.dir='data/outputs/${now:%y%m%d}_logistic_box_finetune_v1'
```
- `optimizer.lr`을 원래 학습(1e-4)보다 낮게(1e-5) 주는 것이 핵심 — 그래야 base policy가 이미 잘하던
  부분을 크게 잊지 않으면서 교정 패턴만 추가로 배움.
- `training.num_epochs`는 짧게 시작 (100~300) 하고 val_loss/rollout으로 과적합 여부를 보면서 늘릴지 결정.
- 저장은 새 `hydra.run.dir`로 나가므로 기존 `logistic_box_unet_abs` 체크포인트는 보존됨 (원본 정책으로
  언제든 되돌아갈 수 있음).

## Step 6: 검증
```python
import torch, dill
from omegaconf import OmegaConf
p = torch.load(open("<NEW_CKPT_PATH>", 'rb'), pickle_module=dill, weights_only=False)
cfg = p['cfg']
print("task.dataset_path:", cfg.task.dataset_path)      # merged hdf5 경로인지
print("optimizer.lr:", cfg.optimizer.lr)                 # 낮은 lr로 학습됐는지
```

## Step 7: 평가
```bash
python rush_eval_real_robot_imp.py \
    --input data/outputs/<날짜>_logistic_box_finetune_v1/checkpoints/latest.ckpt \
    --output data/results/finetune_v1_eval
```
교정했던 실패 케이스가 개선됐는지, 다른 정상 케이스 성능이 유지되는지 함께 확인. 나빠졌다면
(a) `optimizer.lr`을 더 낮추거나 (b) `--oversample`을 낮추거나 (c) num_epochs를 줄여서 재시도.

## 다음 확장 (필요 시)
- 여러 세션 반복: Step1~7을 사이클로 돌리면서 매번 새 correction 데이터를 이전 merged 데이터셋에
  추가 (`rush_merge_hdf5_datasets.py --inputs <이전 merged> <새 correction batch>`).
- Residual policy(B안, cr-dagger 스타일): base policy를 동결하고 작은 MLP가 (인코딩된 obs + base
  action) → Δpose를 회귀하도록 별도 학습 + inference 시 `SE3_base @ SE3_residual`로 합성. 필요하면
  별도로 설계/구현 요청.
