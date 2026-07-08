# bae_diffusion_policy → 현재 repo 병합 계획

> **작성 2026-07-08.** 소스 `/home/rush/bae_diffusion_policy`(비-git, 105M)의 코드를 현재
> repo(`/home/rush/rush_diffusion_policy`, git)로 병합. SSD 병합과 **다른 별도 연구 fork**로,
> BAE는 **force / wrench-encoder / DiT / flow-matching / residual policy / plug 삽입** 라인이다.
> 대용량 데이터·venv·결과물은 가져오지 않는다(코드만).

## 결정 원칙
- **B1(공통 파일 중 BAE가 인코더에 맞춰 파라미터/모델을 추가한 것) → BAE 채택** (사용자 확정).
- **B2(인프라·SSD계보 파일) → 현재 유지** (BAE 고유 줄이 옛 코드/경로임을 diff로 확인).
- **A(BAE 전용 신규) → 순수 추가**. **C(중복/데이터/venv) → 스킵.** **D(현재 전용) → 유지.**

---

## A. 순수 추가 (83개, 충돌 0) — `/tmp/bae_additive.txt` 기준
- `diffusion_policy/residual_policy/` 전체 (26) + `diffusion_policy/config/residual_policy/` (6)
- Force 학습/분석: `train_force_distribution.py`, `visualize_force_feature_space.py`,
  `visualize_force_phase_windows.py`, `tools/` (8), `scripts/` (5 .sh)
- 신규 모델/정책: `model/diffusion/{bae_transformer_for_diffusion_force_adaln_vector,transformer_for_action_diffusion}.py`,
  `model/vision/{transformer_obs_encoder,transformer_obs_wrench_encoder}.py`,
  `policy/{bae_flow_matching_unet_hybrid_image_wrench_encoder_policy,diffusion_transformer_hybrid_image_wrench_encoder_dit_policy,diffusion_transformer_timm_dit_policy,diffusion_transformer_timm_policy}.py`,
  `workspace/train_diffusion_transformer_timm_workspace.py`,
  `diffusion_policy/scripts/` (5 viz)
- 신규 config: `..._force_dit.yaml`, `..._unet_..._no_force.yaml`, flow_matching `_force`/`_no_force`
- Plug 삽입: `bae_eval_real_robot_rightarm_insert_plug{,_fast_loop}.py`,
  `real_world/{bae_real_env_rightarm_hand_insert_plug,rightarm_hand_insert_plug_interpolation_controller}.py`,
  `config/task/bbbae_dualarm_insert_plug_*` (3) + `bbbae_dualarm_box_insertion_wrench_encoder.yaml` + `bbbae_dualarm_erase_board_{no_wrench,wrench_encoder}.yaml`
- data_process: `common_to_diffusion_hand_R_wrench_encoder_{erase,plug}.py`,
  `zarr_common_to_diffusion_box_insertion.py`, `plot_{flip_v2_trajectory_plotly,hdf5_robot_pose_trajectory_plotly,height_actual_vs_desired_pose}.py`
- `m0609.white_weird.urdf`

## B1. BAE 채택 (13개, 현재 덮어씀)
model/policy (8):
- `dataset/base_dataset.py`
- `model/diffusion/ema_model.py`
- `model/diffusion/bae_transformer_for_diffusion_force_adaln.py`
- `model/diffusion/bae_transformer_for_diffusion_force.py`
- `policy/diffusion_transformer_hybrid_image_DiT_policy.py`
- `policy/diffusion_transformer_hybrid_image_policy.py`
- `policy/diffusion_transformer_hybrid_image_wrench_encoder_policy.py` (대형 분기 +bae331)
- `policy/bae_diffusion_unet_hybrid_image_wrench_encoder_policy.py`

학습 config (5, 위 모델과 짝):
- `config/bae_train_diffusion_transformer_real_hybrid_workspace{,_dit,_force}.yaml`
- `config/bae_train_diffusion_unet_real_hybrid_workspace{,_force}.yaml`

## B2. 현재 유지 (인프라·SSD계보, BAE 고유줄=옛코드 확인)
- `model/common/{rotation_transformer,lr_scheduler}.py`, `workspace/base_workspace.py`
- `workspace/train_diffusion_{unet,transformer}_hybrid_workspace.py`
- `common/pose_trajectory_interpolator.py`
- `dataset/{robomimic_replay_image_dataset,bae_robomimic_replay_image_dataset}.py` (zarr codec 호환)
- `policy/diffusion_unet_hybrid_image_policy.py` (CropRandomizer 가드)
- **모든 `real_world/*` env·controller** (현재 +cur가 크게 앞섬; 특히 wrench_encoder controller +cur152)
- 모든 top-level 공통 `bae_eval_*.py`, `scheduling_ddim.py`, `bae_scheduling_ddim_pigdm.py`,
  `README.md`(stub 유지), `setup.py`, `conda_environment.yaml`,
  `m0609.white.urdf`(origin-z **0.1345** 확정)
- `data_process/common_to_diffusion_hand_R_raw_wrench{,_plug}.py`
  (_plug은 +bae5/+cur3 경미 분기 — 필요시 사후 검토)

## C. 스킵
`config/task/task_bae/`(16, 현재 root와 동일 재조직)·`task_old/`(5)·`config/config_old/`(umi 아카이브),
`bae_image_augmentation_test/`(png), `data_process/erase_board_force_data/`(17M png, gitignore),
`venv_dp/`, `numba`(빈 스텁).

## D. 현재 전용 → 유지
rush_* 계열, SSD-inference imp env, leftarm/AIDIN, correction 파이프라인, zarr 툴링 등.

---

## 실행 순서
1. A: `/tmp/bae_additive.txt`의 83개를 구조 유지하며 복사.
2. B1: 13개를 BAE에서 덮어쓰기.
3. 스모크: 신규/변경 .py 문법 파싱 + `import diffusion_policy` + 주요 신규 정책 import 확인.
   (BAE 정책이 현재 유지된 rotation_transformer 등 인프라 API와 호환되는지 특히 확인.)
4. 커밋: 논리 단위(추가/모델·정책 채택)로 분리. push는 사용자.

## 리스크
- BAE 신규 정책/워크스페이스가 B1 모델 파일을 전제 → B1을 함께 가져오므로 자체 정합성 OK.
- 단, B2로 남긴 인프라(특히 real_world env/controller)와 BAE 신규 inference 스크립트 간
  시그니처 불일치 가능 → import/문법 스모크로 1차 확인, 로봇 실행은 별도.
