수집 스크립트는 찾았고(dualarm_data_gen_w_wrench.py), 이어서 학습에 바로 연결되는 변환 스크립트의 실행 방식(인자형/하드코딩형)을 확인해 최종 순서를 완성하겠습니다.

실로봇 데이터 수집→학습 기준으로, 이 레포에서 바로 따라갈 수 있는 최소 실행 순서입니다.
(중요: README의 bae_hdf_maker_abs.py 예시는 현재 워크스페이스에 없고, 실제 수집 스크립트는 dualarm_data_gen_w_wrench.py 입니다.)

1) 환경 준비

작업 폴더 이동: cd /home/rush/diffusion_policy
Conda 환경 생성: conda env create -f conda_environment_real.yaml
활성화: conda activate robodiff
로컬 패키지 등록: pip install -e .
2) 원본 데모 수집 (common_data)

실행: python dualarm_data_gen_w_wrench.py -m
키 조작: s(수집 시작), q(수집 중지), y/n(해당 demo 저장 여부), t(전체 종료)
결과 파일: /data/YYMMDD_HHMM/common_data.hdf5 형태로 저장됨 (스크립트 내부 동작)
3) 학습용 HDF5로 변환

변환 스크립트: common_to_diffusion_hand_R_raw_wrench.py
먼저 스크립트의 input_filenames/output_filename 값을 본인 경로로 수정
실행: python common_to_diffusion_hand_R_raw_wrench.py
구조 확인(권장): python get_hdf_info.py /변환결과/xxx.hdf5
4) 학습 실행 (예: raw wrench + transformer)

명령:
python train.py --config-name=bae_train_diffusion_transformer_real_hybrid_workspace_raw_wrench task=bbbae_dualarm_erase_board_with_raw_wrench task.dataset_path=/변환결과/xxx.hdf5 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}${name}${task_name}'
관련 설정 파일:
bae_train_diffusion_transformer_real_hybrid_workspace_raw_wrench.yaml
bbbae_dualarm_erase_board_with_raw_wrench.yaml
5) 체크포인트 평가

기본 평가: python eval.py -c data/outputs/.../checkpoints/latest.ckpt -o data/results/exp1
실로봇 평가 스크립트(태스크별): bae_eval_real_robot.py 또는 bae_eval_real_robot_dualarm.py 등에서 선택 후 -h로 옵션 확인
원하시면 다음으로, 지금 하시려는 태스크명(예: erase_board/raw_wrench/dualarm_hand)에 맞춰 “수정할 파일 1~2개 + 실행 명령 3줄”만 남긴 초간단 실행판으로 맞춰드릴게요.

변환에서 하는 일 (왜 필요한가)

변환 스크립트 예: common_to_diffusion_hand_R_raw_wrench.py
목적: 원본 멀티레이트/원시센서 데이터를 모델 학습이 바로 가능한 고정 포맷으로 바꾸기
다운샘플: robot/image 20Hz → 10Hz
전처리: wrench 초기 오프셋 제거 + EMA 필터
시간 정렬: 10Hz robot timestamp마다 가장 가까운 wrench 샘플 매칭
표현 변환: joint → TCP pose/quat, quat → 6D rotation
이미지 변환: resize/crop + RGB 변환
채널 선택/축소: 손가락 wrench는 Fz만 사용 등



1) inference에 쓰는 코드

공통 오프라인 평가는 eval.py에서 체크포인트를 로드하고 policy를 꺼내서 runner를 돌림.
실로봇 평가는 bae_eval_real_robot.py:74에서 체크포인트 로드 후, 루프에서 bae_eval_real_robot.py:161, 그리고 bae_eval_real_robot.py:255 순서로 진행됨.
실제 action 생성 핵심은 diffusion_transformer_hybrid_image_policy.py:228에서 수행됨.
2) 어떤 observation + 어떤 checkpoint 파라미터로 action을 만드나

push84 task 기준 obs/action 스키마는 bae_push_image_abs_84.yaml:
image0, image1, robot_eef_pos(3), robot_eef_quat(4) → action(9).
실시간 obs는 real_inference_util.py:11에서 shape_meta에 맞게 변환/정렬됨.
정책은 체크포인트 안의 모델 가중치(보통 ema_model 우선), obs encoder, normalizer 통계값을 사용해서
obs 정규화 → diffusion denoising 샘플링 → action unnormalize → 실행 구간(n_action_steps) 추출로 action을 만듦.
horizon/n_obs_steps/n_action_steps 같은 생성 규칙은 bae_train_diffusion_transformer_real_hybrid_workspace_push84.yaml:11-15 값을 따름.
3) action이 ROS2로 전달되는 방식

공통으로 bae_real_env.py:350-376에서 action들을 timestamp와 함께 schedule_waypoint 큐에 넣음.
ROS2 제어 스택(예: rightarm_hand_with_wrench)은 controller 프로세스에서 rightarm_hand_with_wrench_interpolation_controller.py:568,
rightarm_hand_with_wrench_interpolation_controller.py:181-193,
rightarm_hand_with_wrench_interpolation_controller.py:302-324으로 JointState/PoseStamped 토픽에 내보냄.
단, bae_eval_real_robot.py는 arm 자체는 RTDE 계열 controller를 타므로 “ROS2 퍼블리시”가 아니라 로봇 드라이버 경로로 가는 케이스도 있음.
즉, 어떤 eval 스크립트를 쓰느냐에 따라 마지막 전송층이 ROS2일 수도/아닐 수도 있어.
원하면 네가 실제로 돌릴 스크립트 하나(예: with_wrench_encoder) 기준으로, obs key부터 토픽명까지 1:1 매핑표로 깔끔하게 뽑아줄게.


# Test run 1: fresh start (epoch 0 → 16 정도)
python train.py --config-name=bae_train_diffusion_transformer_real_hybrid_workspace_push84 \
  task.dataset_path=/home/rush/diffusion_policy/data/baetae/bae_push_image_abs_0717.hdf5 \
  exp_name=push84_test_ckpt \
  logging.mode=online


# 중간에 여러 checkpoint가 epoch=0010, epoch=0020, ... 으로 저장될 것

# Test run 2: resume (이전 run에서 epoch 10~ 복구)
python train.py --config-name=bae_train_diffusion_transformer_real_hybrid_workspace_push84 task.dataset_path=/home/rush/diffusion_policy/data/baetae/bae_push_image_abs_0717.hdf5 exp_name=push84_test_resume training.resume=true 'checkpoint.resume_peath="/home/rush/diffusion_policy/outputs/2026.04.20/01.32.42_bae_push_image_abs_84_push84_test_ckpt/checkpoints/epoch=0050-train_loss=0.035.ckpt"' logging.mode=online

python train.py \
  --config-name=bae_train_diffusion_unet_real_hybrid_workspace \
  task=rush_leftarm_pose_only




Processing /media/rush/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7/260429_0102/common_data.hdf5: 10
Data conversion completed / total demos = 64
(robodiff) rush@rush:~/diffusion_policy$ python3 hdf5_viewer.py 
(robodiff) rush@rush:~/diffusion_policy$ python3 hdf5_viewer.py 
(robodiff) rush@rush:~/diffusion_policy$ /home/rush/diffusion_policy/venv_dp/bin/python /home/rush/diffusion_policy/data_process/rush_common_to_diffusion_leftarm_desired_from_F_e_raw.py 
/media/rush/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7/260429_0102/common_data.hdf5 / demo_len = 151
Selective processing: demos [1, 2, 5, 8, 9, 10, 11, 13, 15, 19, 20, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 38, 39, 43, 44, 45, 46, 48, 50, 52, 53, 54, 56, 57, 58, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 82, 83, 85, 86, 87, 88, 90, 93, 94, 96, 98, 99]
Processing /media/rush/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7/260429_0102/common_data.hdf5: 10
Data conversion completed / total demos = 64
(robodiff) rush@rush:~/diffusion_policy$ python3 hdf5_viewer.py 
(robodiff) rush@rush:~/diffusion_policy$ python3 hdf5_viewer.py 
(robodiff) rush@rush:~/diffusion_policy$ python3 hdf5_viewer.py 
(robodiff) rush@rush:~/diffusion_policy$ /home/rush/diffusion_policy/venv_dp/bin/python /home/rush/diffusion_policy/data_process/rush_common_to_diffusion_leftarm_desired_from_F_e_raw.py 
/media/rush/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7/260429_0102/common_data.hdf5 / demo_len = 151
Selective processing: demos [1, 2, 5, 8, 9, 10, 11, 13, 15, 19, 20, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 38, 39, 43, 44, 45, 46, 48, 50, 52, 53, 54, 56, 57, 58, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 82, 83, 85, 86, 87, 88, 90, 93, 94, 96, 98, 99]
Processing /media/rush/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7/260429_0102/common_data.hdf5: 10
Data conversion completed / total demos = 64
(robodiff) rush@rush:~/diffusion_policy$ python3 hdf5_viewer.py 
(robodiff) rush@rush:~/diffusion_policy$ 