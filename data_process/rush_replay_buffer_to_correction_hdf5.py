"""
rush_eval_real_robot_imp.py로 수집한 correction 에피소드
(<output_dir>/replay_buffer.zarr, LeftarmRealEnvImp 포맷)를
학습용 HDF5(rush_logistic_box_pose_only.yaml과 동일 포맷)로 변환한다.

핵심 아이디어 (hindsight relabeling):
  원래 저장된 `action`은 "정책이 그 시점에 낸 명령"이다. 사람이 로봇을 물리적으로
  밀어 교정하면, 명령과 실제로 도달한 pose(robot_pose_L/robot_quat_L)가 달라진다.
  따라서 action 라벨을 원래 명령이 아니라 "한 스텝 뒤에 실제로 측정된 pose"로
  바꿔치기(relabel)하면, 정책은 "사람이 교정한 대로" 행동하도록 학습된다.
  (사람이 개입하지 않은 구간은 명령 ≈ 실제 pose이므로 relabel해도 기존 학습과
  거의 동일한 타깃이 나온다 — 즉 이 변환은 correction 여부와 무관하게 항상 적용
  가능한 일관된 방식이다.)

  correction 여부(rush_eval_real_robot_imp.py에서 'C' 키로 토글, `stage` 필드)가
  1인 스텝이 하나라도 포함된 에피소드는, --oversample 배수만큼 통째로 복제해서
  출력 HDF5에 더 많이 등장시킬 수 있다 (기존 dataset 클래스 코드를 건드리지 않고
  샘플링 비중을 높이는 가장 간단한 방법).

입력:
  <replay_buffer_root>/replay_buffer.zarr
    keys: image0 (T,H,W,3) float32 [0,1], robot_pose_L (T,3), robot_quat_L (T,4,)
          action (T,9), stage (T,), timestamp (T,), meta/episode_ends

출력 HDF5:
  data/demo_i/actions          (T-1, 9)  pos(3,m) + rot6d(6), achieved-pose 기준
  data/demo_i/obs/robot_pose_L (T-1, 3)
  data/demo_i/obs/robot_quat_L (T-1, 4)
  data/demo_i/obs/image0       (T-1, H, W, 3) uint8

사용법:
  conda activate robodiff
  python data_process/rush_replay_buffer_to_correction_hdf5.py \
      --input /home/rush/data/results/replay_buffer.zarr \
      --output /home/rush/Desktop/Datasets/correction_batch1.hdf5 \
      --oversample 3
"""

import argparse
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

from diffusion_policy.common.replay_buffer import ReplayBuffer


def pose_quat_to_9d(pos, quat):
    """pos (N,3), quat (N,4) [x,y,z,w] -> 9D action (N,9): pos(3) + rot6d(6)"""
    rotmats = R.from_quat(quat).as_matrix()  # (N, 3, 3)
    r1 = rotmats[:, :, 0]
    r2 = rotmats[:, :, 1]
    rot6d = np.concatenate([r1, r2], axis=1)
    return np.concatenate([pos, rot6d], axis=1).astype(np.float32)


def convert(input_zarr, output_hdf5, oversample=1, min_episode_len=3):
    buffer = ReplayBuffer.create_from_path(input_zarr, mode='r')
    print(f"입력: {input_zarr}")
    print(f"에피소드 수: {buffer.n_episodes}, 총 스텝 수: {buffer.n_steps}")

    episode_ends = buffer.episode_ends[:]
    episode_starts = np.concatenate([[0], episode_ends[:-1]])

    image0 = buffer['image0']
    robot_pose_L = buffer['robot_pose_L']
    robot_quat_L = buffer['robot_quat_L']
    stage = buffer['stage'] if 'stage' in buffer.keys() else None
    if stage is None:
        print("[WARN] 'stage' 필드가 없음 (구버전 데이터). correction=0으로 간주합니다.")

    with h5py.File(output_hdf5, 'w') as out_f:
        out_data = out_f.create_group('data')
        demo_idx = 0
        n_correction_episodes = 0

        for ep_i, (start, end) in enumerate(zip(episode_starts, episode_ends)):
            T = end - start
            if T < min_episode_len:
                print(f"  episode {ep_i}: 너무 짧음 ({T} steps), 건너뜀")
                continue

            ep_image0 = image0[start:end]
            ep_pose = robot_pose_L[start:end]
            ep_quat = robot_quat_L[start:end]
            ep_stage = stage[start:end] if stage is not None else np.zeros(T, dtype=np.int64)

            # obs: [0, T-1), action(achieved pose): [1, T)  -- 한 스텝 앞의 실제 도달 pose
            obs_image0 = ep_image0[:T - 1]
            obs_pose = ep_pose[:T - 1]
            obs_quat = ep_quat[:T - 1]
            achieved_pose = ep_pose[1:T]
            achieved_quat = ep_quat[1:T]
            action_correction_flag = ep_stage[1:T]

            actions_9d = pose_quat_to_9d(achieved_pose, achieved_quat)

            # image0가 [0,1] float으로 저장되어 있으면 uint8로 되돌림
            if np.issubdtype(obs_image0.dtype, np.floating):
                images_uint8 = np.clip(obs_image0 * 255.0, 0, 255).astype(np.uint8)
            else:
                images_uint8 = obs_image0.astype(np.uint8)

            has_correction = bool(np.any(action_correction_flag))
            n_copies = oversample if has_correction else 1
            if has_correction:
                n_correction_episodes += 1

            for _ in range(n_copies):
                grp = out_data.create_group(f"demo_{demo_idx}")
                obs_grp = grp.create_group("obs")
                obs_grp.create_dataset("robot_pose_L", data=obs_pose.astype(np.float32))
                obs_grp.create_dataset("robot_quat_L", data=obs_quat.astype(np.float32))
                obs_grp.create_dataset("image0", data=images_uint8)
                grp.create_dataset("actions", data=actions_9d)
                # 참고용 메타데이터 (학습 코드는 사용하지 않음, 나중에 분석용)
                grp.create_dataset("is_correction", data=action_correction_flag.astype(np.int64))
                demo_idx += 1

            print(f"  episode {ep_i}: {T} steps, correction={has_correction}, "
                  f"copies={n_copies}")

    print(f"\n완료: {demo_idx}개 demo 저장 (원본 에피소드 중 correction 포함: "
          f"{n_correction_episodes}) -> {output_hdf5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="replay_buffer.zarr 경로")
    parser.add_argument("--output", required=True, help="출력 HDF5 경로")
    parser.add_argument("--oversample", type=int, default=1,
                         help="correction이 포함된 에피소드를 몇 배 복제할지")
    parser.add_argument("--min_episode_len", type=int, default=3)
    args = parser.parse_args()

    convert(args.input, args.output, oversample=args.oversample,
            min_episode_len=args.min_episode_len)
