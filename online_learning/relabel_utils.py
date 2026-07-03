"""
replay_buffer(zarr)의 한 에피소드를 achieved-pose relabel 방식으로 단일 demo HDF5로 변환.

relabel 원리 (data_process/rush_replay_buffer_to_correction_hdf5.py와 동일):
  action 라벨 = "정책이 낸 명령"이 아니라 "한 스텝 뒤 실제 도달한 pose"(robot_pose_L/quat_L).
  사람이 민 결과가 그대로 지도학습 타깃이 된다. F/T 센서 없이도 f/K가 결과 pose에 녹아있음.
"""
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R


def pose_quat_to_9d(pos, quat):
    """pos (N,3), quat (N,4)[xyzw] -> 9D (N,9): pos(3)+rot6d(6)"""
    rotmats = R.from_quat(quat).as_matrix()
    r1 = rotmats[:, :, 0]
    r2 = rotmats[:, :, 1]
    rot6d = np.concatenate([r1, r2], axis=1)
    return np.concatenate([pos, rot6d], axis=1).astype(np.float32)


def _extract_episode_arrays(replay_buffer, ep_index):
    """replay_buffer에서 ep_index 에피소드의 (image0, pose, quat, stage) 슬라이스 추출."""
    episode_ends = replay_buffer.episode_ends[:]
    start = 0 if ep_index == 0 else int(episode_ends[ep_index - 1])
    end = int(episode_ends[ep_index])
    image0 = replay_buffer['image0'][start:end]
    pose = replay_buffer['robot_pose_L'][start:end]
    quat = replay_buffer['robot_quat_L'][start:end]
    stage = replay_buffer['stage'][start:end] if 'stage' in replay_buffer.keys() \
        else np.zeros(end - start, dtype=np.int64)
    return image0, pose, quat, stage


def relabel_episode_to_hdf5(image0, pose, quat, out_path):
    """단일 에피소드 배열 -> 1-demo HDF5 (data/demo_0/...). out_path 반환."""
    T = len(pose)
    obs_image0 = image0[:T - 1]
    obs_pose = pose[:T - 1]
    obs_quat = quat[:T - 1]
    achieved_pose = pose[1:T]
    achieved_quat = quat[1:T]
    actions_9d = pose_quat_to_9d(achieved_pose, achieved_quat)

    if np.issubdtype(obs_image0.dtype, np.floating):
        images_uint8 = np.clip(obs_image0 * 255.0, 0, 255).astype(np.uint8)
    else:
        images_uint8 = obs_image0.astype(np.uint8)

    with h5py.File(out_path, "w") as f:
        data = f.create_group("data")
        grp = data.create_group("demo_0")
        obs = grp.create_group("obs")
        obs.create_dataset("robot_pose_L", data=obs_pose.astype(np.float32))
        obs.create_dataset("robot_quat_L", data=obs_quat.astype(np.float32))
        obs.create_dataset("image0", data=images_uint8)
        grp.create_dataset("actions", data=actions_9d)
    return out_path


def relabel_last_episode_to_hdf5(replay_buffer, out_path):
    """replay_buffer의 가장 마지막 에피소드를 relabel HDF5로 저장."""
    ep_index = replay_buffer.n_episodes - 1
    image0, pose, quat, _ = _extract_episode_arrays(replay_buffer, ep_index)
    return relabel_episode_to_hdf5(image0, pose, quat, out_path)
