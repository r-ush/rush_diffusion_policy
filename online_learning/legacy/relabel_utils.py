"""
replay_buffer(zarr) + videos/ 의 한 에피소드를 achieved-pose relabel 방식으로 단일 demo
HDF5로 변환.

relabel 원리 (data_process/rush_replay_buffer_to_correction_hdf5.py와 동일):
  action 라벨 = "정책이 낸 명령"이 아니라 "한 스텝 뒤 실제 도달한 pose"(robot_pose_L/quat_L).
  사람이 민 결과가 그대로 지도학습 타깃이 된다. F/T 센서 없이도 f/K가 결과 pose에 녹아있음.

이미지 소스에 대한 중요한 사실:
  RightarmRealEnvImp / LeftarmRealEnvImp 의 obs_accumulator는 로봇 lowdim(pose/quat)만
  replay_buffer.zarr에 저장한다. **이미지는 zarr에 없고** <output_dir>/videos/{ep}/{cam}.mp4
  에만 있다. 이 mp4는 record_raw_video=False 경로로 obs 해상도(320×240) RGB를 제어주기
  (frequency fps)로 기록하며, VideoRecorder가 dt 그리드(get_accumulate_timestamp_idxs)에
  맞춰 프레임을 넣으므로 **디코드한 frame k ↔ 에피소드 obs step k 가 1:1로 정렬**된다
  (real_data_conversion.py와 동일한 가정). 따라서 read_video를 enumerate 한 step_idx가
  곧 obs step 인덱스다.
"""
import os

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

from diffusion_policy.real_world.video_recorder import read_video
from diffusion_policy.common.cv2_util import get_image_transform


def pose_quat_to_9d(pos, quat):
    """pos (N,3), quat (N,4)[xyzw] -> 9D (N,9): pos(3)+rot6d(6)"""
    rotmats = R.from_quat(quat).as_matrix()
    r1 = rotmats[:, :, 0]
    r2 = rotmats[:, :, 1]
    rot6d = np.concatenate([r1, r2], axis=1)
    return np.concatenate([pos, rot6d], axis=1).astype(np.float32)


def _extract_lowdim(replay_buffer, ep_index):
    """replay_buffer에서 ep_index 에피소드의 (pose, quat, stage) 슬라이스 추출 (lowdim만)."""
    episode_ends = replay_buffer.episode_ends[:]
    start = 0 if ep_index == 0 else int(episode_ends[ep_index - 1])
    end = int(episode_ends[ep_index])
    pose = replay_buffer['robot_pose_L'][start:end]
    quat = replay_buffer['robot_quat_L'][start:end]
    stage = replay_buffer['stage'][start:end] if 'stage' in replay_buffer.keys() \
        else np.zeros(end - start, dtype=np.int64)
    return pose, quat, stage


def load_episode_frames(env_output_dir, ep_index, n_steps,
                        cam_idx=0, frequency=10.0, out_res=None):
    """videos/{ep_index}/{cam_idx}.mp4 를 디코드해 (M,H,W,3) uint8 RGB 반환.

    frame k ↔ obs step k 로 1:1 정렬(위 모듈 docstring 참고). n_steps 개까지만 읽는다.
    영상 프레임 수 M이 n_steps와 2 프레임 넘게 차이 나면 경고를 출력하고, 호출부에서
    min(M, n_steps)로 자르도록 그대로 반환한다.

    out_res: (width, height). 영상 해상도가 이와 다르면 리사이즈(방어용). None이면 그대로.
    """
    video_path = os.path.join(env_output_dir, "videos", str(ep_index), f"{cam_idx}.mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"에피소드 영상이 없습니다: {video_path} "
            f"(env output_dir/videos/{ep_index}/{cam_idx}.mp4 확인)")

    dt = 1.0 / float(frequency)
    img_transform = None  # 영상은 이미 obs 해상도 RGB. out_res 주면 방어적 리사이즈.
    frames = []
    for step_idx, frame in enumerate(read_video(
            video_path=video_path, dt=dt,
            img_transform=img_transform,
            thread_type='FRAME', thread_count=1)):
        if step_idx >= n_steps:
            break
        if out_res is not None and (frame.shape[1], frame.shape[0]) != tuple(out_res):
            if img_transform is None:
                img_transform = get_image_transform(
                    input_res=(frame.shape[1], frame.shape[0]),
                    output_res=tuple(out_res), bgr_to_rgb=False)
            frame = img_transform(frame)
        frames.append(np.ascontiguousarray(frame, dtype=np.uint8))

    if len(frames) == 0:
        raise RuntimeError(f"영상에서 프레임을 하나도 못 읽음: {video_path}")

    frames = np.stack(frames, axis=0)
    if abs(len(frames) - n_steps) > 2:
        print(f"[relabel][WARN] 영상 프레임 {len(frames)} vs lowdim step {n_steps} "
              f"차이가 큽니다 ({video_path}). min 길이로 자릅니다.")
    return frames


def relabel_episode_to_hdf5(image0, pose, quat, out_path):
    """단일 에피소드 배열 -> 1-demo HDF5 (data/demo_0/...). out_path 반환.

    (GUI 데모/스모크 테스트가 배열을 직접 넘겨 호출하므로 시그니처 유지.)
    """
    T = min(len(pose), len(image0))
    image0 = image0[:T]
    pose = pose[:T]
    quat = quat[:T]

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


def relabel_last_episode_to_hdf5(replay_buffer, env_output_dir, out_path,
                                 cam_idx=0, frequency=10.0, out_res=None):
    """replay_buffer의 가장 마지막 에피소드를 relabel HDF5로 저장.

    replay_buffer: env.replay_buffer (lowdim만 들어있음)
    env_output_dir: env.output_dir (videos/ 를 포함하는 폴더). 이미지는 여기서 디코드.
    """
    ep_index = replay_buffer.n_episodes - 1
    pose, quat, _ = _extract_lowdim(replay_buffer, ep_index)
    T = len(pose)
    frames = load_episode_frames(
        env_output_dir, ep_index, n_steps=T,
        cam_idx=cam_idx, frequency=frequency, out_res=out_res)
    L = min(len(frames), T)
    return relabel_episode_to_hdf5(frames[:L], pose[:L], quat[:L], out_path)


# ============================================================
# 박스삽입(오른팔, wrench) 태스크용 relabel
#   obs 키: image0, robot_pose_R, robot_quat_R, wrench_wrist_R((6,32) 윈도우)
#   action: achieved pose(한 스텝 뒤 robot_pose_R/quat_R)를 9D(pos3+rot6d)로.
#   ⚠ wrench_wrist_R 는 env replay_buffer에 저장된 그대로((T,6,32) 윈도우 obs)를 넣는다.
#      learner의 dataset 로더가 이 shape((6,32))를 obs로 바로 받는지 확인 필요.
# ============================================================
def _extract_lowdim_box(replay_buffer, ep_index):
    episode_ends = replay_buffer.episode_ends[:]
    start = 0 if ep_index == 0 else int(episode_ends[ep_index - 1])
    end = int(episode_ends[ep_index])
    keys = replay_buffer.keys()
    pose = replay_buffer['robot_pose_R'][start:end]
    quat = replay_buffer['robot_quat_R'][start:end]
    wrench = replay_buffer['wrench_wrist_R'][start:end] if 'wrench_wrist_R' in keys else None
    stage = replay_buffer['stage'][start:end] if 'stage' in keys \
        else np.zeros(end - start, dtype=np.int64)
    return pose, quat, wrench, stage


def relabel_last_episode_to_hdf5_box(replay_buffer, env_output_dir, out_path,
                                     cam_idx=0, frequency=10.0, out_res=None):
    """박스삽입 태스크: 마지막 에피소드를 achieved-pose relabel HDF5로 저장.
    obs에 wrench_wrist_R((6,32))도 포함한다."""
    ep_index = replay_buffer.n_episodes - 1
    pose, quat, wrench, _ = _extract_lowdim_box(replay_buffer, ep_index)
    T = len(pose)
    frames = load_episode_frames(
        env_output_dir, ep_index, n_steps=T,
        cam_idx=cam_idx, frequency=frequency, out_res=out_res)
    L = min(len(frames), T)
    image0 = frames[:L]
    pose = pose[:L]
    quat = quat[:L]
    wrench = wrench[:L] if wrench is not None else None

    Tm = min(len(pose), len(image0))
    obs_image0 = image0[:Tm - 1]
    obs_pose = pose[:Tm - 1]
    obs_quat = quat[:Tm - 1]
    obs_wrench = wrench[:Tm - 1] if wrench is not None else None
    achieved_pose = pose[1:Tm]
    achieved_quat = quat[1:Tm]
    actions_9d = pose_quat_to_9d(achieved_pose, achieved_quat)

    if np.issubdtype(obs_image0.dtype, np.floating):
        images_uint8 = np.clip(obs_image0 * 255.0, 0, 255).astype(np.uint8)
    else:
        images_uint8 = obs_image0.astype(np.uint8)

    with h5py.File(out_path, "w") as f:
        data = f.create_group("data")
        grp = data.create_group("demo_0")
        obs = grp.create_group("obs")
        obs.create_dataset("robot_pose_R", data=obs_pose.astype(np.float32))
        obs.create_dataset("robot_quat_R", data=obs_quat.astype(np.float32))
        obs.create_dataset("image0", data=images_uint8)
        if obs_wrench is not None:
            obs.create_dataset("wrench_wrist_R", data=obs_wrench.astype(np.float32))
        grp.create_dataset("actions", data=actions_9d)
    return out_path
