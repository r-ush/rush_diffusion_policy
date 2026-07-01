"""
새로운 zarr 기반 에피소드 데이터셋 → Diffusion Policy HDF5 변환 스크립트

입력 구조 (에피소드별 폴더):
  <dataset_root>/
    episode_000/
      robot/
        ee_pose_se3.zarr       (N, 4, 4) float64 - EE 실제 pose (SE3 행렬)
        ee_pose_time_stamps.zarr (N,)
        command_pose_se3.zarr  (N+1, 4, 4) float64 - 명령 pose (SE3 행렬)
        command_time_stamps.zarr (N+1,)
      camera_0_D405/
        rgb.zarr               (N, 480, 640, 3) uint8 - RGB 이미지
        rgb_time_stamps.zarr   (N,)
    episode_001/
      ...
    meta.json

출력 HDF5 구조 (Diffusion Policy 학습용):
  data/
    demo_0/
      actions         (T, 9)  - pos(3, m) + rot6d(6)
      obs/
        robot_pose_L  (T, 3)  - TCP 위치 [m]
        robot_quat_L  (T, 4)  - TCP 쿼터니언 [x,y,z,w]
        image0        (T, 240, 320, 3) - RGB 이미지

실행 방법 (robodiff 환경):
  conda activate robodiff
  python data_process/zarr_episodes_to_diffusion_hdf5.py
"""

import os
import json
import zarr
import h5py
import numpy as np
import tqdm
from scipy.spatial.transform import Rotation as R
from PIL import Image

# ============================================================
#  설정값 - 여기서 수정
# ============================================================

DATASET_ROOT = "/home/rush/Desktop/Datasets/20260630_195919"
OUTPUT_HDF5  = "/home/rush/Desktop/Datasets/20260630_195919_diffusion.hdf5"

# 변환할 에피소드 인덱스 (None = 전체)
EPISODE_INDICES = None
# EPISODE_INDICES = [0, 1, 2, 3]  # 일부만 테스트할 때

# 다운샘플 스트라이드: 30Hz 원본 → 10Hz (stride=3)
STRIDE = 3

# 출력 이미지 해상도 (width, height)
IMG_OUT_SIZE = (320, 240)

# 어떤 카메라를 image0으로 사용할지
CAMERA_NAME = "camera_0_D405"  # "camera_1_D435"

# command SE3의 translation 단위가 mm인지 여부
# - True  → /1000 해서 m으로 변환
# - False → 이미 m 단위
# meta.json의 command_position_unit: "mm" 기준으로 True가 기본값
# 실제 값을 확인하려면 스크립트 하단의 verify_units()를 먼저 실행
COMMAND_UNIT_MM = False   # SE3에 이미 m로 저장됨 (ROS 토픽은 mm이었지만 저장 시 변환됨)
EE_UNIT_MM      = False   # ee_pose_se3는 FK 결과로 m 단위

# ============================================================

def se3_to_pos_quat(se3_matrices, unit_mm=False):
    """
    SE3 행렬 (N, 4, 4) → pos (N, 3) [m], quat (N, 4) [x,y,z,w]
    """
    pos = se3_matrices[:, :3, 3].copy()
    if unit_mm:
        pos /= 1000.0

    rotmats = se3_matrices[:, :3, :3]
    quat = R.from_matrix(rotmats).as_quat()  # xyzw

    # w가 항상 양수가 되도록 (연속성 유지)
    quat[quat[:, 3] < 0] *= -1
    return pos, quat


def pos_quat_to_9d(pos, quat):
    """
    pos (N, 3), quat (N, 4) → 9D action (N, 9): pos(3) + rot6d(6)
    """
    rotmats = R.from_quat(quat).as_matrix()  # (N, 3, 3)
    r1 = rotmats[:, :, 0]  # (N, 3) 첫 번째 열
    r2 = rotmats[:, :, 1]  # (N, 3) 두 번째 열
    rot6d = np.concatenate([r1, r2], axis=1)  # (N, 6)
    return np.concatenate([pos, rot6d], axis=1).astype(np.float32)


def se3_to_9d(se3_matrices, unit_mm=False):
    """
    SE3 행렬 (N, 4, 4) → 9D (N, 9)
    """
    pos, quat = se3_to_pos_quat(se3_matrices, unit_mm=unit_mm)
    return pos_quat_to_9d(pos, quat)


def resize_images(images_nhwc, out_size=(320, 240)):
    """
    images_nhwc: (N, H, W, C) uint8
    out_size: (width, height)
    반환: (N, out_h, out_w, C) uint8
    """
    out_w, out_h = out_size
    result = np.empty((len(images_nhwc), out_h, out_w, images_nhwc.shape[3]),
                      dtype=np.uint8)
    for i, img in enumerate(images_nhwc):
        pil = Image.fromarray(img)
        pil = pil.resize(out_size, Image.LANCZOS)
        result[i] = np.array(pil)
    return result


def get_episode_dirs(dataset_root, indices=None):
    """데이터셋 루트에서 episode_XXX 폴더 목록 반환"""
    all_eps = sorted([
        d for d in os.listdir(dataset_root)
        if d.startswith("episode_") and
        os.path.isdir(os.path.join(dataset_root, d))
    ])
    if indices is not None:
        all_eps = [all_eps[i] for i in indices if i < len(all_eps)]
    return all_eps


def verify_units(dataset_root, n_episodes=3):
    """
    처음 N개 에피소드의 translation 값 범위를 출력해서
    mm인지 m인지 확인할 수 있게 해줌.
    학습 전에 한 번 실행해서 COMMAND_UNIT_MM, EE_UNIT_MM 설정을 검증하세요.
    """
    print("=== Unit Verification ===")
    eps = get_episode_dirs(dataset_root)[:n_episodes]
    for ep in eps:
        ep_path = os.path.join(dataset_root, ep)
        ee = zarr.open(os.path.join(ep_path, "robot/ee_pose_se3.zarr"), "r")[:]
        cmd = zarr.open(os.path.join(ep_path, "robot/command_pose_se3.zarr"), "r")[:]

        ee_t = ee[:, :3, 3]
        cmd_t = cmd[:, :3, 3]

        print(f"\n[{ep}]")
        print(f"  ee_pose  translation range: min={ee_t.min():.4f}, max={ee_t.max():.4f}")
        print(f"  command  translation range: min={cmd_t.min():.4f}, max={cmd_t.max():.4f}")
        print("  → 값이 수백 이상이면 mm, 0~2 범위이면 m")


def convert(dataset_root, output_hdf5, episode_indices=None,
            stride=3, img_out_size=(320, 240),
            camera_name="camera_0_D405",
            command_unit_mm=True, ee_unit_mm=False):

    episode_dirs = get_episode_dirs(dataset_root, episode_indices)
    print(f"변환할 에피소드 수: {len(episode_dirs)}")

    with h5py.File(output_hdf5, "w") as out_f:
        out_data = out_f.create_group("data")
        demo_idx = 0

        for ep_name in tqdm.tqdm(episode_dirs, desc="Episodes"):
            ep_path = os.path.join(dataset_root, ep_name)

            # --- 데이터 로드 ---
            ee_se3  = zarr.open(os.path.join(ep_path, "robot/ee_pose_se3.zarr"), "r")[:]
            cmd_se3 = zarr.open(os.path.join(ep_path, "robot/command_pose_se3.zarr"), "r")[:]
            rgb     = zarr.open(os.path.join(ep_path, f"{camera_name}/rgb.zarr"), "r")[:]

            N = min(len(ee_se3), len(rgb))  # 유효 프레임 수 (보통 721)

            # obs: [0, N-1), action: [1, N)  (길이 N-1)
            ee_obs  = ee_se3[:N-1:stride]      # stride 적용 후 obs
            cmd_act = cmd_se3[1:N:stride]      # stride 적용 후 action (한 step 앞)
            rgb_obs = rgb[:N-1:stride]         # stride 적용 후 이미지

            T = min(len(ee_obs), len(cmd_act), len(rgb_obs))
            if T < 2:
                print(f"  {ep_name}: 너무 짧음 ({T} steps), 건너뜀")
                continue

            ee_obs  = ee_obs[:T]
            cmd_act = cmd_act[:T]
            rgb_obs = rgb_obs[:T]

            # --- 변환 ---
            # obs: EE pose → pos(3) + quat(4)
            ee_pos, ee_quat = se3_to_pos_quat(ee_obs, unit_mm=ee_unit_mm)

            # action: command → 9D (pos + rot6d)
            actions = se3_to_9d(cmd_act, unit_mm=command_unit_mm)

            # image: resize
            images = resize_images(rgb_obs, out_size=img_out_size)

            # --- HDF5 저장 ---
            grp = out_data.create_group(f"demo_{demo_idx}")
            obs_grp = grp.create_group("obs")

            obs_grp.create_dataset("robot_pose_L", data=ee_pos.astype(np.float32))
            obs_grp.create_dataset("robot_quat_L", data=ee_quat.astype(np.float32))
            obs_grp.create_dataset("image0",       data=images)
            grp.create_dataset("actions",           data=actions)

            demo_idx += 1

    print(f"\n완료: {demo_idx}개 demo → {output_hdf5}")


if __name__ == "__main__":
    # 1단계: 먼저 단위 확인 (주석 해제해서 실행)
    # verify_units(DATASET_ROOT)

    # 2단계: 변환 실행
    convert(
        dataset_root=DATASET_ROOT,
        output_hdf5=OUTPUT_HDF5,
        episode_indices=EPISODE_INDICES,
        stride=STRIDE,
        img_out_size=IMG_OUT_SIZE,
        camera_name=CAMERA_NAME,
        command_unit_mm=COMMAND_UNIT_MM,
        ee_unit_mm=EE_UNIT_MM,
    )
