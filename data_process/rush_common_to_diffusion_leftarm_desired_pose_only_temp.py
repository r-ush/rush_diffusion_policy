import h5py
import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R
import roboticstoolbox as rtb
from PIL import Image

urdf_path = "/home/rush/diffusion_policy/m0609.white.urdf"
robot = rtb.ERobot.URDF(urdf_path)


""" common data — 각 채널별 step 수가 불일치하는 임시 데이터
data
    demo_0
        observations
            [고주파 채널] F_e_raw, current_pose, desired_pose, joint_torque_L, joint_torque_R  (예: 113~142 steps)
            [저주파 채널] image_H, joint_L, joint_R  (예: 99~122 steps)

해결: 각 demo 내에서 모든 채널의 길이 중 최솟값(target_len)을 구하고,
      선형(또는 slerp) 보간을 통해 모든 채널을 target_len으로 리샘플링 후 저장.
"""

""" diffusion data (desired_pose only)
data
    demo_0
        actions  (desired_pose_9d: trans(3) + 6d_rotation(6))
        obs
            robot_pose_L   # m, (N, 3) - FK from joint_L
            robot_quat_L   # (N, 4) x,y,z,w - FK from joint_L
            image0         # (N, 240, 320, 3)
"""


# ─────────────────────────────────────────────────────────────
# 보간 유틸리티
# ─────────────────────────────────────────────────────────────

def front_trim(arr, target_len):
    """배열 앞단을 잘라 마지막 target_len 개만 반환."""
    if arr.shape[0] <= target_len:
        return arr
    return arr[-target_len:]


# ─────────────────────────────────────────────────────────────
# 회전 변환 유틸리티
# ─────────────────────────────────────────────────────────────

def euler_zyx_deg_to_6d(euler_zyx_deg):
    euler_rad = np.deg2rad(euler_zyx_deg)
    rotmats = R.from_euler('ZYX', euler_rad).as_matrix()  # (N, 3, 3)
    r1 = rotmats[:, :, 0]
    r2 = rotmats[:, :, 1]
    return np.concatenate([r1, r2], axis=1)


def resize_images(image_list, size=(320, 240)):
    resized = []
    for img in image_list:
        pil_img = Image.fromarray(img.astype('uint8'))
        pil_img = pil_img.resize(size, Image.LANCZOS)
        resized.append(np.array(pil_img))
    return np.array(resized)


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────

def main():
    # ========== 변환할 demo 번호 설정 ==========
    # None: 모든 demo 변환
    demo_indices = None
    # demo_indices = [0, 1, 2]
    # ==========================================

    input_filenames = ['/media/rush/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7/260518_0136/common_data.hdf5']
    output_filename = '/media/rush/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7/260518_0136/diffusion_plug_desired_pose_only20_temp.hdf5'

    output_demo_idx = 0

    with h5py.File(output_filename, 'w') as output_file:
        output_data = output_file.create_group('data')

        for input_filename in input_filenames:
            with h5py.File(input_filename, 'r') as input_file:
                input_data = input_file['data']
                demo_len = len(input_data)
                print(f"{input_filename}  /  demo_len = {demo_len}")

                indices_to_process = range(demo_len) if demo_indices is None else demo_indices
                if demo_indices is not None:
                    print(f"Selective processing: {list(demo_indices)}")

                for demo_idx in tqdm.tqdm(indices_to_process, desc=f"Processing {input_filename}"):
                    input_demo_name = f'demo_{demo_idx}'
                    if input_demo_name not in input_data:
                        print(f"⚠️  {input_demo_name} not found, skipping...")
                        continue

                    input_obs = input_data[input_demo_name]['observations']

                    # ── 각 채널 로드 ──
                    raw_joint_L    = np.asarray(input_obs['joint_L'])        # (N_low, 6)
                    raw_image_H    = np.asarray(input_obs['image_H'])        # (N_low, H, W, 3)
                    raw_desired    = np.asarray(input_obs['desired_pose'])   # (N_high, 6)

                    # joint_R, joint_torque_L/R 등은 있을 수도 없을 수도 있으므로 optional 처리
                    optional_keys = ['joint_R', 'joint_torque_L', 'joint_torque_R',
                                     'F_e_raw', 'current_pose']
                    optional_data = {}
                    for k in optional_keys:
                        if k in input_obs:
                            optional_data[k] = np.asarray(input_obs[k])

                    # ── target_len: 모든 채널 중 최솟값 ──
                    all_lens = [raw_joint_L.shape[0],
                                raw_image_H.shape[0],
                                raw_desired.shape[0]]
                    for v in optional_data.values():
                        all_lens.append(v.shape[0])

                    target_len = min(all_lens)
                    print(f"  demo_{demo_idx}: lengths={all_lens}  →  target_len={target_len}")

                    if target_len < 2:
                        print(f"  ⚠️  target_len={target_len} < 2, skipping demo_{demo_idx}")
                        continue

                    # ── 앞단 자르기 (긴 채널의 앞부분을 버리고 마지막 target_len개만 사용) ──
                    joint_L_r  = front_trim(raw_joint_L, target_len)   # (T, 6)
                    image_H_r  = front_trim(raw_image_H, target_len)   # (T, H, W, 3)
                    desired_r  = front_trim(raw_desired, target_len)   # (T, 6)

                    # ── desired_pose → 9D action ──
                    desired_pos_m   = desired_r[:, :3] / 1000.0          # mm → m
                    desired_rot_6d  = euler_zyx_deg_to_6d(desired_r[:, 3:6])
                    desired_pose_9d = np.hstack([desired_pos_m, desired_rot_6d]).astype(np.float32)

                    # ── image resize + BGR→RGB ──
                    image_out = resize_images(image_H_r, (320, 240))     # (T, 240, 320, 3)
                    image_out = image_out[..., ::-1].copy()              # BGR → RGB

                    # ── FK: joint_L → TCP pose, quat ──
                    tcp         = robot.fkine(joint_L_r)
                    tcp_pos     = tcp.t                                   # (T, 3)
                    tcp_quat    = R.from_matrix(tcp.R).as_quat()         # (T, 4) x,y,z,w
                    # w < 0 이면 부호 반전 (연속성 보장)
                    tcp_quat    = np.where(tcp_quat[:, 3:4] < 0, -tcp_quat, tcp_quat)

                    # ── HDF5 저장 (마지막 frame 제외: obs[:-1], actions[1:]) ──
                    output_demo_name = f'demo_{output_demo_idx}'
                    output_demo_n = output_data.create_group(output_demo_name)
                    output_obs    = output_demo_n.create_group('obs')

                    output_obs.create_dataset('robot_pose_L', data=tcp_pos[:-1].astype(np.float32))
                    output_obs.create_dataset('robot_quat_L', data=tcp_quat[:-1].astype(np.float32))
                    output_obs.create_dataset('image0',       data=image_out[:-1])
                    output_demo_n.create_dataset('actions',   data=desired_pose_9d[1:])

                    output_demo_idx += 1

        print(f"\nData conversion completed / total demos = {output_demo_idx}")


if __name__ == "__main__":
    main()
