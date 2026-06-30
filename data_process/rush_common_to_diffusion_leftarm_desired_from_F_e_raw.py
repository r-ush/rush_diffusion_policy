import h5py
import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R
import roboticstoolbox as rtb
from PIL import Image

urdf_path = "/home/rush/diffusion_policy/m0609.white.urdf"
robot = rtb.ERobot.URDF(urdf_path)   

"""
새로운 데이터 변환 스크립트:
- action: current_pose - F_e_raw 계산 후 9D(pos3 + rot6d) 변환
- obs: image0, robot_pose_L, robot_quat_L
"""

def euler_zyx_deg_to_6d(euler_zyx_deg):
    """
    euler_zyx_deg: (N, 3) ZYX Euler angles in degrees
    return: (N, 6) 6D rotation representation
    """
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
    return resized

def main():
    # ========== 변환할 demo 번호 설정 ==========
    # None: 모든 demo 변환
    # demo_indices = None  # <- 여기서 숫자 설정
    demo_indices = [1, 2, 5, 8, 9, 10, 11, 13, 15, 19, 20, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 38, 39, 43, 44, 45, 46, 48, 50, 52, 53, 54, 56, 57, 58, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 82, 83, 85, 86, 87, 88, 90, 93, 94, 96, 98, 99]
    # ==========================================
    
    # 저장 주기 설정
    save_hz = 20
    
    # Action 생성을 위한 Scale 변수 (F_e_raw 단위를 조정하기 위함)
    # 필요에 따라 이 값을 조절하여 action 스케일을 맞추세요
    pos_scale = 0.001  # 예: 힘(N)을 미터(m) 단위 변위에 맞게 스케일링
    rot_scale = 0.1    # 예: 토크(Nm)를 각도(deg) 단위 변위에 맞게 스케일링
    
    input_filenames = ['/media/rush/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7/260429_0102/common_data.hdf5']
    output_filename = f'/media/rush/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7/260429_0102/diffusion_data_leftarm_desired_from_f_e_raw_{save_hz}.hdf5'
    output_demo_idx = 0

    with h5py.File(output_filename, 'w') as output_file:
        output_data = output_file.create_group('data')

        for input_filename in input_filenames:
            with h5py.File(input_filename, 'r') as input_file:
                input_data = input_file['data']
                demo_len = len(input_data)
                print(input_filename, '/ demo_len =', demo_len)
                
                # 변환할 demo 인덱스 결정
                if demo_indices is None:
                    indices_to_process = range(demo_len)
                else:
                    indices_to_process = demo_indices
                    print(f"Selective processing: demos {indices_to_process}")

                for demo_idx in tqdm.tqdm(indices_to_process, desc=f"Processing {input_filename}"):
                    
                    input_demo_name = f'demo_{demo_idx}'
                    if input_demo_name not in input_data:
                        print(f"⚠️  {input_demo_name} not found, skipping...")
                        continue
                    
                    output_demo_name = f'demo_{output_demo_idx}'
                    
                    input_demo_n = input_data[input_demo_name]
                    output_demo_n = output_data.create_group(output_demo_name)

                    # observations
                    input_obs = input_demo_n['observations']
                    output_obs = output_demo_n.create_group('obs')

                    # 저장 주기에 맞게 샘플 선택
                    robot_stride = 1 if save_hz == 20 else 2
                    input_joint_L = np.asarray(input_obs['joint_L'])[::robot_stride]
                    input_image_H = np.asarray(input_obs['image_H'])[::robot_stride]
                    timestamp_robot = np.asarray(input_obs['timestamp_robot'])
                    timestamp_wrench = np.asarray(input_obs['timestamp_wrench'])
                    
                    timestamp_robot_target = timestamp_robot[::robot_stride]

                    # F_e_raw 정렬 (250Hz -> target Hz nearest)
                    input_f_e_raw_all = np.asarray(input_obs['F_e_raw'])  # (M, 6)
                    input_f_e_raw_aligned = np.zeros((len(timestamp_robot_target), 6), dtype=np.float32)
                    
                    for i, ts_robot in enumerate(timestamp_robot_target):
                        nearest_idx = np.argmin(np.abs(timestamp_wrench - ts_robot))
                        input_f_e_raw_aligned[i] = input_f_e_raw_all[nearest_idx]

                    # image 해상도 조정 (BGR -> RGB)
                    output_image_H = resize_images(input_image_H, (320, 240))
                    output_image_H = np.array(output_image_H)[..., ::-1]
                    output_image_H = list(output_image_H)

                    # Forward Kinematics 로 현재 Pose 계산
                    output_TCP_L = robot.fkine(input_joint_L)
                    output_TCP_pose_L = output_TCP_L.t  # (N, 3) in meters
                    output_TCP_rotmat_L = output_TCP_L.R # (N, 3, 3)
                    
                    output_TCP_quat_L = R.from_matrix(output_TCP_rotmat_L).as_quat()
                    output_TCP_quat_L = np.array([-q if q[3] < 0 else q for q in output_TCP_quat_L])
                    
                    # 현재 회전을 ZYX Euler (degrees)로 변환
                    current_euler_zyx_deg = R.from_matrix(output_TCP_rotmat_L).as_euler('ZYX', degrees=True)

                    # =================================================================
                    # ACTION 계산: action = current_pose - F_e_raw
                    # 1. 3D Position: current_pos_m - (F_xyz * pos_scale)
                    # 2. 3D Rotation: current_euler_deg - (M_xyz * rot_scale)
                    # =================================================================
                    target_pos_m = output_TCP_pose_L - (input_f_e_raw_aligned[:, :3] * pos_scale)
                    target_euler_zyx_deg = current_euler_zyx_deg - (input_f_e_raw_aligned[:, 3:6] * rot_scale)
                    
                    # 9D 변환 (3D Position + 6D Rotation)
                    target_rot_6d = euler_zyx_deg_to_6d(target_euler_zyx_deg)
                    action_9d = np.hstack([target_pos_m, target_rot_6d]).astype(np.float32)

                    # output_obs에 데이터 저장 (마지막 frame 제외)
                    output_obs.create_dataset('robot_pose_L', data=output_TCP_pose_L[:-1])
                    output_obs.create_dataset('robot_quat_L', data=output_TCP_quat_L[:-1])
                    output_obs.create_dataset('image0', data=output_image_H[:-1])

                    # actions 저장: (마지막 frame 제외를 위해 데이터 한 칸 당김)
                    output_demo_n.create_dataset('actions', data=action_9d[1:])

                    output_demo_idx += 1
        
        print(f"Data conversion completed / total demos = {output_demo_idx}")

if __name__ == "__main__":
    main()
