import h5py
import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R
import roboticstoolbox as rtb
from roboticstoolbox import ERobot
from spatialmath import SE3, UnitQuaternion
from PIL import Image

urdf_path = "/home/rush/diffusion_policy/m0609.white.urdf"
robot = rtb.ERobot.URDF(urdf_path)   


""" common data, 20Hz + 250Hz wrench
data
    demo_0
        observations
            joint_L   # rad, len=6
            image_H   # (640, 480)   
            desired_pose  # len=6 (x,y,z in mm, rz,ry,rx in deg) - ZYX Euler
"""

""" diffusion data (desired_pose only), 10Hz
data
    demo_0
        actions (desired_pose_9d(9))  # 9D: trans(3) + 6d_rotation(6)
        obs
            robot_pose_L   # m, len=3 (x,y,z) - FK from joint_L
            robot_quat_L   # len=4 (x,y,z,w) - FK from joint_L
            image0   # (320, 240) - image_H
"""

def quat_to_6d(quats):
    """
    quats: [[x,y,z,w], [x,y,z,w], ...]  (x,y,z,w 순서)
    return: [[r11,r21,r31,r12,r22,r32], ...] (각각 6D 회전 표현)
    """
    quats = np.asarray(quats)
    rotation_matrix = R.from_quat(quats).as_matrix()  # (N, 3, 3)
    r1 = rotation_matrix[:, :, 0]
    r2 = rotation_matrix[:, :, 1]
    rotation_6d = np.concatenate([r1, r2], axis=1)  # (N, 6)
    return rotation_6d


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
    """
    image_list : [img1, img2, ...] (각 img는 numpy array, shape (480,640,3))
    size       : (width, height)
    return     : [resized_img1, resized_img2, ...] (모두 (240,320,3))
    """
    resized = []
    for img in image_list:
        pil_img = Image.fromarray(img.astype('uint8'))
        pil_img = pil_img.resize(size, Image.LANCZOS)
        resized.append(np.array(pil_img))
    return resized


def main():
    # ========== 변환할 demo 번호 설정 ==========
    # None: 모든 demo 변환
    demo_indices = None  # <- 여기서 숫자 설정
    # demo_indices = [1, 2, 5, 8, 9, 10, 11, 13, 15, 19, 20, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 38, 39, 43, 44, 45, 46, 48, 50, 52, 53, 54, 56, 57, 58, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 82, 83, 85, 86, 87, 88, 90, 93, 94, 96, 98, 99]
    # demo_indices = [0, 1, 2, 3]
    # ==========================================

    # 저장 주기 설정
    # 20이면 원본 robot timestamp를 그대로 사용
    # 10이면 20Hz -> 10Hz로 절반 다운샘플
    save_hz = 20
    
    # input_filenames = ['/media/rush/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7/260429_0102/common_data.hdf5']
    # output_filename = '/media/rush/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7/260429_0102/diffusion_data_leftarm_desired_pose_only20.hdf5'
    input_filenames = ['/media/rush/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7/260518_0136/common_data.hdf5']
    output_filename = '/media/rush/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7/260518_0136/diffusion_plug_desired_pose_only20_2.hdf5'

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

                    # 저장 주기에 맞게 로봇/이미지 샘플 선택
                    robot_stride = 1 if save_hz == 20 else 2
                    input_joint_L = np.asarray(input_obs['joint_L'])[::robot_stride]
                    input_image_H = np.asarray(input_obs['image_H'])[::robot_stride]
                    timestamp_robot = np.asarray(input_obs['timestamp_robot'])  # 20Hz
                    timestamp_wrench = np.asarray(input_obs['timestamp_wrench'])  # 250Hz

                    timestamp_robot_target = timestamp_robot[::robot_stride]

                    # desired_pose 정렬 (250Hz -> 10Hz nearest)
                    desired_pose_all = np.asarray(input_obs['desired_pose'])  # (M, 6)
                    desired_pose_aligned = np.zeros((len(timestamp_robot_target), 6), dtype=np.float64)
                    for i, ts_robot in enumerate(timestamp_robot_target):
                        nearest_idx = np.argmin(np.abs(timestamp_wrench - ts_robot))
                        desired_pose_aligned[i] = desired_pose_all[nearest_idx]

                    # desired_pose -> 9D (3D trans in m + 6D rotation)
                    desired_pos_m = desired_pose_aligned[:, :3] / 1000.0  # mm -> m
                    desired_euler_zyx_deg = desired_pose_aligned[:, 3:6]
                    desired_rot_6d = euler_zyx_deg_to_6d(desired_euler_zyx_deg)
                    desired_pose_9d = np.hstack([desired_pos_m, desired_rot_6d]).astype(np.float32)

                    # image 해상도 조정
                    output_image_H = resize_images(input_image_H, (320, 240))

                    # bgr -> rgb
                    output_image_H = np.array(output_image_H)[..., ::-1]
                    output_image_H = list(output_image_H)

                    # joint_L -> pose, quat (FK)
                    output_TCP_L = robot.fkine(input_joint_L)
                    output_TCP_pose_L = output_TCP_L.t
                    output_TCP_rotmat_L = output_TCP_L.R
                    output_TCP_quat_L = R.from_matrix(output_TCP_rotmat_L).as_quat()
                    
                    # quaternion w가 양수가 되도록 변경
                    output_TCP_quat_L = np.array([-q if q[3] < 0 else q for q in output_TCP_quat_L])

                    # output_obs에 데이터 저장 (마지막 frame 제외)
                    output_obs.create_dataset('robot_pose_L', data=output_TCP_pose_L[:-1])
                    output_obs.create_dataset('robot_quat_L', data=output_TCP_quat_L[:-1])
                    output_obs.create_dataset('image0', data=output_image_H[:-1])

                    # actions 저장: desired_pose_9d(9) only
                    output_demo_n.create_dataset('actions', data=desired_pose_9d[1:])

                    output_demo_idx += 1
        
        print(f"Data conversion completed / total demos = {output_demo_idx}")
        

if __name__ == "__main__":
    main()
