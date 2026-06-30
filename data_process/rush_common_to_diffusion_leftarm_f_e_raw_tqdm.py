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
            image_T   # (640, 480)
            image_L   # (640, 480)
            image_R   # (640, 480)
            F_e_raw   # len=6 (fx, fy, fz, mx, my, mz)
            desired_pose  # len=6 (x,y,z in mm, rz,ry,rx in deg) - ZYX Euler
"""

""" diffusion data, 10Hz
data
    demo_0
        actions (robot_pose_L(3), robot_6d_L(6), desired_pose_9d(9), f_e_raw(6))
        obs
            robot_pose_L   # m, len=3 (x,y,z)
            robot_quat_L   # len=4 (x,y,z,w)
            image0   # (320, 240) - image_H
            f_e_raw  # len=6 (fx,fy,fz,mx,my,mz)
            desired_pose_9d  # len=9 (x,y,z in m + 6d rotation)
"""

def quat_to_6d(quats):
    """
    quats: [[x,y,z,w], [x,y,z,w], ...]  (x,y,z,w 순서)
    return: [[r11,r21,r31,r12,r22,r32], ...] (각각 6D 회전 표현)
    """
    quats = np.asarray(quats)
    rotation_matrix = R.from_quat(quats).as_matrix()  # (N, 3, 3)
    
    # 열 단위로 뽑기
    r1 = rotation_matrix[:, :, 0]  # 첫 번째 column → (N, 3)
    r2 = rotation_matrix[:, :, 1]  # 두 번째 column → (N, 3)
    
    # [r1, r2] 붙이기
    rotation_6d = np.concatenate([r1, r2], axis=1)  # (N, 6)
    return rotation_6d


def rotmat_to_6d(rotmats):
    """
    rotmats: (N, 3, 3) rotation matrices
    return: (N, 6) 6D rotation representation (first two columns of rotation matrix)
    """
    r1 = rotmats[:, :, 0]  # (N, 3)
    r2 = rotmats[:, :, 1]  # (N, 3)
    return np.concatenate([r1, r2], axis=1)  # (N, 6)


def euler_zyx_deg_to_6d(euler_zyx_deg):
    """
    euler_zyx_deg: (N, 3) ZYX Euler angles in degrees
    return: (N, 6) 6D rotation representation
    """
    euler_rad = np.deg2rad(euler_zyx_deg)
    rotmats = R.from_euler('ZYX', euler_rad).as_matrix()  # (N, 3, 3)
    return rotmat_to_6d(rotmats)


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
    # [0, 1, 5]: demo_0, demo_1, demo_5만 변환
    # 예: [0, 1, 2, 3, 4] 또는 list(range(10)) 등
    # demo_indices = None  # <- 여기서 숫자 설정
    demo_indices = [1, 2, 5, 8, 9, 10, 11, 13, 15, 19, 20, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 38, 39, 43, 44, 45, 46, 48, 50, 52, 53, 54, 56, 57, 58, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 82, 83, 85, 86, 87, 88, 90, 93, 94, 96, 98, 99]  # <- 여기서 숫자 설정
    # ==========================================
    
    input_filenames = ['/media/rush/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7/260429_0102/common_data.hdf5']
    output_filename = '/media/rush/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7/260429_0102/diffusion_data_leftarm_f_e_raw.hdf5'
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
                    
                    # n번째 demo 생성
                    input_demo_name = f'demo_{demo_idx}'
                    
                    # 해당 demo가 존재하는지 확인
                    if input_demo_name not in input_data:
                        print(f"⚠️  {input_demo_name} not found, skipping...")
                        continue
                    
                    output_demo_name = f'demo_{output_demo_idx}'
                    
                    input_demo_n = input_data[input_demo_name]
                    output_demo_n = output_data.create_group(output_demo_name)

                    # observations
                    input_obs = input_demo_n['observations']
                    output_obs = output_demo_n.create_group('obs')

                    # 20Hz -> 10Hz
                    input_joint_L = np.asarray(input_obs['joint_L'])[::2]
                    input_image_H = np.asarray(input_obs['image_H'])[::2]
                    timestamp_robot = np.asarray(input_obs['timestamp_robot'])  # (128,) - 20Hz
                    timestamp_wrench = np.asarray(input_obs['timestamp_wrench'])  # (139,) - 250Hz
                    
                    timestamp_robot_10hz = timestamp_robot[::2]  # 10Hz timestamp
                    
                    # F_e_raw는 250Hz이므로, 각 10Hz robot timestamp에서 가장 가까운 wrench를 찾음
                    input_f_e_raw_all = np.asarray(input_obs['F_e_raw'])  # (139, 6)
                    input_f_e_raw = np.zeros((len(timestamp_robot_10hz), 6), dtype=np.float32)

                    # desired_pose 역시 250Hz으로 저장되어 있으면 동일하게 정렬
                    desired_pose_all = np.asarray(input_obs['desired_pose']) if 'desired_pose' in input_obs else None
                    desired_pose_aligned = None  # will hold (N, 6) raw aligned desired_pose
                    for i, ts_robot in enumerate(timestamp_robot_10hz):
                        # robot timestamp에 가장 가까운 wrench/pose index 찾기
                        nearest_idx = np.argmin(np.abs(timestamp_wrench - ts_robot))
                        input_f_e_raw[i] = input_f_e_raw_all[nearest_idx]
                        if desired_pose_all is not None:
                            if desired_pose_aligned is None:
                                desired_pose_aligned = np.zeros((len(timestamp_robot_10hz), 6), dtype=np.float64)
                            desired_pose_aligned[i] = desired_pose_all[nearest_idx]

                    # desired_pose -> 9D (3D trans in m + 6D rotation)
                    desired_pose_9d = None
                    if desired_pose_aligned is not None:
                        # position: mm -> m
                        desired_pos_m = desired_pose_aligned[:, :3] / 1000.0
                        # orientation: ZYX Euler degrees -> 6D rotation
                        desired_euler_zyx_deg = desired_pose_aligned[:, 3:6]
                        desired_rot_6d = euler_zyx_deg_to_6d(desired_euler_zyx_deg)
                        # concat: (N, 3) + (N, 6) -> (N, 9)
                        desired_pose_9d = np.hstack([desired_pos_m, desired_rot_6d]).astype(np.float32)

                    # image 해상도 조정
                    output_image_H = resize_images(input_image_H, (320, 240))

                    # bgr -> rgb
                    output_image_H = np.array(output_image_H)[..., ::-1]
                    output_image_H = list(output_image_H)

                    # joint_L -> pose, quat
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
                    output_obs.create_dataset('f_e_raw', data=input_f_e_raw[:-1])
                    if desired_pose_9d is not None:
                        output_obs.create_dataset('desired_pose_9d', data=desired_pose_9d[:-1])

                    # actions 저장
                    # quat -> 6d rotation
                    output_6d_rotation_L = quat_to_6d(output_TCP_quat_L)

                    # actions = [pose_L(3), 6d_rotation_L(6), desired_pose_9d(9), f_e_raw(6)]
                    parts = [output_TCP_pose_L, output_6d_rotation_L]
                    if desired_pose_9d is not None:
                        parts.append(desired_pose_9d)
                    parts.append(input_f_e_raw)
                    output_actions = np.hstack(parts).tolist()

                    output_demo_n.create_dataset('actions', data=output_actions[1:])

                    output_demo_idx += 1
        
        print(f"Data conversion completed / total demos = {output_demo_idx}")
        

if __name__ == "__main__":
    main()
