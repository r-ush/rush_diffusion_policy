#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import rclpy
from rclpy.node import Node
import os
import time
import cv2
import signal
import numpy as np
from scipy.spatial.transform import Rotation as R

# ROS2 messages
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import JointState
from std_msgs.msg import UInt8

# Spatial math
from spatialmath import SE3
import spatialmath.base as smb
import roboticstoolbox as rtb

# Global variables
running = True
teleop_enabled = False  # 외부 제어 신호 상태
moving_to_preset = False  # 사전 정의된 위치로 이동 중인지 상태
preset_target_joints = None  # 목표 조인트 값
movement_speed = 0.02  # 조인트 이동 속도 (rad/s)
random_enabled = False  # 랜덤 모드 활성화 상태
random_offset = None  # 현재 데모의 랜덤 오프셋

# Left arm variables
T_tracker_init_left = None
T_station2track_left = None
current_joints_rad_left = None
current_pose_se3_left = None
robot_pose_init_left = None

robot = None
node = None

# 추가
################################################################################
# base_link in W (회전 = I)
T_base_link = SE3(0.0, 0.0, 0.3055)

# left_base_link in W (주의: q2r는 [w,x,y,z] 순서)
# R_left_base_link = smb.q2r([0.000304743, 0.000735709, 0.923878, 0.382685])
# t_left_base_link = np.array([0.0, 0.25, 0.52])
# T_left_base_link = SE3.Rt(R_left_base_link, t_left_base_link)

# 쿼터니언: w,x,y,z (사용자 제공값)
qL_wxyz = np.array([0.000304743, 0.000735709, 0.923878, 0.382685], dtype=float)
# SciPy는 [x, y, z, w] 순서를 받음
R_left_mat = R.from_quat([qL_wxyz[1], qL_wxyz[2], qL_wxyz[3], qL_wxyz[0]]).as_matrix()
# sanity check (선택)
# assert R_left_mat.shape == (3, 3)
# assert np.allclose(R_left_mat @ R_left_mat.T, np.eye(3), atol=1e-6)
# assert abs(np.linalg.det(R_left_mat) - 1.0) < 1e-6
# 위치
t_left_base_link = np.array([0.0, 0.25, 0.52], dtype=float)
# 3x3 회전행렬을 직접 전달
T_left_base_link = SE3.Rt(R_left_mat, t_left_base_link)


# left -> base
T_left_to_base = T_base_link.inv() * T_left_base_link
print("T_left_to_base.R =\n", T_left_to_base.R)
print("T_left_to_base.t =", T_left_to_base.t)


# Safety state (for rotation_safety_lock)
angle_lock = np.array([[False, False, False],   # from pos->neg
                       [False, False, False]])  # from neg->pos
locked_value = np.zeros(3)
# Base Frame 기준 position 값으로 limits 설정 (robot base frame, meters) <- 수정 필요
# XB_MIN, XB_MAX = 0.20,  0.80
# YB_MIN, YB_MAX = 0.05,  0.60
# ZB_MIN, ZB_MAX = -0.20,  0.20
XB_MIN, XB_MAX = 0.35,  0.8
YB_MIN, YB_MAX = -0.1,  0.85
ZB_MIN, ZB_MAX = -0.2,  0.2
################################################################################


# VR tracker target frame ID
TARGET_FRAME_LEFT = "vive_tracker_LHR_331800F6"

# Calibration matrix - transformation from robot base to tracking station
T_robot2station_left = SE3.CopyFrom(
    np.array([[-0.65021182,  0.10097476, -0.75301308, -1.66658299],
              [ 0.59292999,  0.68714443, -0.41984111, -0.18493201],
              [ 0.47503539, -0.71946968, -0.50666039, -1.58759511],
              [ 0.        ,  0.        ,  0.        ,  1.        ]]),
    
    check=False
)

# Control parameters
Kp_pos = 1.0  # Position gain
Kp_rot = 0.8  # Rotation gain
step_gain = 0.5   # Step size for joint updates
acc_limit = 30.0  # Acceleration limit

PRESET_JOINT_POSITIONS = np.array([
    # 2.0, -48.0, -112.0, -12.0, 75.0, -50.0
    2.0, -48.0, -112.0, -12.0, 75.0, 0.0
])
PRESET_JOINT_POSITIONS = np.deg2rad(PRESET_JOINT_POSITIONS)

def generate_random_offset():
    """매 데모마다 새로운 랜덤 오프셋 생성 (degrees 단위에서 radians로 변환)"""
    # 각 조인트에 대해 ±2도 범위의 랜덤 오프셋 생성
    random_deg = np.random.uniform(-2.0, 2.0, 5)
    random_deg = np.append(random_deg, np.random.uniform(-10.0, 10.0))  # larger_randomness to 6th joint
    print(f"Left Arm: Moving to PRESET position with random offset: {random_deg} deg")

    return np.deg2rad(random_deg)

def signal_handler(sig, frame):
    global running
    print('\nLeft Arm: Program termination requested.')
    running = False

signal.signal(signal.SIGINT, signal_handler)

def random_control_callback(msg):
    """랜덤 제어 신호를 받는 콜백 함수"""
    global random_enabled
    
    if msg.data == 1:
        random_enabled = True
        print("Left Arm: Random mode ENABLED")
        
    elif msg.data == 0:
        random_enabled = False
        print("Left Arm: Random mode DISABLED")
        
    else:
        print(f"Left Arm: Invalid random control signal: {msg.data} (expected 0 or 1)")

def teleop_control_callback(msg):
    """외부 제어 신호를 받는 콜백 함수"""
    print("Left Arm: Received teleop control signal:", msg.data)
    global teleop_enabled, T_tracker_init_left, robot_pose_init_left
    global T_station2track_left, current_pose_se3_left
    global moving_to_preset, preset_target_joints
    
    prev_teleop_enabled = teleop_enabled
    
    if msg.data == 1:
        teleop_enabled = True
        moving_to_preset = False  # VR 제어 모드로 전환 시 preset 이동 중지
        print("Left Arm: Teleoperation ENABLED")
        
        # teleop이 비활성화 상태에서 활성화로 전환되는 경우
        if not prev_teleop_enabled:
            # 현재 로봇 포즈와 VR 트래커 포즈를 새로운 초기값으로 설정
            if current_pose_se3_left is not None and T_station2track_left is not None:
                robot_pose_init_left = current_pose_se3_left.copy()
                T_tracker_init_left = T_station2track_left.copy()
                print("Left Arm: Re-initialized reference poses for teleoperation")
                print(f"Left Arm: New initial tracker pose: {T_tracker_init_left.t}")
                print(f"Left Arm: New initial robot pose: {robot_pose_init_left.t}")
            else:
                print("Left Arm: Warning - Cannot re-initialize poses (missing current data)")
                
    elif msg.data == 0:
        teleop_enabled = False
        moving_to_preset = False  # 정지 모드로 전환 시 preset 이동 중지
        print("Left Arm: Teleoperation DISABLED")
        
    elif msg.data == 2:
        teleop_enabled = False
        moving_to_preset = True
        preset_target_joints = PRESET_JOINT_POSITIONS.copy()
        
        # 랜덤 모드가 활성화되어 있으면 오프셋 적용
        if random_enabled:
            random_offset = generate_random_offset()
            preset_target_joints += random_offset
        else:
            print("Left Arm: Moving to PRESET position")
        
    else:
        print(f"Left Arm: Invalid control signal: {msg.data} (expected 0, 1, or 2)")

def vr_pose_callback(msg):
    """Callback function for VR tracker pose messages"""
    global T_station2track_left, T_tracker_init_left, robot_pose_init_left, current_pose_se3_left, teleop_enabled
    
    for transform in msg.transforms:
        # Left tracker
        if transform.child_frame_id == TARGET_FRAME_LEFT:
            trans = transform.transform.translation
            rot = transform.transform.rotation
            pos = np.array([trans.x, trans.y, trans.z])
            quat = [rot.w, rot.x, rot.y, rot.z]
            R_matrix = smb.q2r(quat)
            T_station2track_left = SE3.Rt(R_matrix, pos)
            
            # Store initial tracker pose and robot pose for left arm (최초 1회만)
            if T_tracker_init_left is None and current_pose_se3_left is not None:
                T_tracker_init_left = T_station2track_left.copy()
                robot_pose_init_left = current_pose_se3_left.copy()
                print("Left Arm: Initial VR tracker and robot poses stored.")
                print(f"Left Arm: Initial tracker pose: {T_tracker_init_left.t}")
                print(f"Left Arm: Initial robot pose: {robot_pose_init_left.t}")
            
            # teleoperation이 비활성화된 경우 초기화된 값들을 사용하지 않음
            if not teleop_enabled:
                return

def joint_state_callback(msg):
    """Callback for joint state updates"""
    global current_joints_rad_left, current_pose_se3_left, robot
    
    if not msg.name or not msg.position:
        return
        
    left_joint_positions = [0.0] * 6  # Initialize with zeros for 6 joints  
    left_joint_names = []
    
    # Filter and sort joints for left arm
    for i, name in enumerate(msg.name):
        if i < len(msg.position) and not np.isnan(msg.position[i]):
            # Left arm joints
            if name.startswith('left_joint'):
                try:
                    joint_num = int(name.split('_')[-1])
                    if 1 <= joint_num <= 6:
                        left_joint_positions[joint_num - 1] = msg.position[i]
                        left_joint_names.append(name)
                except (ValueError, IndexError):
                    continue
    
    # Update left arm joint positions and pose
    if len(left_joint_names) >= 6:
        current_joints_rad_left = np.array(left_joint_positions)
        
        # Calculate left arm current pose using forward kinematics
        if robot is not None:
            try:
                current_pose_se3_left = robot.fkine(current_joints_rad_left, end=robot.links[7])
            except Exception as e:
                try:
                    current_pose_se3_left = robot.fkine(current_joints_rad_left, end=robot.links[5])
                except Exception as e2:
                    print(f"Left Arm: Forward kinematics failed: {e2}")

# 추가
################################################################################
# Rotation Safety
############################################################
def rotation_safety_lock(delta_rot_rad, threshold_deg=178.0):
    global angle_lock
    global locked_value
    max_rad = np.deg2rad(threshold_deg)
    min_rad = -max_rad
    margin = np.deg2rad(1.0)
    delta_rot_out = delta_rot_rad.copy()

    for i in range(3):
        raw = delta_rot_rad[i]

        if angle_lock[0][i]:

            if max_rad - margin > raw > 0: # 0 ~ 177
                angle_lock[0][i] = False
            else:
                delta_rot_out[i] = locked_value[i]
                continue

        elif angle_lock[1][i]:
            if min_rad + margin < raw < 0: # -177 ~ 0
                angle_lock[1][i] = False
            else:
                delta_rot_out[i] = locked_value[i]
                continue

        if raw >= max_rad: # 177 ~
            angle_lock[0][i] = True
            locked_value[i] = max_rad
            delta_rot_out[i] = max_rad

        elif raw <= min_rad: # ~ -177
            angle_lock[1][i] = True
            locked_value[i] = min_rad
            delta_rot_out[i] = min_rad

    return delta_rot_out

############################################################

# Position Safety
############################################################
# def clip_pose(pose, xlim, ylim, zlim):
#     t = pose.t.copy()
#     t[0] = np.clip(t[0], xlim[0], xlim[1])
#     t[1] = np.clip(t[1], ylim[0], ylim[1])
#     t[2] = np.clip(t[2], zlim[0], zlim[1])
#     return SE3.Rt(pose.R, t)


def clip_pose_in_frame(
    pose_left: SE3,
    xlim_base, ylim_base, zlim_base,
    T_left_to_base: SE3
):
    """
    pose_left: left_base_link 기준 EE 목표 포즈
    x/y/zlim_base: base_link 기준 [min, max]
    T_left_to_base: left_base_link -> base_link
    """
    # left -> base (위치 변환)
    p_left = pose_left.t
    p_base_raw = T_left_to_base.R @ p_left + T_left_to_base.t

    # base 프레임에서 축별 클리핑
    p_base = p_base_raw.copy()
    p_base[0] = np.clip(p_base[0], xlim_base[0], xlim_base[1])
    p_base[1] = np.clip(p_base[1], ylim_base[0], ylim_base[1])
    p_base[2] = np.clip(p_base[2], zlim_base[0], zlim_base[1])

    # 디버그: 클립 여부 로그
    if not np.allclose(p_base_raw, p_base, atol=1e-9):
        print(f"[DEBUG-CLIP] base raw {p_base_raw} -> clipped {p_base} | "
              f"X{ xlim_base }, Y{ ylim_base }, Z{ zlim_base }")


    # base -> left 복원
    T_base_to_left = T_left_to_base.inv()
    p_left_clipped = T_base_to_left.R @ (p_base - T_left_to_base.t)

    # 자세는 유지, 위치만 교체
    return SE3.Rt(pose_left.R, p_left_clipped)
############################################################
################################################################################


def calculate_target_pose_left():
    """Calculate target pose for LEFT arm based on VR tracker movement"""
    global T_tracker_init_left, T_station2track_left, T_robot2station_left, robot_pose_init_left
    
    if T_tracker_init_left is None or T_station2track_left is None or robot_pose_init_left is None:
        return None
    
    # Calculate tracker movement from initial position
    T_tracker_delta = T_tracker_init_left.inv() * T_station2track_left
    delta_vr_pos = T_tracker_delta.t  # Relative position in meters
    delta_vr_rot = smb.tr2rpy(T_tracker_delta.R, unit='rad')  # Relative rotation in radians
    
    # 추가
    ################################################################################
    delta_vr_rot = rotation_safety_lock(delta_vr_rot) # Rotation safety lock
    ################################################################################


    # Transform deltas using proper frame transformations
    R_base2station = T_robot2station_left.R
    R_tracker_init = T_tracker_init_left.R
    R_end2base_init = robot_pose_init_left.inv().R
    
    # Transform deltas to end-effector frame
    delta_pos_end = R_end2base_init @ R_base2station @ R_tracker_init @ delta_vr_pos
    delta_rot_end = R_end2base_init @ R_base2station @ R_tracker_init @ delta_vr_rot
    
    # Apply scaling
    scaled_delta_pos = Kp_pos * delta_pos_end
    scaled_delta_rot = Kp_rot * delta_rot_end
    
    # Calculate target pose
    target_pose = robot_pose_init_left * SE3(scaled_delta_pos) * SE3.RPY(*scaled_delta_rot, unit='rad')
    
    # 추가
    ################################################################################
    # Clip target pose to limits
    # target_pose = clip_pose(target_pose, (X_MIN, X_MAX), (Y_MIN, Y_MAX), (Z_MIN, Z_MAX))
    target_pose = clip_pose_in_frame(target_pose, 
                                    (XB_MIN, XB_MAX), (YB_MIN, YB_MAX), (ZB_MIN, ZB_MAX),
                                    T_left_to_base)
    ################################################################################

    return target_pose

def calculate_joint_delta_left(target_pose):
    """Calculate joint delta for LEFT arm using Jacobian pseudoinverse"""
    global robot, current_joints_rad_left, current_pose_se3_left
    
    if current_pose_se3_left is None or robot is None:
        return None
    
    # Calculate pose error
    pose_error = current_pose_se3_left.inv() * target_pose
    
    # Extract position and rotation errors
    err_pos = target_pose.t - current_pose_se3_left.t
    err_rot_ee = smb.tr2rpy(pose_error.A, unit='rad')
    err_rot_base = current_pose_se3_left.R @ err_rot_ee
    # print(f"Left Arm: Position error: {err_pos}, Rotation error: {err_rot_base}")
    
    # Combine into 6D error vector
    err_6d = np.concatenate((err_pos, err_rot_base))
    
    # Compute Jacobian and pseudo-inverse solution
    try:
        J = robot.jacob0(current_joints_rad_left, end=robot.links[7])
    except:
        try:
            J = robot.jacob0(current_joints_rad_left, end=robot.links[5])
        except Exception as e:
            print(f"Left Arm: Jacobian calculation failed: {e}")
            return None
    
    # Calculate joint delta using pseudoinverse
    dq = np.linalg.pinv(J) @ err_6d
    
    # Apply acceleration limits for all joints
    if np.linalg.norm(dq) > acc_limit:
        dq *= acc_limit / np.linalg.norm(dq)
    
    # Additional safety: limit individual joint velocities
    max_joint_vel = np.deg2rad(20.0)  # 20 deg/s max per joint
    dq = np.clip(dq, -max_joint_vel, max_joint_vel)
    
    return dq

def calculate_preset_movement_left():
    """사전 정의된 위치로 천천히 이동하는 조인트 값 계산"""
    global current_joints_rad_left, preset_target_joints, movement_speed, moving_to_preset
    
    if current_joints_rad_left is None or preset_target_joints is None:
        return None
    
    # 현재 조인트와 목표 조인트 사이의 차이 계산
    joint_diff = preset_target_joints - current_joints_rad_left
    
    # 각 조인트별로 최대 이동 거리 제한 (천천히 이동)
    max_movement = movement_speed  # rad per iteration
    
    # 각 조인트의 이동량을 제한
    joint_movement = np.sign(joint_diff) * np.minimum(np.abs(joint_diff), max_movement)
    
    # 목표 조인트 계산
    next_joints = current_joints_rad_left + joint_movement
    
    # 목표에 도달했는지 확인 (임계값: 0.01 rad ≈ 0.57도)
    if np.all(np.abs(joint_diff) < 0.01):
        moving_to_preset = False
        print("Left Arm: Reached PRESET position")
        return preset_target_joints  # 정확한 목표 위치로 설정
    
    return next_joints

def publish_joint_command_left(joint_angles_rad):
    """Publish joint angles for LEFT arm to joint_state_command"""
    global node
    
    try:
        # Create JointState message
        joint_msg = JointState()
        joint_msg.header.stamp = node.get_clock().now().to_msg()
        
        # Set joint names (left arm joints in order)
        joint_msg.name = [
            'left_joint_1', 'left_joint_2', 'left_joint_3',
            'left_joint_4', 'left_joint_5', 'left_joint_6'
        ]
        
        # Set joint positions in radians
        joint_msg.position = joint_angles_rad.tolist()
        joint_msg.velocity = []
        joint_msg.effort = []
        
        # Publish command
        node.joint_command_pub.publish(joint_msg)
        return True
        
    except Exception as e:
        print(f"Left Arm: Failed to publish joint command: {e}")
        return False

class LeftArmVRServoNode(Node):
    def __init__(self):
        super().__init__('left_arm_vr_servo_node')
        
        # Publishers
        self.joint_command_pub = self.create_publisher(
            JointState,
            '/left_dsr_joint_controller/joint_state_command',
            10
        )
        
        # Subscribers
        self.tf_subscriber = self.create_subscription(
            TFMessage,
            '/tf',
            vr_pose_callback,
            10
        )
        
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            joint_state_callback,
            10
        )
        
        # 외부 제어 신호 구독자 추가
        self.teleop_control_sub = self.create_subscription(
            UInt8,
            '/teleop_control',  # 토픽 이름 (필요에 따라 변경 가능)
            teleop_control_callback,
            10
        )
        
        # 랜덤 제어 신호 구독자 추가
        self.random_control_sub = self.create_subscription(
            UInt8,
            '/random_control',  # 새로운 랜덤 제어 토픽
            random_control_callback,
            10
        )
        
        print("Left Arm VR Servo Node initialized")
        print("Left Arm: Listening for teleop control signals on /teleop_control topic")
        print("Left Arm: Listening for random control signals on /random_control topic")

def main(args=None):
    global running, robot, node
    global current_joints_rad_left
    global T_tracker_init_left, T_station2track_left, robot_pose_init_left
    
    # Load robot model
    try:
        robot = rtb.ERobot.URDF("/home/vision/dualarm_ws/src/doosan-robot2/dsr_description2/urdf/m0609.white.urdf")
        print("Left Arm: Robot model loaded successfully")
        print(f"Left Arm: Robot DOF: {robot.n}")
    except Exception as e:
        print(f"Left Arm: Error: Could not load robot model: {e}")
        return
    
    # Initialize ROS2
    rclpy.init(args=args)
    node = LeftArmVRServoNode()
    
    # Wait for joint states to be available for left arm
    print("Left Arm: Waiting for joint states...")
    while current_joints_rad_left is None and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        
    if current_joints_rad_left is None:
        print("Left Arm: Error: No joint states received")
        return
        
    print("Left Arm: Joint states received, starting VR servo control...")

    print("Left Arm: VR Servo Control Started!")
    print("Left Arm: Calibration matrices loaded from previous calibration results")
    print("Left Arm: Control will start automatically when VR tracker is detected")
    print("Left Arm: Waiting for control signals:")
    print("Left Arm:   - Publish 1 to /teleop_control to enable VR control")
    print("Left Arm:   - Publish 2 to /teleop_control to move to preset position")
    print("Left Arm:   - Publish 0 to /teleop_control to stop/hold position")
    print("Left Arm:   - Publish 1 to /random_control to enable random preset offsets")
    print("Left Arm:   - Publish 0 to /random_control to disable random preset offsets")

    try:
        while running and rclpy.ok():
            # Process ROS messages
            rclpy.spin_once(node, timeout_sec=0.005)
            
            # Initialize joint command
            next_joints_rad_left = None
            
            # Control LEFT arm (teleoperation이 활성화된 경우에만)
            if (teleop_enabled and 
                T_station2track_left is not None and 
                T_tracker_init_left is not None and 
                current_joints_rad_left is not None):
                
                # Calculate target pose from LEFT VR tracker movement
                target_pose_left = calculate_target_pose_left()
                if target_pose_left is not None:
                    
                    # Calculate joint delta using Jacobian for LEFT arm
                    dq_left = calculate_joint_delta_left(target_pose_left)
                    if dq_left is not None:
                        
                        # Update LEFT arm joint positions
                        next_joints_rad_left = current_joints_rad_left + dq_left * step_gain
                        
                        # print(f"Left Arm: Current joints: {np.rad2deg(current_joints_rad_left)[:]}")
                        # print(f"Left Arm: Joint delta: {np.rad2deg(dq_left)[:]}")
                        # print(f"Left Arm: Next joints: {np.rad2deg(next_joints_rad_left)[:]}")
            
            # Preset 위치로 이동 중인 경우
            elif moving_to_preset and current_joints_rad_left is not None:
                next_joints_rad_left = calculate_preset_movement_left()
            
            # Send joint command for left arm
            if next_joints_rad_left is not None:
                publish_joint_command_left(next_joints_rad_left)
            elif current_joints_rad_left is not None:
                # Hold current position for left arm (teleoperation 상태와 관계없이 항상 현재 위치 유지)
                publish_joint_command_left(current_joints_rad_left)
                
            # Small delay to prevent excessive CPU usage
            # time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nLeft Arm: Keyboard interrupt received. Stopping...")
    
    finally:
        running = False
        print("Left Arm: VR Servo Control terminated")
        
        # Shutdown ROS2
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
