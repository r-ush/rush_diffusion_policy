
# 오른팔 임피던스 컨트롤러 (leftarm_interpolation_controller_imp.py의 오른팔 버전)
# - 목표 pose 발행 토픽: /calculated_pose  (왼팔과 동일, 오른팔 임피던스 노드가 구독)
# - joint_states 구독 토픽: /dsr01/joint_states  (왼팔과 동일)
# - obs key 이름(robot_pose_L / robot_quat_L)은 체크포인트 shape_meta가 요구하는 라벨이라
#   오른팔이더라도 그대로 유지한다.

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from spatialmath import SE3
import spatialmath.base as smb
from std_msgs.msg import Int32, Float64, String, Float64MultiArray
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation as R

import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np

from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator


def rot6d_to_rotvec(rot6d: np.ndarray) -> np.ndarray:
    a1 = rot6d[:3]
    a2 = rot6d[3:]
    b1 = a1 / np.linalg.norm(a1)
    a2_proj = np.dot(b1, a2) * b1
    b2 = a2 - a2_proj
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    R_mat = np.stack((b1, b2, b3), axis=1)
    rot = R.from_matrix(R_mat)
    rot_vec = rot.as_rotvec()
    return rot_vec


class RightarmImp(Node):
    def __init__(self):
        super().__init__('rightarm_imp_node')

        self.joint_name = [f"joint_{i}" for i in range(1,7)]

        self.joint_subscriber = self.create_subscription(
            JointState,
            '/dsr01/joint_states',
            self.joint_callback,
            10
        )

        # impedance control target pose (Float64MultiArray)
        self.pose_command_publisher = self.create_publisher(
            Float64MultiArray,
            '/calculated_pose',
            10
        )

        # trajectory 확인용
        self.tcp_publisher_R = self.create_publisher(
            PoseStamped,
            '/TCP_target_pose_R',
            10
        )

    def joint_callback(self, msg):
        global latest_joint_R
        joint_mapping = {n: p for n, p in zip(msg.name, msg.position)}
        joint_position = [joint_mapping.get(j) for j in self.joint_name]
        latest_joint_R = joint_position[:6]

    def pose_command_publish(self, pose_array):
        msg = Float64MultiArray()
        # pose_array: [x(mm), y(mm), z(mm), z_rot(deg), y_rot(deg), x_rot(deg)]
        msg.data = [float(x) for x in pose_array]
        self.pose_command_publisher.publish(msg)

    # trajectory 확인용
    def tcp_pose_publish_R(self, tcp_pose):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(tcp_pose[0])
        msg.pose.position.y = float(tcp_pose[1])
        msg.pose.position.z = float(tcp_pose[2])
        self.tcp_publisher_R.publish(msg)

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2


class RightarmInterpolationControllerImp(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """

    def __init__(self,
            shm_manager: SharedMemoryManager,
            robot_ip,
            frequency=125,
            lookahead_time=0.1,
            gain=300,
            max_pos_speed=0.25, # 5% of max speed
            max_rot_speed=0.16, # 5% of max speed
            launch_timeout=3,
            tcp_offset_pose=None,
            payload_mass=None,
            payload_cog=None,
            joints_init=None,
            joints_init_speed=1.05,
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=128,   # 30
            ):
        """
        frequency: CB2=125, UR3e=500
        lookahead_time: [0.03, 0.2]s smoothens the trajectory with this lookahead time
        gain: [100, 2000] proportional gain for following target position
        max_pos_speed: m/s
        max_rot_speed: rad/s
        """
        # verify
        assert 0 < frequency <= 500
        assert 0.03 <= lookahead_time <= 0.2
        assert 100 <= gain <= 2000
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed

        super().__init__(name="RTDEPositionalController")
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose

        # build input queue; action 담아놓을 메모리
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((9,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer; state 담아놓을 메모리
        # NOTE: key 이름은 체크포인트 shape_meta가 요구하는 라벨이라 오른팔이어도 _L 유지
        if receive_keys is None:
            receive_keys = [
                'robot_pose_L',
                'robot_quat_L',
            ]

        example = dict()
        for key in receive_keys:
            if key == 'robot_pose_L':
                example[key] = np.zeros((3,), dtype=np.float64)
            elif key == 'robot_quat_L':
                example[key] = np.zeros((4,), dtype=np.float64)

        example['robot_receive_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )


        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
        print("[DEBUG] Robot Controller initialized (Impedance Mode, Rightarm)")

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[RTDEPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def schedule_waypoint(self, pose, target_time):   # 이거 사용
        assert target_time > time.time()
        pose = np.array(pose)
        assert pose.shape == (9,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))
        print("[DEBUG] Robot Running (Impedance Mode, Rightarm)")
        # start rtde
        robot_ip = self.robot_ip

        urdf_path = "/home/vision/dualarm_ws/src/doosan-robot2/dsr_description2/urdf/m0609.white.urdf"
        doosan_robot = rtb.ERobot.URDF(urdf_path)

        global latest_joint_R
        latest_joint_R = None
        rclpy.init(args=None)
        node = RightarmImp()

        try:
            # wait until first joint state is received
            while latest_joint_R is None or None in latest_joint_R:
                rclpy.spin_once(node, timeout_sec=0.1)

            # main loop
            dt = 1. / self.frequency

            # current state; m, rad
            curr_joint_R = latest_joint_R

            curr_tcp_R = doosan_robot.fkine(curr_joint_R)

            curr_tcp_pose_R = curr_tcp_R.t
            curr_tcp_rotmat_R = curr_tcp_R.R

            curr_tcp_quat_R = R.from_matrix(curr_tcp_rotmat_R).as_quat()

            if curr_tcp_quat_R[3] < 0:
                curr_tcp_quat_R = -curr_tcp_quat_R

            curr_tcp_rotvec_R = R.from_quat(curr_tcp_quat_R).as_rotvec()

            curr_pose = np.concatenate([curr_tcp_pose_R, curr_tcp_rotvec_R])

            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],     # [ time ]
                poses=[curr_pose],  # [ [x,y,z,rx,ry,rz] ]
                action_type='leftarm'   # 단일팔 6d(pos3+rotvec3) 보간기 (팔 무관, 재사용)
            )

            iter_idx = 0
            keep_running = True

            t_start = time.monotonic()   # 수동 제어 주기 맞추기

            # trajectory 확인용
            cmd_index = 0

            while keep_running:   # 루프 시작
                rclpy.spin_once(node)
                # start control iteration
                # send command to robot
                t_now = time.monotonic()
                pose_command = pose_interp(t_now)   # 보간 해놓고 현재 시간의 목표 pose 가져옴 (pose_R, rotvec_R)

                # 두산 로봇 제어
                # pose_command 해체
                target_pose_R = pose_command[:3] # meters
                target_rotvec_R = pose_command[3:6]

                # C++ Impedance Controller 요구 규격에 맞게 변환
                # 1. Position: meters -> millimeters
                target_pos_mm = target_pose_R * 1000.0

                # 2. Orientation: rotvec -> ZYX Euler angles in degrees
                target_rot = R.from_rotvec(target_rotvec_R)
                target_euler_deg = target_rot.as_euler('ZYX', degrees=True) # [Z_rot, Y_rot, X_rot]

                # Float64MultiArray (6,)
                pose_array = np.concatenate([target_pos_mm, target_euler_deg])

                # 토픽발사
                node.pose_command_publish(pose_array)

                # current state (Forward Kinematics로 현재 TCP 갱신)
                curr_joint_R = latest_joint_R
                curr_tcp_R = doosan_robot.fkine(curr_joint_R)

                curr_tcp_pose_R = curr_tcp_R.t
                curr_tcp_rotmat_R = curr_tcp_R.R

                curr_tcp_quat_R = R.from_matrix(curr_tcp_rotmat_R).as_quat()

                if curr_tcp_quat_R[3] < 0:
                    curr_tcp_quat_R = -curr_tcp_quat_R

                curr_tcp_rotvec_R = R.from_quat(curr_tcp_quat_R).as_rotvec()

                # 현재 State 저장
                # update robot state; ringbuffer에 state 저장
                # NOTE: key 이름은 체크포인트 shape_meta 라벨이라 오른팔이어도 _L 유지
                state = dict()

                for key in self.receive_keys:
                    if key == 'robot_pose_L':
                        state[key] = np.array(curr_tcp_pose_R)
                    elif key == 'robot_quat_L':
                        state[key] = np.array(curr_tcp_quat_R)

                state['robot_receive_timestamp'] = time.time()
                self.ring_buffer.put(state)


                # fetch command from queue

                try:
                    commands = self.input_queue.get_all()   # command 긁어옴
                    n_cmd = len(commands['cmd'])
                    print("[DEBUG] received n_cmd:", n_cmd)

                except Empty:
                    n_cmd = 0

                # execute commands
                # action 한번에 참

                for i in range(n_cmd):   # 가져온 cmd 수만큼 실행
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:   # STOP: 그만두기
                        keep_running = False
                        # stop immediately, ignore later commands
                        break

                    # 이걸로 제어 (n_cmd 1개씩 제어)
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']   # abs; (pose_R, rot6d_R)

                        target_position_R = target_pose[:3]   # 3d position, m
                        target_rotvec_R = rot6d_to_rotvec(target_pose[3:9])   # 6d rotation -> rot_vec
                        target_pose = np.concatenate([target_position_R, target_rotvec_R])

                        if cmd_index < 6:
                            node.tcp_pose_publish_R(target_position_R)
                        cmd_index += 1
                        if cmd_index == 14:
                            cmd_index = 0

                        target_time = float(command['target_time'])   # time.time 기준
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time   # time.monotonic 기준
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(   # 여기서 pose_interp 갱신
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time,
                            action_type='leftarm'   # 단일팔 6d 보간기 재사용
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                # regulate frequency
                t_elapsed = time.monotonic() - t_now
                sleep_time = dt - t_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)   # 수동 제어 주기

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()   # 준비완료; wait하던거 실행됨
                iter_idx += 1

                if self.verbose:
                    print(f"[RTDEPositionalController] Actual frequency {1/(time.perf_counter() - t_start)}")

        finally:
            # terminate
            node.destroy_node()
            rclpy.shutdown()

            self.ready_event.set()

            if self.verbose:
                print(f"[RTDEPositionalController] Disconnected from robot: {robot_ip}")
