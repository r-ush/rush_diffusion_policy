import os
import sys
import h5py
import numpy as np
from pynput import keyboard
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
import cv2
import time
from datetime import datetime
import argparse
import multiprocessing
from multiprocessing import Process, Queue, Event
import queue

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray, UInt8
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import JointState, MultiDOFJointState
from geometry_msgs.msg import WrenchStamped, TransformStamped
from rclpy.logging import get_logger
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import threading

# Import ForceVisualizer
try:
    from force_visualizer import ForceVisualizer
except ImportError:
    ForceVisualizer = None

_logger = get_logger('Logger')

# 전역 공유 상태 객체
G = {
    'terminal': False,
    'recording': False,
    'homming': False,
    
    'latest_joint_L': None,
    'latest_joint_R': None,
    'latest_hand_L': None,
    'latest_hand_R': None,
    'latest_joint_torque_L': None,
    'latest_joint_torque_R': None,
    
    'latest_wrench_aft_L': None,
    'latest_wrench_aft_R': None,
    'latest_wrench_thumb_L': None,
    'latest_wrench_thumb_R': None,
    'latest_wrench_index_L': None,
    'latest_wrench_index_R': None,
    'latest_wrench_middle_L': None,
    'latest_wrench_middle_R': None,
    'latest_wrench_ring_L': None,
    'latest_wrench_ring_R': None,
    'latest_wrench_baby_L': None,
    'latest_wrench_baby_R': None,
    
    'latest_F_e_raw': None,
    'latest_desired_pose': None,
    'latest_current_pose': None,
    
    'teleop_controller': None,
}

def init_buffer():   
    return {
        'observations': {
            ## Robot and Image data (20Hz)
            'joint_L': [], 
            'joint_R': [],
            'hand_L': [],
            'hand_R': [],
            'image_H': [], 
            'image_T': [], 
            'image_L': [], 
            'image_R': [], 
            'timestamp_robot': [],

            ## Wrench data (250Hz)
            'wrench_wrist_L': [], 
            'wrench_thumb_L': [],  
            'wrench_index_L': [],  
            'wrench_middle_L': [], 
            'wrench_ring_L': [],   
            'wrench_baby_L': [],   
            'wrench_wrist_R': [], 
            'wrench_thumb_R': [],  
            'wrench_index_R': [],  
            'wrench_middle_R': [], 
            'wrench_ring_R': [],   
            'wrench_baby_R': [],   

            'joint_torque_L': [], 
            'joint_torque_R': [],
            'timestamp_wrench': [],

            'F_e_raw': [],
            'desired_pose': [],
            'current_pose': []
        }
    }

def save_wrench_data(buffer, last_save_time):
    # Left
    buffer['observations']['wrench_wrist_L'].append(G['latest_wrench_aft_L'])
    buffer['observations']['wrench_thumb_L'].append(G['latest_wrench_thumb_L'])
    buffer['observations']['wrench_index_L'].append(G['latest_wrench_index_L'])
    buffer['observations']['wrench_middle_L'].append(G['latest_wrench_middle_L'])
    buffer['observations']['wrench_ring_L'].append(G['latest_wrench_ring_L'])
    buffer['observations']['wrench_baby_L'].append(G['latest_wrench_baby_L'])
    buffer['observations']['joint_torque_L'].append(G['latest_joint_torque_L'])
    # Right
    buffer['observations']['wrench_wrist_R'].append(G['latest_wrench_aft_R'])
    buffer['observations']['wrench_thumb_R'].append(G['latest_wrench_thumb_R'])
    buffer['observations']['wrench_index_R'].append(G['latest_wrench_index_R'])
    buffer['observations']['wrench_middle_R'].append(G['latest_wrench_middle_R'])
    buffer['observations']['wrench_ring_R'].append(G['latest_wrench_ring_R'])
    buffer['observations']['wrench_baby_R'].append(G['latest_wrench_baby_R'])
    buffer['observations']['joint_torque_R'].append(G['latest_joint_torque_R'])
    
    buffer['observations']['F_e_raw'].append(G['latest_F_e_raw'])
    buffer['observations']['desired_pose'].append(G['latest_desired_pose'])
    buffer['observations']['current_pose'].append(G['latest_current_pose'])
    buffer['observations']['timestamp_wrench'].append(last_save_time)

def save_robot_data(buffer, images, last_save_time):
    buffer['observations']['joint_L'].append(G['latest_joint_L'])
    buffer['observations']['joint_R'].append(G['latest_joint_R'])
    buffer['observations']['hand_L'].append(G['latest_hand_L'])
    buffer['observations']['hand_R'].append(G['latest_hand_R'])
    
    # Images (Head, Table, Left, Right)
    for i, key in enumerate(['image_H', 'image_T', 'image_L', 'image_R']):
        if images and i < len(images) and images[i] is not None:
            buffer['observations'][key].append(images[i].copy())
        else:
            buffer['observations'][key].append(None)
            
    buffer['observations']['timestamp_robot'].append(last_save_time)

def on_press(key):
    try:
        if key.char == 's':
            if not G['recording']:
                G['recording'] = True
                print("\n[EVENT] Start recording")
                if G['teleop_controller']:
                    G['teleop_controller'].send_teleop_command(1)  # data: 1
        elif key.char == 'q':
            if G['recording']:
                G['recording'] = False
                print("\n[EVENT] Stop recording")
                if G['teleop_controller']:
                    G['teleop_controller'].send_teleop_command(2)  # data: 0
        elif key.char == 't':
            G['terminal'] = True
            print("\n[EVENT] Terminal signal received")
    except AttributeError:
        pass

def get_device_serials():
    ctx = rs.context()
    serials = []
    for device in ctx.query_devices():
        serials.append(device.get_info(rs.camera_info.serial_number))
    return serials

def make_demo_n(buffer, data, f):
    n = len(data.keys())
    demo_n = data.create_group(f'demo_{n}')
    obs = demo_n.create_group('observations')
    for name, values in buffer['observations'].items():
        arr = np.array(values)
        if arr.dtype == np.object_:
            filtered_values = [v for v in values if v is not None]
            if len(filtered_values) > 0:
                # Try to handle inconsistent shapes if any
                try:
                    arr = np.array(filtered_values)
                except:
                    print(f"Warning: Inconsistent shapes in {name}, saving as object array.")
                    arr = np.array(filtered_values, dtype=object)
            else:
                continue
        obs.create_dataset(name, data=arr)
    f.flush()
    print(f"Stored demo_{n}, {len(buffer['observations']['joint_L'])} robot steps")

class Pipeline:
    def __init__(self, serial, exposure=None):
        self.serial = serial
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        try:
            profile = self.pipeline.start(self.config)
            for i in range(3):
                self.pipeline.wait_for_frames(timeout_ms=1000)
            self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            print(f"Camera {serial} initialized.")
        except Exception as e:
            print(f"Failed to initialize camera {serial}: {e}")
            self.pipeline = None

    def get_frame(self):
        if not self.pipeline: return None
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=50)
            color_frame = frames.get_color_frame()
            if not color_frame: return None
            return np.asanyarray(color_frame.get_data())
        except: return None

    def get_camera_matrix(self):
        if not self.pipeline: return None, None
        camera_matrix = np.array([[self.intrinsics.fx, 0, self.intrinsics.ppx],[0, self.intrinsics.fy, self.intrinsics.ppy],[0, 0, 1]])
        return camera_matrix, np.array(self.intrinsics.coeffs)

class JointSubscriber(Node):
    def __init__(self):
        super().__init__('joint_node')
        self.group = ReentrantCallbackGroup()
        self.joint_name = [f"left_joint_{i}" for i in range(1,7)] + [f"right_joint_{i}" for i in range(1,7)]
        self.hand_name = [f"left_{f}_joint{i}" for f in ['thumb','index','middle','ring','baby'] for i in range(1,4)] + \
                         [f"right_{f}_joint{i}" for f in ['thumb','index','middle','ring','baby'] for i in range(1,4)]
        
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10, callback_group=self.group)
        self.create_subscription(JointState, '/dsr01/joint_states', self.joint_cb, 10, callback_group=self.group)
        self.create_subscription(JointState, '/dsr02/joint_states', self.joint_cb, 10, callback_group=self.group)
        self.create_subscription(WrenchStamped, '/aft_sensor1/wrench', self.left_aft_cb, 10, callback_group=self.group)
        self.create_subscription(WrenchStamped, '/aft_sensor2/wrench', self.right_aft_cb, 10, callback_group=self.group)
        self.create_subscription(MultiDOFJointState, '/left_ft_sensor_broadcaster/wrench', self.left_ft_cb, 10, callback_group=self.group)
        self.create_subscription(MultiDOFJointState, '/right_ft_sensor_broadcaster/wrench', self.right_ft_cb, 10, callback_group=self.group)
        self.create_subscription(Float64MultiArray, '/F_e_raw', self.fe_cb, 10, callback_group=self.group)
        self.create_subscription(Float64MultiArray, '/desired_pose', self.des_cb, 10, callback_group=self.group)
        self.create_subscription(Float64MultiArray, '/current_pose', self.cur_cb, 10, callback_group=self.group)
        self.teleop_pub = self.create_publisher(UInt8, '/teleop_control', 10)
        self.names_printed = False

    def publish_teleop(self, value: int):
        msg = UInt8()
        msg.data = value
        self.teleop_pub.publish(msg)
        print(f"[TELEOP] Published /teleop_control: {value}")

    def joint_cb(self, msg):
        if not self.names_printed:
            print(f"\n[DEBUG] Received joint states from {msg.name}")
            self.names_printed = True

        m = {n: p for n, p in zip(msg.name, msg.position)}
        e = {n: t for n, t in zip(msg.name, msg.effort)} if msg.effort else {}

        # Fallback to generic "joint_X"
        generic_pos = [m.get(f"joint_{i}") for i in range(1,7)]
        generic_torq = [e.get(f"joint_{i}", 0.0) for i in range(1,7)]

        # Left Arm
        left_pos = [m.get(f"left_joint_{i}") for i in range(1,7)]
        if None in left_pos: left_pos = [m.get(f"dsr01_joint{i}") for i in range(1,7)]
        if None in left_pos: left_pos = generic_pos
        
        if None not in left_pos:
            G['latest_joint_L'] = left_pos
            left_torq = [e.get(f"left_joint_{i}", 0.0) for i in range(1,7)]
            if 0.0 in left_torq: left_torq = [e.get(f"dsr01_joint{i}", 0.0) for i in range(1,7)]
            if 0.0 in left_torq: left_torq = generic_torq
            G['latest_joint_torque_L'] = left_torq

        # Right Arm
        right_pos = [m.get(f"right_joint_{i}") for i in range(1,7)]
        if None in right_pos: right_pos = [m.get(f"dsr02_joint{i}") for i in range(1,7)]
        if None in right_pos: right_pos = generic_pos
        
        if None not in right_pos:
            G['latest_joint_R'] = right_pos
            right_torq = [e.get(f"right_joint_{i}", 0.0) for i in range(1,7)]
            if 0.0 in right_torq: right_torq = [e.get(f"dsr02_joint{i}", 0.0) for i in range(1,7)]
            if 0.0 in right_torq: right_torq = generic_torq
            G['latest_joint_torque_R'] = right_torq

        # Hands
        G['latest_hand_L'] = [m.get(j, 0.0) for j in self.hand_name[:15]]
        G['latest_hand_R'] = [m.get(j, 0.0) for j in self.hand_name[15:30]]
        
    def left_aft_cb(self, msg): G['latest_wrench_aft_L'] = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z, msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
    def right_aft_cb(self, msg): G['latest_wrench_aft_R'] = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z, msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
    def left_ft_cb(self, msg): 
        f = [np.array([w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z]) for w in msg.wrench]
        if len(f)>=5: G['latest_wrench_thumb_L'], G['latest_wrench_index_L'], G['latest_wrench_middle_L'], G['latest_wrench_ring_L'], G['latest_wrench_baby_L'] = f[:5]
    def right_ft_cb(self, msg): 
        f = [np.array([w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z]) for w in msg.wrench]
        if len(f)>=5: G['latest_wrench_thumb_R'], G['latest_wrench_index_R'], G['latest_wrench_middle_R'], G['latest_wrench_ring_R'], G['latest_wrench_baby_R'] = f[:5]
    def fe_cb(self, msg): G['latest_F_e_raw'] = np.array(msg.data)
    def des_cb(self, msg): G['latest_desired_pose'] = np.array(msg.data)
    def cur_cb(self, msg): G['latest_current_pose'] = np.array(msg.data)

class TeleopController(Node):
    def __init__(self):
        super().__init__('teleop_controller')
        self.teleop_control_pub = self.create_publisher(UInt8, '/teleop_control', 10)
    def send_teleop_command(self, command):
        msg = UInt8()
        msg.data = command
        self.teleop_control_pub.publish(msg)
        print(f"Teleop command sent: {command}")

def display_process_func(img_q, confirm_q, response_q, window_names, stop_event):
    confirming = False
    last_imgs = None
    last_recording = False

    while not stop_event.is_set():
        # 확인 요청 체크
        try:
            _ = confirm_q.get_nowait()
            confirming = True
        except: pass

        # 이미지 큐에서 최신 프레임 가져오기
        try:
            last_imgs, last_recording = img_q.get_nowait()
        except: pass

        # 화면 렌더링
        if last_imgs is not None:
            for i, (name, img) in enumerate(zip(window_names, last_imgs)):
                if img is None: continue
                disp = img.copy()
                if confirming:
                    # 반투명 어두운 오버레이
                    overlay = disp.copy()
                    cv2.rectangle(overlay, (0, 0), (disp.shape[1], disp.shape[0]), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.5, disp, 0.5, 0, disp)
                    cv2.putText(disp, 'Save this demo?', (30, disp.shape[0]//2 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                    cv2.putText(disp, '[Y] Save    [N] Discard', (30, disp.shape[0]//2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 150), 2)
                else:
                    label = 'REC' if last_recording else 'LIVE'
                    color = (0, 0, 255) if last_recording else (255, 255, 255)
                    cv2.putText(disp, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.imshow(name, disp)

        key = cv2.waitKey(30) & 0xFF
        if confirming:
            if key == ord('y'):
                response_q.put('y')
                confirming = False
            elif key == ord('n'):
                response_q.put('n')
                confirming = False

    cv2.destroyAllWindows()

def main():
    today = datetime.now().strftime('%y%m%d_%H%M')
    data_dir = f'/media/vision/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7/{today}'
    os.makedirs(data_dir, exist_ok=True)
    f = h5py.File(os.path.join(data_dir, 'common_data.hdf5'), 'a')
    data = f.require_group('data')

    connected_serials = get_device_serials()
    print(f"Connected serials: {connected_serials}")

    # [Head, Table, Left, Right] target serials
    target_serials = ['242422304502', '151222078010', '218622276386', '126122270712'] 
    
    # Actually connected cameras among targets
    serials = [s for s in target_serials if s in connected_serials]
    if not serials:
        # If none of the targets are found, use all currently connected cameras
        print("Warning: None of the target cameras found. Using all connected devices.")
        serials = connected_serials

    pipelines = []
    window_names = []
    for s in serials:
        try:
            p = Pipeline(s)
            pipelines.append(p)
            window_names.append(f'Cam_{s}')
        except Exception as e:
            print(f"Skipping camera {s} due to initialization error: {e}")

    if not pipelines:
        print("WARNING: No cameras initialized. Recording will proceed without images.")

    rclpy.init()
    node = JointSubscriber()
    G['teleop_controller'] = TeleopController()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.add_node(G['teleop_controller'])
    
    threading.Thread(target=executor.spin, daemon=True).start()

    img_q = Queue(maxsize=2)
    confirm_q = Queue()
    response_q = Queue()
    stop_event = Event()
    display_proc = Process(target=display_process_func, args=(img_q, confirm_q, response_q, window_names, stop_event))
    display_proc.daemon = True
    display_proc.start()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    listener_holder = [listener]  # mutable holder so we can replace it

    robot_rate = 1.0 / 20
    wrench_rate = 1.0 / 250
    buffer = init_buffer()
    episode_start_time = None
    last_display_time = 0
    last_debug_time = 0

    print("\ns: Start, q: Stop, t: Terminate")
    print("Ready for demonstration!")

    try:
        while rclpy.ok():
            now = time.monotonic()
            
            if now - last_debug_time >= 0.5:
                last_debug_time = now
                joint_status = "OK" if G['latest_joint_L'] is not None else "None"
                print(f"Loop | Rec: {G['recording']} | Term: {G['terminal']} | Joint: {joint_status}", end='\r')

            if G['terminal']: break

            if now - last_display_time >= 0.033:
                last_display_time = now
                imgs = [p.get_frame() for p in pipelines]
                try:
                    img_q.put_nowait((imgs, G['recording']))
                except: pass

            if G['recording']:
                if episode_start_time is None:
                    episode_start_time = now
                    last_robot_time = now
                    last_wrench_time = now
                    # Initial save
                    save_wrench_data(buffer, now)
                    save_robot_data(buffer, imgs, now)
                
                # Wrench recording (250Hz)
                if now >= last_wrench_time + wrench_rate:
                    last_wrench_time = now
                    save_wrench_data(buffer, now)
                
                # Robot/Image recording (20Hz)
                if now >= last_robot_time + robot_rate:
                    last_robot_time = now
                    # Refresh frames for high-rate saving
                    current_imgs = [p.get_frame() for p in pipelines]
                    save_robot_data(buffer, current_imgs, now)
                    if len(buffer['observations']['joint_L']) % 20 == 0:
                        print(f"Recording... {len(buffer['observations']['joint_L'])} steps", end='\r')
            
            elif episode_start_time is not None:
                steps = len(buffer['observations']['joint_L'])
                print(f"\nStopped. Collected {steps} steps. [Press Y/N in camera window, or T to exit]")
                if steps > 0:
                    confirm_q.put(True)
                    # 0.1초마다 폴링 - t를 누르면 즉시 빠져나옴
                    data_store = 'n'
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        if G['terminal']:
                            break
                        try:
                            data_store = response_q.get_nowait()
                            break
                        except:
                            time.sleep(0.1)
                    else:
                        print("Timeout: demo discarded.")
                    
                    if not G['terminal']:
                        if data_store == 'y':
                            make_demo_n(buffer, data, f)
                            print("Demo saved!")
                        else:
                            print("Demo discarded.")
                buffer = init_buffer()
                episode_start_time = None
            
            time.sleep(0.001)

    finally:
        stop_event.set()
        display_proc.terminate()
        f.close()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        print("\nCleaned up gracefully.")

if __name__ == '__main__':
    main()
