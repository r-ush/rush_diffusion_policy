#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped, Vector3
from sensor_msgs.msg import JointState
import numpy as np
from tf2_ros import TransformListener, Buffer
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs

class WristWrenchAggregator(Node):
    """
    각 손가락 끝에서 받는 힘을 손목 프레임 기준으로 변환하여 총합을 계산
    
    Forward Kinematics를 고려하여:
    1. 각 손가락 끝 프레임의 힘을 TF를 통해 손목 프레임으로 변환
    2. 힘(force)과 토크(torque)를 모두 계산
    3. 모든 손가락의 힘/토크를 합산
    """
    
    def __init__(self):
        super().__init__('wrist_wrench_aggregator')
        
        # Parameters
        self.declare_parameter('hand_prefix', 'left_')
        self.declare_parameter('wrist_frame', 'left_hand_base_link')
        self.declare_parameter('finger_tips', [
            'left_thumb_tip',
            'left_index_tip',
            'left_middle_tip',
            'left_ring_tip',
            'left_baby_tip'
        ])
        
        self.hand_prefix = self.get_parameter('hand_prefix').value
        self.wrist_frame = self.get_parameter('wrist_frame').value
        self.finger_tips = self.get_parameter('finger_tips').value
        
        # TF Buffer and Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 각 손가락 끝에서 받는 힘 저장
        self.finger_wrenches = {}
        for tip in self.finger_tips:
            self.finger_wrenches[tip] = None
        
        # Subscribers for each finger tip force
        self.force_subscribers = {}
        for tip in self.finger_tips:
            finger_name = tip.replace(self.hand_prefix, '').replace('_tip', '')
            # force_control_gui.py publishes to /left_hand/force/{finger}
            topic = f'/{self.hand_prefix}hand/force/{finger_name}'
            self.force_subscribers[tip] = self.create_subscription(
                WrenchStamped,
                topic,
                lambda msg, t=tip: self.finger_force_callback(msg, t),
                10
            )
            self.get_logger().info(f'Subscribed to {topic} for {tip}')
        
        # Joint State Subscriber (for debugging/monitoring)
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Publisher for aggregated wrench at wrist
        self.wrist_wrench_pub = self.create_publisher(
            WrenchStamped,
            f'/{self.hand_prefix}wrist_wrench',
            10
        )
        
        # Timer to compute and publish aggregated wrench
        self.timer = self.create_timer(0.02, self.compute_wrist_wrench)  # 50Hz
        
        self.get_logger().info(f'Wrist Wrench Aggregator initialized')
        self.get_logger().info(f'Wrist frame: {self.wrist_frame}')
        self.get_logger().info(f'Monitoring finger tips: {self.finger_tips}')
    
    def finger_force_callback(self, msg: WrenchStamped, finger_tip: str):
        """각 손가락 끝에서 받는 힘 저장"""
        self.finger_wrenches[finger_tip] = msg
    
    def joint_state_callback(self, msg: JointState):
        """Joint state 모니터링 (디버깅용)"""
        pass
    
    def transform_wrench(self, wrench_stamped: WrenchStamped, target_frame: str):
        """
        WrenchStamped를 target_frame으로 변환
        
        힘(force)과 토크(torque) 모두 변환:
        - Force: 회전만 적용
        - Torque: 회전 + 위치에 따른 추가 토크 (r × F)
        """
        try:
            # Get transform from source frame to target frame
            source_frame = wrench_stamped.header.frame_id
            
            # Wait for transform (최대 0.1초)
            transform = self.tf_buffer.lookup_transform(
                target_frame,      # 손목 프레임 (left_hand_base_link)
                source_frame,      # 손가락 끝 프레임 (left_thumb_tip 등)
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            # Extract rotation and translation
            q = transform.transform.rotation
            t = transform.transform.translation
            
            # Convert quaternion to rotation matrix
            R = self.quaternion_to_rotation_matrix(q)
            
            # Original force and torque
            f_orig = np.array([
                wrench_stamped.wrench.force.x,
                wrench_stamped.wrench.force.y,
                wrench_stamped.wrench.force.z
            ])
            
            tau_orig = np.array([
                wrench_stamped.wrench.torque.x,
                wrench_stamped.wrench.torque.y,
                wrench_stamped.wrench.torque.z
            ])
            
            # Transform force: f_new = R * f_orig
            f_new = R @ f_orig
            
            # Transform torque: tau_new = R * tau_orig + r × (R * f_orig)
            # where r is the position vector from target frame to source frame
            r = np.array([t.x, t.y, t.z])
            tau_new = R @ tau_orig + np.cross(r, f_new)
            
            # Create transformed wrench
            transformed_wrench = WrenchStamped()
            transformed_wrench.header.stamp = self.get_clock().now().to_msg()
            transformed_wrench.header.frame_id = target_frame
            
            transformed_wrench.wrench.force.x = f_new[0]
            transformed_wrench.wrench.force.y = f_new[1]
            transformed_wrench.wrench.force.z = f_new[2]
            
            transformed_wrench.wrench.torque.x = tau_new[0]
            transformed_wrench.wrench.torque.y = tau_new[1]
            transformed_wrench.wrench.torque.z = tau_new[2]
            
            return transformed_wrench
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(
                f'Could not transform wrench from {source_frame} to {target_frame}: {e}',
                throttle_duration_sec=2.0
            )
            return None
    
    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to 3x3 rotation matrix"""
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])
        
        return R
    
    def compute_wrist_wrench(self):
        """
        각 손가락의 힘을 손목 프레임으로 변환하여 총합 계산
        """
        # 총 힘과 토크 초기화
        total_force = np.zeros(3)
        total_torque = np.zeros(3)
        
        valid_count = 0
        
        # 각 손가락의 힘을 변환하여 합산
        for finger_tip, wrench in self.finger_wrenches.items():
            if wrench is None:
                continue
            
            # Transform wrench to wrist frame
            transformed_wrench = self.transform_wrench(wrench, self.wrist_frame)
            
            if transformed_wrench is not None:
                # 힘 합산
                total_force[0] += transformed_wrench.wrench.force.x
                total_force[1] += transformed_wrench.wrench.force.y
                total_force[2] += transformed_wrench.wrench.force.z
                
                # 토크 합산
                total_torque[0] += transformed_wrench.wrench.torque.x
                total_torque[1] += transformed_wrench.wrench.torque.y
                total_torque[2] += transformed_wrench.wrench.torque.z
                
                valid_count += 1
        
        # Publish aggregated wrench
        if valid_count > 0:
            wrist_wrench = WrenchStamped()
            wrist_wrench.header.stamp = self.get_clock().now().to_msg()
            wrist_wrench.header.frame_id = self.wrist_frame
            
            wrist_wrench.wrench.force.x = total_force[0]
            wrist_wrench.wrench.force.y = total_force[1]
            wrist_wrench.wrench.force.z = total_force[2]
            
            wrist_wrench.wrench.torque.x = total_torque[0]
            wrist_wrench.wrench.torque.y = total_torque[1]
            wrist_wrench.wrench.torque.z = total_torque[2]
            
            self.wrist_wrench_pub.publish(wrist_wrench)
            
            # Log for debugging (throttled)
            if valid_count == len(self.finger_tips):
                force_mag = np.linalg.norm(total_force)
                torque_mag = np.linalg.norm(total_torque)
                self.get_logger().debug(
                    f'Wrist wrench - Force: {force_mag:.3f}N, Torque: {torque_mag:.3f}Nm',
                    throttle_duration_sec=1.0
                )


def main(args=None):
    rclpy.init(args=args)
    node = WristWrenchAggregator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
