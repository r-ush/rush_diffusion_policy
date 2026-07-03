#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import math

class ForceVisualizer(Node):
    """
    손가락 끝의 힘을 RViz에 화살표로 시각화하는 노드
    
    Subscribe:
        /left_hand/force/{finger_name} (geometry_msgs/WrenchStamped)
        
    Publish:
        /left_hand/force_markers (visualization_msgs/MarkerArray)
    """
    
    def __init__(self):
        super().__init__('left_hand_force_visualizer')
        
        # Parameters
        self.declare_parameter('hand_prefix', 'left_')
        self.declare_parameter('force_scale', 0.1)  # 1N = 0.1m
        self.declare_parameter('arrow_diameter', 0.005)  # 5mm
        self.declare_parameter('finger_tips', ['left_thumb_tip', 'left_index_tip', 
                                               'left_middle_tip', 'left_ring_tip', 
                                               'left_baby_tip'])
        
        self.hand_prefix = self.get_parameter('hand_prefix').value
        self.force_scale = self.get_parameter('force_scale').value
        self.arrow_diameter = self.get_parameter('arrow_diameter').value
        self.finger_tips = self.get_parameter('finger_tips').value
        
        # 실제 URDF의 프레임 이름으로 매핑
        # URDF: left_*_frame, 토픽: /left_hand/force/*
        self.frame_mapping = {
            'thumb': 'left_thumb_frame',
            'index': 'left_index_frame',
            'middle': 'left_middle_frame',
            'ring': 'left_ring_frame',
            'baby': 'left_baby_frame'
        }
        
        # Publishers
        self.marker_pub = self.create_publisher(
            MarkerArray, 
            '/left_hand/force_markers', 
            10
        )
        
        # Subscribers - 각 손가락 끝의 힘 토픽 구독
        self.force_subscribers = {}
        self.current_forces = {}
        
        fingers = ['thumb', 'index', 'middle', 'ring', 'baby']
        for finger in fingers:
            topic_name = f'/left_hand/force/{finger}'
            self.current_forces[finger] = None
            
            self.force_subscribers[finger] = self.create_subscription(
                WrenchStamped,
                topic_name,
                lambda msg, f=finger: self.force_callback(msg, f),
                10
            )
            
            self.get_logger().info(f'Subscribed to {topic_name}')
        
        # Timer for publishing markers
        self.timer = self.create_timer(0.05, self.publish_markers)  # 20Hz
        
        self.get_logger().info('Force Visualizer initialized')
        self.get_logger().info(f'Force scale: {self.force_scale} m/N')
        self.get_logger().info(f'Arrow diameter: {self.arrow_diameter} m')
    
    def force_callback(self, msg: WrenchStamped, finger: str):
        """힘 데이터 수신 콜백"""
        self.current_forces[finger] = msg
    
    def publish_markers(self):
        """힘을 화살표로 시각화"""
        marker_array = MarkerArray()
        marker_id = 0
        
        fingers = ['thumb', 'index', 'middle', 'ring', 'baby']
        colors = {
            'thumb': ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8),   # 빨강
            'index': ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8),   # 초록
            'middle': ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8),  # 파랑
            'ring': ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8),    # 노랑
            'baby': ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.8)     # 자홍
        }
        
        for finger in fingers:
            if self.current_forces[finger] is None:
                continue
            
            wrench_msg = self.current_forces[finger]
            force = wrench_msg.wrench.force
            
            # 힘의 크기 계산
            force_magnitude = math.sqrt(
                force.x**2 + force.y**2 + force.z**2
            )
            
            # 힘이 너무 작으면 스킵
            if force_magnitude < 0.01:  # 0.01N 이하
                continue
            
            # Arrow Marker 생성
            marker = Marker()
            marker.header.frame_id = self.frame_mapping[finger]
            # 타임스탬프를 0으로 설정하여 최신 TF 사용
            marker.header.stamp.sec = 0
            marker.header.stamp.nanosec = 0
            marker.ns = f'force_{finger}'
            marker.id = marker_id
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            # 시작점 (손가락 끝)
            start_point = Point()
            start_point.x = 0.0
            start_point.y = 0.0
            start_point.z = 0.0
            marker.points.append(start_point)
            
            # 끝점 (힘 방향과 크기)
            scaled_force_x = force.x * self.force_scale
            scaled_force_y = force.y * self.force_scale
            scaled_force_z = force.z * self.force_scale
            
            end_point = Point()
            end_point.x = scaled_force_x
            end_point.y = scaled_force_y
            end_point.z = scaled_force_z
            marker.points.append(end_point)
            
            # 화살표 스타일
            marker.scale.x = self.arrow_diameter  # 축 지름
            marker.scale.y = self.arrow_diameter * 2  # 머리 지름
            marker.scale.z = 0.0  # 미사용
            
            # 색상 (힘의 크기에 따라 투명도 조절)
            marker.color = colors[finger]
            marker.color.a = min(1.0, force_magnitude / 10.0)  # 10N에서 완전 불투명
            
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 100000000  # 0.1초
            
            marker_array.markers.append(marker)
            marker_id += 1
            
            # Text Marker (힘의 크기 표시)
            text_marker = Marker()
            text_marker.header.frame_id = self.frame_mapping[finger]
            # 타임스탬프를 0으로 설정하여 최신 TF 사용
            text_marker.header.stamp.sec = 0
            text_marker.header.stamp.nanosec = 0
            text_marker.ns = f'force_text_{finger}'
            text_marker.id = marker_id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = scaled_force_x
            text_marker.pose.position.y = scaled_force_y
            text_marker.pose.position.z = scaled_force_z + 0.02  # 화살표 위에
            
            text_marker.text = f'{force_magnitude:.2f}N'
            text_marker.scale.z = 0.01  # 텍스트 크기
            text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)  # 흰색
            
            text_marker.lifetime.sec = 0
            text_marker.lifetime.nanosec = 100000000
            
            marker_array.markers.append(text_marker)
            marker_id += 1
        
        # Publish markers
        if len(marker_array.markers) > 0:
            self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    
    force_visualizer = ForceVisualizer()
    
    try:
        rclpy.spin(force_visualizer)
    except KeyboardInterrupt:
        pass
    finally:
        force_visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
