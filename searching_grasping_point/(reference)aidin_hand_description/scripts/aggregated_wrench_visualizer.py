#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
import math

class AggregatedWrenchVisualizer(Node):
    """
    Aggregated wrench를 RViz에 화살표로 시각화하는 노드
    
    Subscribe:
        /{hand}_wrist_wrench (geometry_msgs/WrenchStamped)
        
    Publish:
        /{hand}_aggregated_wrench_markers (visualization_msgs/MarkerArray)
    """
    
    def __init__(self):
        super().__init__('aggregated_wrench_visualizer')
        
        # Parameters
        self.declare_parameter('hand_prefix', 'right_')
        self.declare_parameter('wrist_frame', 'right_hand_base_link')
        self.declare_parameter('force_scale', 0.01)  # 1N = 0.01m (10mm)
        self.declare_parameter('torque_scale', 0.1)  # 1Nm = 0.1m (100mm)
        self.declare_parameter('force_arrow_diameter', 0.008)  # 8mm
        self.declare_parameter('torque_arrow_diameter', 0.006)  # 6mm
        
        self.hand_prefix = self.get_parameter('hand_prefix').value
        self.wrist_frame = self.get_parameter('wrist_frame').value
        self.force_scale = self.get_parameter('force_scale').value
        self.torque_scale = self.get_parameter('torque_scale').value
        self.force_arrow_diameter = self.get_parameter('force_arrow_diameter').value
        self.torque_arrow_diameter = self.get_parameter('torque_arrow_diameter').value
        
        hand_side = self.hand_prefix.rstrip('_')
        
        # Publisher
        self.marker_pub = self.create_publisher(
            MarkerArray, 
            f'/{hand_side}_aggregated_wrench_markers', 
            10
        )
        
        # Subscriber
        aggregated_topic = f'/{hand_side}_wrist_wrench'
        self.wrench_sub = self.create_subscription(
            WrenchStamped,
            aggregated_topic,
            self.wrench_callback,
            10
        )
        
        self.current_wrench = None
        
        # Timer for publishing markers
        self.timer = self.create_timer(0.05, self.publish_markers)  # 20Hz
        
        self.get_logger().info(f'Aggregated Wrench Visualizer initialized')
        self.get_logger().info(f'Subscribed to: {aggregated_topic}')
        self.get_logger().info(f'Wrist frame: {self.wrist_frame}')
        self.get_logger().info(f'Force scale: {self.force_scale} m/N')
        self.get_logger().info(f'Torque scale: {self.torque_scale} m/Nm')
    
    def wrench_callback(self, msg: WrenchStamped):
        """Aggregated wrench 수신 콜백"""
        self.current_wrench = msg.wrench
    
    def publish_markers(self):
        """Wrench를 화살표로 시각화"""
        if self.current_wrench is None:
            return
        
        marker_array = MarkerArray()
        
        force = self.current_wrench.force
        torque = self.current_wrench.torque
        
        # 힘의 크기 계산
        force_magnitude = math.sqrt(
            force.x**2 + force.y**2 + force.z**2
        )
        
        # 토크의 크기 계산
        torque_magnitude = math.sqrt(
            torque.x**2 + torque.y**2 + torque.z**2
        )
        
        # Force Arrow Marker
        if force_magnitude > 0.01:  # 0.01N 이상
            force_marker = Marker()
            force_marker.header.frame_id = self.wrist_frame
            force_marker.header.stamp = self.get_clock().now().to_msg()
            force_marker.ns = 'aggregated_force'
            force_marker.id = 0
            force_marker.type = Marker.ARROW
            force_marker.action = Marker.ADD
            
            # 시작점 (손목 중심)
            start_point = Point()
            start_point.x = 0.0
            start_point.y = 0.0
            start_point.z = 0.0
            force_marker.points.append(start_point)
            
            # 끝점 (힘 방향과 크기)
            end_point = Point()
            end_point.x = force.x * self.force_scale
            end_point.y = force.y * self.force_scale
            end_point.z = force.z * self.force_scale
            force_marker.points.append(end_point)
            
            # 화살표 스타일
            force_marker.scale.x = self.force_arrow_diameter  # 축 지름
            force_marker.scale.y = self.force_arrow_diameter * 2  # 머리 지름
            force_marker.scale.z = 0.0
            
            # 색상 - 빨강 (힘의 크기에 따라 밝기 조절)
            force_marker.color = ColorRGBA(
                r=1.0, 
                g=0.0, 
                b=0.0, 
                a=min(1.0, force_magnitude / 50.0 + 0.3)  # 최소 0.3, 50N에서 완전 불투명
            )
            
            force_marker.lifetime.sec = 0
            force_marker.lifetime.nanosec = 100000000  # 0.1초
            
            marker_array.markers.append(force_marker)
            
            # Force Text Marker
            force_text = Marker()
            force_text.header.frame_id = self.wrist_frame
            force_text.header.stamp = self.get_clock().now().to_msg()
            force_text.ns = 'aggregated_force_text'
            force_text.id = 1
            force_text.type = Marker.TEXT_VIEW_FACING
            force_text.action = Marker.ADD
            
            force_text.pose.position.x = end_point.x
            force_text.pose.position.y = end_point.y
            force_text.pose.position.z = end_point.z + 0.02
            
            force_text.text = f'F: {force_magnitude:.2f}N'
            force_text.scale.z = 0.015  # 텍스트 크기
            force_text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)  # 흰색
            
            force_text.lifetime.sec = 0
            force_text.lifetime.nanosec = 100000000
            
            marker_array.markers.append(force_text)
        
        # Torque Arrow Marker
        if torque_magnitude > 0.001:  # 0.001Nm 이상
            torque_marker = Marker()
            torque_marker.header.frame_id = self.wrist_frame
            torque_marker.header.stamp = self.get_clock().now().to_msg()
            torque_marker.ns = 'aggregated_torque'
            torque_marker.id = 2
            torque_marker.type = Marker.ARROW
            torque_marker.action = Marker.ADD
            
            # 시작점 (손목 중심에서 약간 옆으로)
            start_point = Point()
            start_point.x = 0.0
            start_point.y = 0.03  # 3cm 옆
            start_point.z = 0.0
            torque_marker.points.append(start_point)
            
            # 끝점 (토크 방향과 크기)
            end_point = Point()
            end_point.x = torque.x * self.torque_scale
            end_point.y = 0.03 + torque.y * self.torque_scale
            end_point.z = torque.z * self.torque_scale
            torque_marker.points.append(end_point)
            
            # 화살표 스타일
            torque_marker.scale.x = self.torque_arrow_diameter  # 축 지름
            torque_marker.scale.y = self.torque_arrow_diameter * 2  # 머리 지름
            torque_marker.scale.z = 0.0
            
            # 색상 - 파랑 (토크의 크기에 따라 밝기 조절)
            torque_marker.color = ColorRGBA(
                r=0.0, 
                g=0.5, 
                b=1.0, 
                a=min(1.0, torque_magnitude / 5.0 + 0.3)  # 최소 0.3, 5Nm에서 완전 불투명
            )
            
            torque_marker.lifetime.sec = 0
            torque_marker.lifetime.nanosec = 100000000
            
            marker_array.markers.append(torque_marker)
            
            # Torque Text Marker
            torque_text = Marker()
            torque_text.header.frame_id = self.wrist_frame
            torque_text.header.stamp = self.get_clock().now().to_msg()
            torque_text.ns = 'aggregated_torque_text'
            torque_text.id = 3
            torque_text.type = Marker.TEXT_VIEW_FACING
            torque_text.action = Marker.ADD
            
            torque_text.pose.position.x = end_point.x
            torque_text.pose.position.y = end_point.y
            torque_text.pose.position.z = end_point.z + 0.02
            
            torque_text.text = f'T: {torque_magnitude:.3f}Nm'
            torque_text.scale.z = 0.012  # 텍스트 크기
            torque_text.color = ColorRGBA(r=0.5, g=0.8, b=1.0, a=1.0)  # 하늘색
            
            torque_text.lifetime.sec = 0
            torque_text.lifetime.nanosec = 100000000
            
            marker_array.markers.append(torque_text)
        
        # Component Text Markers (상세 정보)
        detail_text = Marker()
        detail_text.header.frame_id = self.wrist_frame
        detail_text.header.stamp = self.get_clock().now().to_msg()
        detail_text.ns = 'wrench_details'
        detail_text.id = 4
        detail_text.type = Marker.TEXT_VIEW_FACING
        detail_text.action = Marker.ADD
        
        detail_text.pose.position.x = 0.0
        detail_text.pose.position.y = -0.05
        detail_text.pose.position.z = 0.05
        
        detail_text.text = (
            f'Force: [{force.x:.1f}, {force.y:.1f}, {force.z:.1f}] N\n'
            f'Torque: [{torque.x:.2f}, {torque.y:.2f}, {torque.z:.2f}] Nm'
        )
        detail_text.scale.z = 0.008  # 작은 텍스트
        detail_text.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8)  # 노란색
        
        detail_text.lifetime.sec = 0
        detail_text.lifetime.nanosec = 100000000
        
        marker_array.markers.append(detail_text)
        
        # Publish markers
        if len(marker_array.markers) > 0:
            self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    
    visualizer = AggregatedWrenchVisualizer()
    
    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
