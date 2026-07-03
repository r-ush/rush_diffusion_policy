#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                              QLabel, QSlider, QPushButton, QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt, QTimer

class ForceControlGUI(QWidget):
    """
    손가락 끝의 힘을 슬라이더로 제어하는 GUI
    각 손가락의 X, Y, Z축 힘을 개별적으로 조절 가능
    """
    
    def __init__(self, node):
        super().__init__()
        self.node = node
        
        # Force 값 저장 (-10N ~ 10N)
        self.forces = {
            'thumb': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'index': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'middle': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'ring': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'baby': {'x': 0.0, 'y': 0.0, 'z': 0.0}
        }
        
        # 실제 URDF의 프레임 이름으로 매핑
        self.frame_mapping = {
            'thumb': 'left_thumb_frame',
            'index': 'left_index_frame',
            'middle': 'left_middle_frame',
            'ring': 'left_ring_frame',
            'baby': 'left_baby_frame'
        }
        
        # Publishers
        self.publishers = {}
        for finger in self.forces.keys():
            topic_name = f'/left_hand/force/{finger}'
            self.publishers[finger] = self.node.create_publisher(
                WrenchStamped,
                topic_name,
                10
            )
        
        # GUI 초기화
        self.init_ui()
        
        # Timer로 주기적으로 힘 발행 (20Hz)
        self.timer = QTimer()
        self.timer.timeout.connect(self.publish_forces)
        self.timer.start(50)  # 50ms = 20Hz
        
    def init_ui(self):
        """GUI 구성"""
        self.setWindowTitle('Left Hand Force Control')
        self.setGeometry(100, 100, 600, 700)
        
        main_layout = QVBoxLayout()
        
        # 각 손가락별 컨트롤
        fingers = ['thumb', 'index', 'middle', 'ring', 'baby']
        finger_names = ['Thumb (엄지)', 'Index (검지)', 'Middle (중지)', 
                       'Ring (약지)', 'Baby (소지)']
        
        for finger, display_name in zip(fingers, finger_names):
            group_box = self.create_finger_control(finger, display_name)
            main_layout.addWidget(group_box)
        
        # Reset 버튼
        reset_layout = QHBoxLayout()
        reset_button = QPushButton('Reset All Forces')
        reset_button.clicked.connect(self.reset_all_forces)
        reset_layout.addWidget(reset_button)
        
        main_layout.addLayout(reset_layout)
        
        self.setLayout(main_layout)
        
    def create_finger_control(self, finger, display_name):
        """각 손가락의 XYZ 힘 제어 슬라이더 생성"""
        group_box = QGroupBox(display_name)
        layout = QGridLayout()
        
        axes = ['x', 'y', 'z']
        axis_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Cyan, Blue
        
        self.sliders = getattr(self, 'sliders', {})
        self.labels = getattr(self, 'labels', {})
        
        if finger not in self.sliders:
            self.sliders[finger] = {}
            self.labels[finger] = {}
        
        for i, axis in enumerate(axes):
            # Axis 레이블
            axis_label = QLabel(f'{axis.upper()} Force:')
            axis_label.setStyleSheet(f'color: {axis_colors[i]}; font-weight: bold;')
            layout.addWidget(axis_label, i, 0)
            
            # 슬라이더 (-10N ~ 10N, 0.1N 단위)
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-100)
            slider.setMaximum(100)
            slider.setValue(0)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(10)
            slider.valueChanged.connect(
                lambda value, f=finger, a=axis: self.slider_changed(f, a, value)
            )
            layout.addWidget(slider, i, 1)
            
            # 값 표시 레이블
            value_label = QLabel('0.0 N')
            value_label.setMinimumWidth(60)
            value_label.setStyleSheet('font-weight: bold;')
            layout.addWidget(value_label, i, 2)
            
            # Reset 버튼
            reset_btn = QPushButton('0')
            reset_btn.setMaximumWidth(40)
            reset_btn.clicked.connect(
                lambda checked, s=slider: s.setValue(0)
            )
            layout.addWidget(reset_btn, i, 3)
            
            self.sliders[finger][axis] = slider
            self.labels[finger][axis] = value_label
        
        group_box.setLayout(layout)
        return group_box
    
    def slider_changed(self, finger, axis, value):
        """슬라이더 값 변경 시"""
        force_value = value / 10.0  # -10.0 ~ 10.0 N
        self.forces[finger][axis] = force_value
        self.labels[finger][axis].setText(f'{force_value:.1f} N')
    
    def reset_all_forces(self):
        """모든 힘을 0으로 리셋"""
        for finger in self.sliders:
            for axis in self.sliders[finger]:
                self.sliders[finger][axis].setValue(0)
    
    def publish_forces(self):
        """현재 힘 값을 ROS2 토픽으로 발행"""
        for finger, force_dict in self.forces.items():
            msg = WrenchStamped()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_mapping[finger]
            
            msg.wrench.force.x = force_dict['x']
            msg.wrench.force.y = force_dict['y']
            msg.wrench.force.z = force_dict['z']
            
            # Torque는 0으로 설정
            msg.wrench.torque.x = 0.0
            msg.wrench.torque.y = 0.0
            msg.wrench.torque.z = 0.0
            
            self.publishers[finger].publish(msg)


class ForceControlNode(Node):
    """ROS2 노드 래퍼"""
    def __init__(self):
        super().__init__('force_control_gui')
        self.get_logger().info('Force Control GUI Node initialized')


def main(args=None):
    rclpy.init(args=args)
    
    # ROS2 노드 생성
    node = ForceControlNode()
    
    # Qt Application
    app = QApplication(sys.argv)
    
    # GUI 생성
    gui = ForceControlGUI(node)
    gui.show()
    
    # ROS2 스핀을 Qt 이벤트 루프와 통합
    timer = QTimer()
    timer.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0))
    timer.start(10)  # 100Hz
    
    # Qt 이벤트 루프 실행
    exit_code = app.exec_()
    
    # Cleanup
    node.destroy_node()
    rclpy.shutdown()
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
