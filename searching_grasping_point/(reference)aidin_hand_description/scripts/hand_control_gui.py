#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                              QLabel, QSlider, QPushButton, QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt, QTimer

class HandControlGUI(QWidget):
    """
    손가락 관절을 슬라이더로 제어하는 GUI
    Joint State Publisher GUI의 centering 진동 문제를 해결
    """
    
    def __init__(self, node):
        super().__init__()
        self.node = node
        
        # 손가락 관절 정의 (라디안 단위)
        self.joints = {}
        fingers = ['thumb', 'index', 'middle', 'ring', 'baby']
        for finger in fingers:
            for i in range(1, 5):  # joint1~4
                joint_name = f'left_{finger}_joint{i}'
                self.joints[joint_name] = 0.0
        
        # Publisher
        self.joint_pub = self.node.create_publisher(
            JointState,
            '/joint_states',
            10
        )
        
        # GUI 초기화
        self.init_ui()
        
        # Timer로 주기적으로 joint state 발행 (50Hz)
        self.timer = QTimer()
        self.timer.timeout.connect(self.publish_joint_states)
        self.timer.start(20)  # 20ms = 50Hz
        
    def init_ui(self):
        """GUI 구성"""
        self.setWindowTitle('Hand Joint Control')
        self.setGeometry(100, 100, 700, 600)
        
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
        reset_button = QPushButton('Reset All Joints to 0')
        reset_button.clicked.connect(self.reset_all_joints)
        reset_layout.addWidget(reset_button)
        
        main_layout.addLayout(reset_layout)
        
        self.setLayout(main_layout)
        
    def create_finger_control(self, finger, display_name):
        """각 손가락의 4개 관절 제어 슬라이더 생성"""
        group_box = QGroupBox(display_name)
        layout = QGridLayout()
        
        self.sliders = getattr(self, 'sliders', {})
        self.labels = getattr(self, 'labels', {})
        
        if finger not in self.sliders:
            self.sliders[finger] = {}
            self.labels[finger] = {}
        
        # 각 손가락의 4개 관절
        for i in range(1, 5):
            joint_name = f'left_{finger}_joint{i}'
            
            # Joint 레이블
            joint_label = QLabel(f'Joint {i}:')
            joint_label.setMinimumWidth(60)
            layout.addWidget(joint_label, i-1, 0)
            
            # 슬라이더 (0 ~ 1.57 rad = 0 ~ 90도)
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(157)  # 1.57 rad * 100
            slider.setValue(0)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(20)
            slider.valueChanged.connect(
                lambda value, jn=joint_name: self.slider_changed(jn, value)
            )
            layout.addWidget(slider, i-1, 1)
            
            # 값 표시 레이블 (도 단위)
            value_label = QLabel('0.00 rad (0°)')
            value_label.setMinimumWidth(100)
            layout.addWidget(value_label, i-1, 2)
            
            # Reset 버튼
            reset_btn = QPushButton('0')
            reset_btn.setMaximumWidth(40)
            reset_btn.clicked.connect(
                lambda checked, s=slider: s.setValue(0)
            )
            layout.addWidget(reset_btn, i-1, 3)
            
            self.sliders[finger][i] = slider
            self.labels[finger][i] = value_label
        
        group_box.setLayout(layout)
        return group_box
    
    def slider_changed(self, joint_name, value):
        """슬라이더 값 변경 시"""
        rad_value = value / 100.0  # 0 ~ 1.57 rad
        deg_value = rad_value * 57.2958  # 라디안 → 도
        
        self.joints[joint_name] = rad_value
        
        # 레이블 업데이트
        finger = joint_name.split('_')[1]
        joint_num = int(joint_name.split('joint')[1])
        self.labels[finger][joint_num].setText(f'{rad_value:.2f} rad ({deg_value:.0f}°)')
    
    def reset_all_joints(self):
        """모든 관절을 0으로 리셋"""
        for finger in self.sliders:
            for joint_num in self.sliders[finger]:
                self.sliders[finger][joint_num].setValue(0)
    
    def publish_joint_states(self):
        """현재 관절 값을 JointState 토픽으로 발행"""
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        
        # 관절 이름과 위치 설정
        msg.name = list(self.joints.keys())
        msg.position = list(self.joints.values())
        
        self.joint_pub.publish(msg)


class HandControlNode(Node):
    """ROS2 노드 래퍼"""
    def __init__(self):
        super().__init__('hand_control_gui')
        self.get_logger().info('Hand Control GUI Node initialized')


def main(args=None):
    rclpy.init(args=args)
    
    # ROS2 노드 생성
    node = HandControlNode()
    
    # Qt Application
    app = QApplication(sys.argv)
    
    # GUI 생성
    gui = HandControlGUI(node)
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
