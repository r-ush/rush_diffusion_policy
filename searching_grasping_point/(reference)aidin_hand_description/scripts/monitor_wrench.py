#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
import math

class WrenchMonitor(Node):
    """
    손가락 힘과 손목 집계 힘을 모니터링하는 노드
    
    Subscribe:
        /left_hand/force/{finger} - 각 손가락 힘
        /left_wrist_wrench - 손목 집계 힘
    """
    
    def __init__(self):
        super().__init__('wrench_monitor')
        
        # 손가락별 힘 저장
        self.finger_forces = {
            'thumb': None,
            'index': None,
            'middle': None,
            'ring': None,
            'baby': None
        }
        
        # Subscribers for each finger
        for finger in self.finger_forces.keys():
            topic = f'/left_hand/force/{finger}'
            self.create_subscription(
                WrenchStamped,
                topic,
                lambda msg, f=finger: self.finger_callback(msg, f),
                10
            )
        
        # Subscriber for wrist wrench
        self.create_subscription(
            WrenchStamped,
            '/left_wrist_wrench',
            self.wrist_callback,
            10
        )
        
        self.wrist_wrench = None
        
        # Timer to print summary
        self.timer = self.create_timer(1.0, self.print_summary)
        
        self.get_logger().info('Wrench Monitor started')
    
    def finger_callback(self, msg: WrenchStamped, finger: str):
        """개별 손가락 힘 저장"""
        self.finger_forces[finger] = msg.wrench
    
    def wrist_callback(self, msg: WrenchStamped):
        """손목 집계 힘 저장"""
        self.wrist_wrench = msg.wrench
    
    def print_summary(self):
        """현재 상태 출력"""
        print("\n" + "="*80)
        print("FINGER TIP FORCES (Original Frame)")
        print("="*80)
        
        for finger, wrench in self.finger_forces.items():
            if wrench is not None:
                f = wrench.force
                t = wrench.torque
                f_mag = math.sqrt(f.x**2 + f.y**2 + f.z**2)
                t_mag = math.sqrt(t.x**2 + t.y**2 + t.z**2)
                
                print(f"{finger.upper():8s} | Force: [{f.x:7.2f}, {f.y:7.2f}, {f.z:7.2f}] N  (|F| = {f_mag:7.2f} N)")
                if t_mag > 0.001:
                    print(f"         | Torque: [{t.x:7.3f}, {t.y:7.3f}, {t.z:7.3f}] Nm (|T| = {t_mag:7.3f} Nm)")
        
        print("\n" + "="*80)
        print("WRIST WRENCH (Transformed & Aggregated)")
        print("="*80)
        
        if self.wrist_wrench is not None:
            f = self.wrist_wrench.force
            t = self.wrist_wrench.torque
            f_mag = math.sqrt(f.x**2 + f.y**2 + f.z**2)
            t_mag = math.sqrt(t.x**2 + t.y**2 + t.z**2)
            
            print(f"TOTAL    | Force:  [{f.x:7.2f}, {f.y:7.2f}, {f.z:7.2f}] N  (|F| = {f_mag:7.2f} N)")
            print(f"         | Torque: [{t.x:7.3f}, {t.y:7.3f}, {t.z:7.3f}] Nm (|T| = {t_mag:7.3f} Nm)")
        else:
            print("No wrist wrench data received yet")
        
        print("="*80 + "\n")


def main(args=None):
    rclpy.init(args=args)
    
    monitor = WrenchMonitor()
    
    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
