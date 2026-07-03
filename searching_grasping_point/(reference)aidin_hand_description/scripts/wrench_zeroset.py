#!/usr/bin/env python3

# ros2 topic pub --once /left_zeroset std_msgs/msg/UInt8 "{data: 1}"

import copy
from collections import defaultdict

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench
from sensor_msgs.msg import MultiDOFJointState
from std_msgs.msg import UInt8


class WrenchZeroSet(Node):
    def __init__(self):
        super().__init__('wrench_zeroset')

        self.declare_parameter('hand_prefix', 'left_')
        self.declare_parameter('input_topic', '')
        self.declare_parameter('output_topic', '')
        self.declare_parameter('zero_trigger_topic', '')
        self.declare_parameter('zero_trigger_value', 1)
        self.declare_parameter('sample_count', 50)
        self.declare_parameter('zero_on_start', False)
        self.declare_parameter('finger_names', ['thumb', 'index', 'middle', 'ring', 'baby'])
        self.declare_parameter('publish_when_uninitialized', True)

        self.hand_prefix = self.get_parameter('hand_prefix').value
        self.sample_count = int(self.get_parameter('sample_count').value)
        self.zero_on_start = bool(self.get_parameter('zero_on_start').value)
        self.publish_when_uninitialized = bool(
            self.get_parameter('publish_when_uninitialized').value)

        self.finger_names = list(self.get_parameter('finger_names').value)
        self.expected_joint_names = [str(name) for name in self.finger_names]

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        zero_trigger_topic = self.get_parameter('zero_trigger_topic').value
        zero_trigger_value = int(self.get_parameter('zero_trigger_value').value)

        hand_side = self.hand_prefix[:-1] if self.hand_prefix.endswith('_') else self.hand_prefix
        if not input_topic:
            input_topic = f'/{hand_side}_ft_sensor_broadcaster/wrench'
        if not output_topic:
            output_topic = f'/{hand_side}_wrench_zeroset'
        if not zero_trigger_topic:
            zero_trigger_topic = f'/{hand_side}_zeroset'

        self.input_topic = input_topic
        self.output_topic = output_topic
        self.zero_trigger_topic = zero_trigger_topic
        self.zero_trigger_value = zero_trigger_value

        self._zeroing_active = False
        self._sample_index = 0
        self._offset_initialized = False

        self._force_sum = defaultdict(lambda: [0.0, 0.0, 0.0])
        self._torque_sum = defaultdict(lambda: [0.0, 0.0, 0.0])
        self._sample_count_by_finger = defaultdict(int)
        self._force_offset = defaultdict(lambda: [0.0, 0.0, 0.0])
        self._torque_offset = defaultdict(lambda: [0.0, 0.0, 0.0])

        self.input_sub = self.create_subscription(
            MultiDOFJointState,
            self.input_topic,
            self.input_callback,
            10,
        )
        self.trigger_sub = self.create_subscription(
            UInt8,
            self.zero_trigger_topic,
            self.trigger_callback,
            10,
        )
        self.output_pub = self.create_publisher(MultiDOFJointState, self.output_topic, 10)

        self.get_logger().info(f'Subscribed to {self.input_topic}')
        self.get_logger().info(f'Publishing zeroed wrench to {self.output_topic}')
        self.get_logger().info(
            f'Zero trigger topic: {self.zero_trigger_topic} (value={self.zero_trigger_value})')
        self.get_logger().info(f'Sample count: {self.sample_count}')
        self.get_logger().info(f'Finger names: {self.expected_joint_names}')

        if self.zero_on_start:
            self.start_zeroing('zero_on_start')

    def start_zeroing(self, reason: str):
        self._zeroing_active = True
        self._sample_index = 0
        for name in self.expected_joint_names:
            self._force_sum[name] = [0.0, 0.0, 0.0]
            self._torque_sum[name] = [0.0, 0.0, 0.0]
            self._sample_count_by_finger[name] = 0
        self.get_logger().info(
            f'Starting zero set ({reason}); collecting {self.sample_count} samples')

    def trigger_callback(self, msg: UInt8):
        if msg.data != self.zero_trigger_value:
            return
        if self._zeroing_active:
            self.get_logger().info('Zero set already active; ignoring trigger')
            return
        self.start_zeroing('trigger')

    def input_callback(self, msg: MultiDOFJointState):
        joint_to_wrench = {}
        for idx, joint_name in enumerate(msg.joint_names):
            if idx >= len(msg.wrench):
                break
            joint_to_wrench[str(joint_name)] = msg.wrench[idx]

        if self._zeroing_active:
            self._accumulate_sample(joint_to_wrench)
            self._sample_index += 1

            if self._sample_index >= self.sample_count:
                self._finalize_zero_set()
            return

        if (not self._offset_initialized) and (not self.publish_when_uninitialized):
            return

        out_msg = self._build_zeroed_message(msg, joint_to_wrench)
        self.output_pub.publish(out_msg)

    def _accumulate_sample(self, joint_to_wrench):
        for name in self.expected_joint_names:
            wrench = joint_to_wrench.get(name)
            if wrench is None:
                continue

            self._force_sum[name][0] += float(wrench.force.x)
            self._force_sum[name][1] += float(wrench.force.y)
            self._force_sum[name][2] += float(wrench.force.z)
            self._torque_sum[name][0] += float(wrench.torque.x)
            self._torque_sum[name][1] += float(wrench.torque.y)
            self._torque_sum[name][2] += float(wrench.torque.z)
            self._sample_count_by_finger[name] += 1

    def _finalize_zero_set(self):
        valid_fingers = 0
        for name in self.expected_joint_names:
            count = max(1, int(self._sample_count_by_finger[name]))
            self._force_offset[name] = [value / float(count) for value in self._force_sum[name]]
            self._torque_offset[name] = [value / float(count) for value in self._torque_sum[name]]
            valid_fingers += 1

        self._zeroing_active = False
        self._offset_initialized = True

        self.get_logger().info(
            f'Zero set completed for {valid_fingers} finger tips using {self.sample_count} samples')
        for name in self.expected_joint_names:
            f = self._force_offset[name]
            t = self._torque_offset[name]
            self.get_logger().info(
                f'  {name}: force=[{f[0]:.3f}, {f[1]:.3f}, {f[2]:.3f}] '
                f'torque=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]')

    def _build_zeroed_message(self, src_msg: MultiDOFJointState, joint_to_wrench):
        out_msg = MultiDOFJointState()
        out_msg.header = copy.deepcopy(src_msg.header)
        out_msg.joint_names = list(src_msg.joint_names)
        out_msg.transforms = copy.deepcopy(src_msg.transforms)
        out_msg.twist = copy.deepcopy(src_msg.twist)

        for joint_name in src_msg.joint_names:
            wrench = joint_to_wrench.get(str(joint_name))
            if wrench is None:
                out_msg.wrench.append(Wrench())
                continue

            zeroed = copy.deepcopy(wrench)
            offset_f = self._force_offset[str(joint_name)]
            offset_t = self._torque_offset[str(joint_name)]
            zeroed.force.x = float(wrench.force.x) - offset_f[0]
            zeroed.force.y = float(wrench.force.y) - offset_f[1]
            zeroed.force.z = float(wrench.force.z) - offset_f[2]
            zeroed.torque.x = float(wrench.torque.x) - offset_t[0]
            zeroed.torque.y = float(wrench.torque.y) - offset_t[1]
            zeroed.torque.z = float(wrench.torque.z) - offset_t[2]
            out_msg.wrench.append(zeroed)

        return out_msg


def main(args=None):
    rclpy.init(args=args)
    node = WrenchZeroSet()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
