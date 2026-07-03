from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import FindExecutable, PathJoinSubstitution, Command
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('aidin_hand_description')

    node = Node(
        package='aidin_hand_description',
        executable='wrench_zeroset.py',
        name='wrench_zeroset',
        output='screen',
        parameters=[{
            'hand_prefix': 'left_',
            'input_topic': '/left_ft_sensor_broadcaster/wrench',
            'output_topic': '/left_wrench_zeroset',
            'zero_trigger_topic': '/left_zeroset',
            'zero_trigger_value': 1,
            'sample_count': 50,
            'zero_on_start': False,
            'finger_names': ['thumb', 'index', 'middle', 'ring', 'baby'],
            'publish_when_uninitialized': True,
        }],
    )

    return LaunchDescription([node])
