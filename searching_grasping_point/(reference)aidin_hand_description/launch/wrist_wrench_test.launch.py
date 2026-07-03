import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution, Command, LaunchConfiguration, FindExecutable
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    description_package = FindPackageShare('aidin_hand_description')
    
    # Robot Description - Left Hand Only
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([description_package, "urdf", 'hand.urdf.xacro']),
            " use_right_hand:=false",
            " use_left_hand:=true",
            " right_hand_prefix:=right_",
            " left_hand_prefix:=left_",
            " right_hand_parent:=world",
            " left_hand_parent:=world",
            " right_hand_xyz:='0 0 0'",
            " left_hand_xyz:='0 0 0'",
            " right_hand_rpy:='0 0 0'",
            " left_hand_rpy:='0 0 0'",
        ]
    )
    robot_description = {'robot_description': robot_description_content}

    # Robot State Publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    # Joint State Publisher (simple, no GUI)
    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        output='screen'
    )

    # Wrist Wrench Aggregator Node (손목 힘/토크 집계)
    wrist_wrench_aggregator_node = Node(
        package='aidin_hand_description',
        executable='wrist_wrench_aggregator.py',
        name='wrist_wrench_aggregator',
        output='screen',
        parameters=[{
            'hand_prefix': 'left_',
            'wrist_frame': 'left_hand_base_link',
            'finger_tips': [
                'left_thumb',
                'left_index', 
                'left_middle',
                'left_ring',
                'left_baby'
            ]
        }]
    )

    # Force Visualizer Node (개별 손가락 힘 화살표 시각화)
    force_visualizer_node = Node(
        package='aidin_hand_description',
        executable='force_visualizer.py',
        name='force_visualizer',
        output='screen',
        parameters=[{
            'hand_prefix': 'left_',
            'force_scale': 0.01,  # 1N = 0.01m (10mm)
            'arrow_diameter': 0.005
        }]
    )

    # Aggregated Wrench Visualizer Node (손목 집계 힘/토크 화살표 시각화)
    aggregated_wrench_visualizer_node = Node(
        package='aidin_hand_description',
        executable='aggregated_wrench_visualizer.py',
        name='aggregated_wrench_visualizer',
        output='screen',
        parameters=[{
            'hand_prefix': 'left_',
            'wrist_frame': 'left_hand_base_link',
            'force_scale': 0.01,  # 1N = 0.01m (10mm)
            'torque_scale': 0.1,  # 1Nm = 0.1m (100mm)
            'force_arrow_diameter': 0.008,
            'torque_arrow_diameter': 0.006
        }]
    )

    # Hand Control GUI (joint angle control)
    hand_control_gui_node = Node(
        package='aidin_hand_description',
        executable='hand_control_gui.py',
        name='hand_control_gui',
        output='screen'
    )

    # Force Control GUI (slider로 힘 조절)
    force_control_gui_node = Node(
        package='aidin_hand_description',
        executable='force_control_gui.py',
        name='force_control_gui',
        output='screen',
        parameters=[{
            'hand_prefix': 'left_',
            'finger_tips': [
                'left_link4_thumb',
                'left_link4_index',
                'left_link4_middle',
                'left_link4_ring',
                'left_link4_baby'
            ]
        }]
    )

    # RViz
    rviz_config_file = PathJoinSubstitution([description_package, 'rviz', 'default.rviz'])
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file]
    )

    nodes = [
        robot_state_publisher_node,
        joint_state_publisher_node,
        wrist_wrench_aggregator_node,
        force_visualizer_node,
        aggregated_wrench_visualizer_node,
        hand_control_gui_node,
        force_control_gui_node,
        rviz_node
    ]

    return LaunchDescription(nodes)
