import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution, Command, LaunchConfiguration, FindExecutable
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition, UnlessCondition

ARGUMENTS = [
    DeclareLaunchArgument(
        'left_hand_prefix',
        default_value='left_',
        description='Left Hand Prefix'
    ),
    DeclareLaunchArgument(
        'left_hand_parent',
        default_value='world',
        description='Left Hand Parent'
    ),
    DeclareLaunchArgument(
        'left_hand_xyz',
        default_value="'0 0 0'",
        description='Left Hand XYZ'
    ),
    DeclareLaunchArgument(
        'left_hand_rpy',
        default_value="'0 0 0'",
        description='Left Hand RPY'
    ),
    DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Use RViz'
    ),
    DeclareLaunchArgument(
        'use_joint_state_publisher',
        default_value='true',
        description='Use Joint State Publisher GUI'
    ),
    DeclareLaunchArgument(
        'use_force_control_gui',
        default_value='true',
        description='Use Force Control GUI'
    ),
    DeclareLaunchArgument(
        'force_scale',
        default_value='0.1',
        description='Scale factor for force visualization (meters per Newton)'
    ),
    DeclareLaunchArgument(
        'force_arrow_diameter',
        default_value='0.005',
        description='Diameter of force arrows in meters'
    )
]	

def generate_launch_description():
    description_package = FindPackageShare('aidin_hand_description')
    
    # RViz with custom config for force visualization
    rviz_config_file = get_package_share_directory('aidin_hand_description') + "/rviz/default.rviz"
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        condition=IfCondition(LaunchConfiguration("use_rviz"))
    )
    
    # Robot Description - Left Hand Only
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([description_package, "urdf", 'hand.urdf.xacro']),
            " use_right_hand:=false",  # 오른손 비활성화
            " use_left_hand:=true",    # 왼손만 활성화
            " right_hand_prefix:=right_",
            " left_hand_prefix:=", LaunchConfiguration("left_hand_prefix"),
            " right_hand_parent:=world",
            " left_hand_parent:=", LaunchConfiguration("left_hand_parent"),
            " right_hand_xyz:='0 0 0'",
            " left_hand_xyz:=", LaunchConfiguration("left_hand_xyz"),
            " right_hand_rpy:='0 0 0'",
            " left_hand_rpy:=", LaunchConfiguration("left_hand_rpy"),
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

    # Hand Control GUI (손가락 관절 제어 - centering 진동 없음)
    hand_control_gui_node = Node(
        package='aidin_hand_description',
        executable='hand_control_gui.py',
        name='hand_control_gui',
        output='screen',
        condition=IfCondition(LaunchConfiguration("use_joint_state_publisher"))
    )
    
    # Joint State Publisher (GUI 없이 기본값 발행)
    # 주석 처리: hand_control_gui와 충돌하여 joint_states가 진동함
    # joint_state_publisher_node = Node(
    #     package='joint_state_publisher',
    #     executable='joint_state_publisher',
    #     output='screen',
    #     condition=UnlessCondition(LaunchConfiguration("use_joint_state_publisher"))
    # )

    # Force Control GUI (힘 제어 슬라이더)
    force_control_gui_node = Node(
        package='aidin_hand_description',
        executable='force_control_gui.py',
        name='force_control_gui',
        output='screen',
        condition=IfCondition(LaunchConfiguration("use_force_control_gui"))
    )

    # Force Visualization Node
    # 각 손가락 끝에서 받는 힘을 WrenchStamped 토픽으로 받아 시각화
    force_visualizer_node = Node(
        package='aidin_hand_description',
        executable='force_visualizer.py',
        name='left_hand_force_visualizer',
        output='screen',
        parameters=[{
            'hand_prefix': LaunchConfiguration('left_hand_prefix'),
            'force_scale': LaunchConfiguration('force_scale'),
            'arrow_diameter': LaunchConfiguration('force_arrow_diameter'),
            'finger_tips': [
                'left_link4_thumb',
                'left_link4_index', 
                'left_link4_middle',
                'left_link4_ring',
                'left_link4_baby'
            ]
        }]
    )

    # Wrist Wrench Aggregator Node
    # 각 손가락의 힘을 손목 프레임 기준으로 변환하여 총합 계산
    # Forward Kinematics를 고려하여 TF를 통해 변환
    wrist_wrench_aggregator_node = Node(
        package='aidin_hand_description',
        executable='wrist_wrench_aggregator.py',
        name='wrist_wrench_aggregator',
        output='screen',
        parameters=[{
            'hand_prefix': LaunchConfiguration('left_hand_prefix'),
            'wrist_frame': 'left_hand_base_link',
            'finger_tips': [
                'left_link4_thumb',
                'left_link4_index', 
                'left_link4_middle',
                'left_link4_ring',
                'left_link4_baby'
            ]
        }]
    )

    # Static transforms for finger tip frames (if not in URDF)
    # 손가락 끝 프레임이 URDF에 이미 정의되어 있으므로 static_transform_publisher 제거
    finger_tip_transforms = []
    
    # URDF에 이미 정의되어 있음:
    # left_thumb_tip (parent: left_thumb_link4)
    # left_index_tip (parent: left_index_link4)
    # left_middle_tip (parent: left_middle_link4)
    # left_ring_tip (parent: left_ring_link4)
    # left_baby_tip (parent: left_baby_link4)

    nodes = [
        robot_state_publisher_node,
        hand_control_gui_node,
        # joint_state_publisher_node,  # 주석 처리: hand_control_gui와 충돌
        force_control_gui_node,
        force_visualizer_node,
        wrist_wrench_aggregator_node,
        rviz_node
    ]

    return LaunchDescription(ARGUMENTS + nodes)
