import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution, Command, LaunchConfiguration, FindExecutable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from launch.launch_context import LaunchContext
from typing import List
from launch.conditions import IfCondition

ARGUMENTS = [
    DeclareLaunchArgument(
        'use_right_hand',
        default_value='true',
        description='Use Right Hand'
    ),  
    DeclareLaunchArgument(
        'use_left_hand',
        default_value='true',
        description='Use Left Hand'
    ),
    DeclareLaunchArgument(
        'right_hand_prefix',
        default_value='right_',
        description='Right Hand Prefix'
    ),
    DeclareLaunchArgument(
        'left_hand_prefix',
        default_value='left_',
        description='Left Hand Prefix'
    ),
    DeclareLaunchArgument(
        'right_hand_parent',
        default_value='world',
        description='Right Hand Parent'
    ),
    DeclareLaunchArgument(
        'left_hand_parent',
        default_value='world',
        description='Left Hand Parent'
    ),
    DeclareLaunchArgument(
        'right_hand_xyz',
        default_value="'0.1 0 0'",  
        description='Right Hand XYZ'
    ),
    DeclareLaunchArgument(
        'left_hand_xyz',
        default_value="'-0.1 0 0'",
        description='Left Hand XYZ'
    ),
    DeclareLaunchArgument(
        'right_hand_rpy',
        default_value="'0 0 0'",
        description='Right Hand RPY'
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
        description='Use Joint State Publisher'
    )
    ]	

def generate_launch_description():
    description_package = FindPackageShare('aidin_hand_description')
    

    # RViz
    rviz_config_file = get_package_share_directory('aidin_hand_description') + "/rviz/default.rviz"
    rviz_node = Node(package='rviz2',
                     executable='rviz2',
                     name='rviz2',
                     output='log',
                     arguments=['-d', rviz_config_file],
                     condition=IfCondition(LaunchConfiguration("use_rviz"))
                     )
    
    # Robot Description
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([description_package, "urdf", 'hand.urdf.xacro']),
            " use_right_hand:=", LaunchConfiguration("use_right_hand"),
            " use_left_hand:=", LaunchConfiguration("use_left_hand"),
            " right_hand_prefix:=", LaunchConfiguration("right_hand_prefix"),
            " left_hand_prefix:=", LaunchConfiguration("left_hand_prefix"),
            " right_hand_parent:=", LaunchConfiguration("right_hand_parent"),
            " left_hand_parent:=", LaunchConfiguration("left_hand_parent"),
            " right_hand_xyz:=", LaunchConfiguration("right_hand_xyz"),
            " left_hand_xyz:=", LaunchConfiguration("left_hand_xyz"),
            " right_hand_rpy:=", LaunchConfiguration("right_hand_rpy"),
            " left_hand_rpy:=", LaunchConfiguration("left_hand_rpy"),
        ]
    )
    robot_description = {'robot_description': robot_description_content}

    # Robot State Publisher GUI
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    # Robot Joint State Publisher GUI
    joint_state_publisher_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        output='screen',
        condition=IfCondition(LaunchConfiguration("use_joint_state_publisher"))
    )

    nodes = [
        robot_state_publisher_node,
        joint_state_publisher_node,
        rviz_node
    ]

    return LaunchDescription(ARGUMENTS + nodes)
