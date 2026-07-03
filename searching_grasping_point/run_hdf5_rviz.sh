#!/bin/bash
# HDF5 + Force RViz Visualization Launcher
# This script isolates the playback from the live robot by using /hdf5_ prefixed topics.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

source /opt/ros/humble/setup.bash

echo "[1/3] Starting isolated robot_state_publisher..."
# Publish to /hdf5_robot_description and subscribe to /hdf5_joint_states
# Map TFs to isolated topics so they don't mess up live robot
ros2 run robot_state_publisher robot_state_publisher \
  --ros-args \
  -p robot_description:="$(cat hand_abs_with_sensors.urdf)" \
  -r /robot_description:=/hdf5_robot_description \
  -r /joint_states:=/hdf5_joint_states \
  -r /tf:=/hdf5_tf \
  -r /tf_static:=/hdf5_tf_static > /dev/null 2>&1 &
RSP_PID=$!

echo "[2/3] Starting RViz2..."
# Open RViz mapping its tf topics to the isolated ones
rviz2 -d hdf5_force_rviz.rviz \
  --ros-args \
  -r /tf:=/hdf5_tf \
  -r /tf_static:=/hdf5_tf_static > /dev/null 2>&1 &
RVIZ_PID=$!

echo "[3/3] Starting HDF5 Publisher..."
# Play the HDF5 demo
python3 hdf5_rviz_publisher.py common_data_plug.hdf5 --demo 3 --speed 1.0 --force-scale 0.02 --loop

# Cleanup when python script is stopped via Ctrl+C
echo "Cleaning up..."
kill $RSP_PID
kill $RVIZ_PID
exit 0
