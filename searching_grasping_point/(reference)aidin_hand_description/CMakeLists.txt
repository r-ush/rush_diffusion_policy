cmake_minimum_required(VERSION 3.8)
project(aidin_hand_description)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(THIS_PACKAGE_INCLUDE_DEPENDS
  urdf
  xacro
  rclcpp
  rclcpp_components
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  Eigen3
)

# find dependencies
find_package(ament_cmake REQUIRED)
foreach(Dependency IN ITEMS ${THIS_PACKAGE_INCLUDE_DEPENDS})
  find_package(${Dependency} REQUIRED)
endforeach()

# Wrist Wrench Aggregator Component
add_library(wrist_wrench_aggregator_component SHARED
  src/wrist_wrench_aggregator.cpp
)

target_include_directories(wrist_wrench_aggregator_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(wrist_wrench_aggregator_component
  ${rclcpp_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${geometry_msgs_LIBRARIES}
  ${tf2_ros_LIBRARIES}
  ${tf2_eigen_LIBRARIES}
)

ament_target_dependencies(wrist_wrench_aggregator_component
  rclcpp
  rclcpp_components
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  Eigen3
)

rclcpp_components_register_node(wrist_wrench_aggregator_component
  PLUGIN "aidin_hand_description::WristWrenchAggregator"
  EXECUTABLE wrist_wrench_aggregator
)

# Install component library
install(TARGETS wrist_wrench_aggregator_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY meshes urdf launch rviz config
  DESTINATION share/${PROJECT_NAME})

# Install Python scripts
install(PROGRAMS
  scripts/wrist_wrench_aggregator.py
  scripts/wrench_zeroset.py
  scripts/aggregated_wrench_visualizer.py
  scripts/force_control_gui.py
  scripts/force_visualizer.py
  scripts/hand_control_gui.py
  scripts/monitor_wrench.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
