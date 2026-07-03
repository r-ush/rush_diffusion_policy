#ifndef AIDIN_HAND_DESCRIPTION__WRIST_WRENCH_AGGREGATOR_HPP_
#define AIDIN_HAND_DESCRIPTION__WRIST_WRENCH_AGGREGATOR_HPP_

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/multi_dof_joint_state.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Dense>
#include <map>
#include <string>
#include <vector>
#include <mutex>

namespace aidin_hand_description
{

class WristWrenchAggregator : public rclcpp::Node
{
public:
  explicit WristWrenchAggregator(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~WristWrenchAggregator() = default;

private:
  // Parameters
  std::string hand_prefix_;
  std::string wrist_frame_;
  std::vector<std::string> finger_tips_;

  // TF
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // Storage for finger wrenches
  std::map<std::string, geometry_msgs::msg::Wrench> finger_wrenches_;
  std::map<std::string, std::string> finger_frame_ids_;
  std::vector<std::string> received_fingers_;  // Track which fingers in current message
  
  // Zero offset storage for sensor calibration
  std::map<std::string, geometry_msgs::msg::Wrench> zero_offsets_;
  std::map<std::string, geometry_msgs::msg::Wrench> zero_accumulator_;  // For averaging
  bool zero_offset_initialized_;
  int zeroset_counter_;
  int zeroset_average_num_;  // Number of samples for averaging (like reference code)
  bool zeroset_active_;  // Flag to indicate zeroset process is active
  bool data_ready_;  // Flag to indicate all finger data received
  
  // Thread safety
  std::mutex wrench_mutex_;
  
  // Timestamp tracking for throttling
  rclcpp::Time last_publish_time_;
  double min_publish_interval_;  // Minimum seconds between publishes

  // Subscribers
  rclcpp::Subscription<sensor_msgs::msg::MultiDOFJointState>::SharedPtr ft_sensor_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr zeroset_sub_;

  // Publisher
  rclcpp::Publisher<sensor_msgs::msg::MultiDOFJointState>::SharedPtr aggregated_wrench_pub_;

  // Callbacks
  void ftSensorCallback(const sensor_msgs::msg::MultiDOFJointState::SharedPtr msg);
  
  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
  
  void zerosetCallback(const std_msgs::msg::UInt8::SharedPtr msg);
  
  void computeWristWrench();

  // Transform functions
  bool transformWrench(
    const geometry_msgs::msg::Wrench & wrench,
    const std::string & source_frame,
    const std::string & target_frame,
    geometry_msgs::msg::Wrench & transformed_wrench);

  Eigen::Matrix3d quaternionToRotationMatrix(const geometry_msgs::msg::Quaternion & q);
};

}  // namespace aidin_hand_description

#endif  // AIDIN_HAND_DESCRIPTION__WRIST_WRENCH_AGGREGATOR_HPP_
