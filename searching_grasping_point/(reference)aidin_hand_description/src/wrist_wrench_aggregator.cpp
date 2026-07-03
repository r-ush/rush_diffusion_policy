#include "aidin_hand_description/wrist_wrench_aggregator.hpp"
#include <tf2_eigen/tf2_eigen.hpp>

namespace aidin_hand_description
{

WristWrenchAggregator::WristWrenchAggregator(const rclcpp::NodeOptions & options)
: Node("wrist_wrench_aggregator", options)
{
  // Declare and get parameters
  this->declare_parameter<std::string>("hand_prefix", "left_");
  this->declare_parameter<std::string>("wrist_frame", "left_hand_base_link");
  this->declare_parameter<std::vector<std::string>>("finger_tips", 
    std::vector<std::string>{
      "left_link4_thumb",
      "left_link4_index",
      "left_link4_middle",
      "left_link4_ring",
      "left_link4_baby"
    });

  this->get_parameter("hand_prefix", hand_prefix_);
  this->get_parameter("wrist_frame", wrist_frame_);
  this->get_parameter("finger_tips", finger_tips_);

  // Initialize TF
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Initialize finger wrench storage
  for (const auto & tip : finger_tips_) {
    finger_wrenches_[tip] = geometry_msgs::msg::Wrench();
    zero_offsets_[tip] = geometry_msgs::msg::Wrench();  // Initialize zero offsets
    zero_accumulator_[tip] = geometry_msgs::msg::Wrench();  // Initialize accumulator
  }
  zero_offset_initialized_ = false;
  zeroset_counter_ = 0;
  zeroset_average_num_ = 50;  // Average over 50 samples like reference code
  zeroset_active_ = false;  // Zeroset not active initially
  data_ready_ = false;
  last_publish_time_ = this->now();
  min_publish_interval_ = 0.02;  // 50Hz max (20ms minimum interval)

  // Create subscriber for FT sensor broadcaster
  // Topic name: /left_ft_sensor_broadcaster/wrench or /right_ft_sensor_broadcaster/wrench
  std::string hand_side = hand_prefix_;
  if (hand_side.back() == '_') {
    hand_side = hand_side.substr(0, hand_side.length() - 1);
  }
  std::string ft_topic = "/" + hand_side + "_ft_sensor_broadcaster/wrench";
  
  ft_sensor_sub_ = this->create_subscription<sensor_msgs::msg::MultiDOFJointState>(
    ft_topic, 10,
    std::bind(&WristWrenchAggregator::ftSensorCallback, this, std::placeholders::_1));
  
  RCLCPP_INFO(this->get_logger(), "Subscribed to %s", ft_topic.c_str());

  // Joint state subscriber (for debugging/monitoring)
  joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states", 10,
    std::bind(&WristWrenchAggregator::jointStateCallback, this, std::placeholders::_1));

  // Zeroset subscriber for sensor calibration
  std::string zeroset_topic = "/" + hand_side + "_zeroset";
  zeroset_sub_ = this->create_subscription<std_msgs::msg::UInt8>(
    zeroset_topic, 10,
    std::bind(&WristWrenchAggregator::zerosetCallback, this, std::placeholders::_1));
  
  RCLCPP_INFO(this->get_logger(), "Subscribed to %s for sensor calibration", zeroset_topic.c_str());

  // Publisher for aggregated wrench
  std::string aggregated_topic = "/" + hand_side + "_aggregated_wrench";
  aggregated_wrench_pub_ = this->create_publisher<sensor_msgs::msg::MultiDOFJointState>(
    aggregated_topic, 10);
  
  RCLCPP_INFO(this->get_logger(), "Publishing aggregated wrench to %s", aggregated_topic.c_str());

  RCLCPP_INFO(this->get_logger(), "Wrist Wrench Aggregator initialized");
  RCLCPP_INFO(this->get_logger(), "Wrist frame: %s", wrist_frame_.c_str());
  RCLCPP_INFO(this->get_logger(), "Monitoring %zu finger tips", finger_tips_.size());
}

void WristWrenchAggregator::ftSensorCallback(
  const sensor_msgs::msg::MultiDOFJointState::SharedPtr msg)
{
  // Lock mutex for thread-safe access
  std::lock_guard<std::mutex> lock(wrench_mutex_);
  
  // Reset data_ready flag and clear received fingers list
  data_ready_ = false;
  received_fingers_.clear();
  
  // Sensor to RViz coordinate transformation matrix
  // This matrix transforms sensor frame to match RViz coordinate system
  Eigen::Matrix3d sensor_to_rviz;
  sensor_to_rviz <<  0,  0,  1,
                    -1,  0,  0,
                     0, -1,  0;
  
  // Parse the MultiDOFJointState message and store wrenches for each finger
  for (size_t i = 0; i < msg->joint_names.size() && i < msg->wrench.size(); ++i) {
    const std::string & joint_name = msg->joint_names[i];  // e.g., "thumb", "index"
    const auto & wrench = msg->wrench[i];
    
    // Transform force and torque from sensor frame to RViz frame
    Eigen::Vector3d f_sensor(wrench.force.x, wrench.force.y, wrench.force.z);
    Eigen::Vector3d tau_sensor(wrench.torque.x, wrench.torque.y, wrench.torque.z);
    
    Eigen::Vector3d f_rviz = sensor_to_rviz * f_sensor;
    Eigen::Vector3d tau_rviz = sensor_to_rviz * tau_sensor;
    
    // Create transformed wrench in RViz coordinates
    geometry_msgs::msg::Wrench transformed_wrench;
    transformed_wrench.force.x = f_rviz(0);
    transformed_wrench.force.y = f_rviz(1);
    transformed_wrench.force.z = f_rviz(2);
    transformed_wrench.torque.x = tau_rviz(0);
    transformed_wrench.torque.y = tau_rviz(1);
    transformed_wrench.torque.z = tau_rviz(2);
    
    // Create full finger tip name: e.g., "left_link4_thumb"
    std::string finger_tip = hand_prefix_ + "link4_" + joint_name;
    
    // Store the transformed wrench
    finger_wrenches_[finger_tip] = transformed_wrench;
    
    // Store frame_id (use link4 frame as it's the actual fingertip link)
    finger_frame_ids_[finger_tip] = hand_prefix_ + "link4_" + joint_name;
    
    // Track that this finger was received in current message
    received_fingers_.push_back(finger_tip);
  }
  
  // Set data ready flag when all fingers received (5 fingers)
  data_ready_ = (msg->joint_names.size() == finger_tips_.size());
  
  // Safety check: verify received_fingers_ size matches
  if (data_ready_ && received_fingers_.size() != finger_tips_.size()) {
    RCLCPP_WARN(this->get_logger(), 
                "Mismatch: msg has %zu fingers but received_fingers_ has %zu",
                msg->joint_names.size(), received_fingers_.size());
    data_ready_ = false;
  }
  
  // If zeroset is active, accumulate samples automatically
  if (zeroset_active_ && data_ready_) {
    // Only accumulate fingers received in THIS message
    for (const auto & finger_tip : received_fingers_) {
      const auto & wrench = finger_wrenches_[finger_tip];
      auto & accumulator = zero_accumulator_[finger_tip];
      accumulator.force.x += wrench.force.x / zeroset_average_num_;
      accumulator.force.y += wrench.force.y / zeroset_average_num_;
      accumulator.force.z += wrench.force.z / zeroset_average_num_;
      accumulator.torque.x += wrench.torque.x / zeroset_average_num_;
      accumulator.torque.y += wrench.torque.y / zeroset_average_num_;
      accumulator.torque.z += wrench.torque.z / zeroset_average_num_;
    }
    
    zeroset_counter_++;
    
    // When averaging complete, store as zero offset
    if (zeroset_counter_ >= zeroset_average_num_) {
      zero_offsets_ = zero_accumulator_;
      zero_offset_initialized_ = true;
      zeroset_active_ = false;  // Deactivate zeroset
      zeroset_counter_ = 0;
      
      RCLCPP_INFO(this->get_logger(), "Zero offset calibration completed!");
      RCLCPP_INFO(this->get_logger(), "Averaged %d samples as zero reference", 
                  zeroset_average_num_);
    } else {
      // Log progress
      if (zeroset_counter_ % 10 == 0) {
        RCLCPP_INFO(this->get_logger(), "Zeroset progress: %d/%d samples", 
                    zeroset_counter_, zeroset_average_num_);
      }
    }
  }
  
  // Compute and publish wrist wrench immediately when all finger data received
  // This prevents race condition with timer-based approach
  // But throttle to max 50Hz even though input is 250Hz
  if (data_ready_ && !zeroset_active_) {
    auto current_time = this->now();
    double time_diff = (current_time - last_publish_time_).seconds();
    
    if (time_diff >= min_publish_interval_) {
      data_ready_ = false;  // Prevent duplicate publish
      last_publish_time_ = current_time;
      computeWristWrench();
    }
  }
}

void WristWrenchAggregator::jointStateCallback(
  const sensor_msgs::msg::JointState::SharedPtr msg)
{
  // For debugging/monitoring (currently not used)
  (void)msg;
}

// # 3. Zero calibration 실행 (별도 터미널) - ONE TIME trigger
// ros2 topic pub --once /left_zeroset std_msgs/msg/UInt8 "{data: 1}"
void WristWrenchAggregator::zerosetCallback(
  const std_msgs::msg::UInt8::SharedPtr msg)
{
  if (msg->data == 1) {
    std::lock_guard<std::mutex> lock(wrench_mutex_);
    
    // Initialize accumulator and start zeroset process
    for (auto & [finger_tip, accumulator] : zero_accumulator_) {
      accumulator.force.x = 0.0;
      accumulator.force.y = 0.0;
      accumulator.force.z = 0.0;
      accumulator.torque.x = 0.0;
      accumulator.torque.y = 0.0;
      accumulator.torque.z = 0.0;
    }
    
    zeroset_counter_ = 0;
    zero_offset_initialized_ = false;
    zeroset_active_ = true;  // Activate automatic averaging
    
    RCLCPP_INFO(this->get_logger(), "Starting zero offset calibration...");
    RCLCPP_INFO(this->get_logger(), "Will average %d samples automatically", 
                zeroset_average_num_);
  } else {
    RCLCPP_WARN(this->get_logger(), "Invalid zeroset message received (expected: 1, got: %d)", 
                msg->data);
  }
}

Eigen::Matrix3d WristWrenchAggregator::quaternionToRotationMatrix(
  const geometry_msgs::msg::Quaternion & q)
{
  // Convert quaternion to rotation matrix
  double qx = q.x;
  double qy = q.y;
  double qz = q.z;
  double qw = q.w;

  Eigen::Matrix3d R;
  R(0, 0) = 1 - 2 * (qy * qy + qz * qz);
  R(0, 1) = 2 * (qx * qy - qz * qw);
  R(0, 2) = 2 * (qx * qz + qy * qw);

  R(1, 0) = 2 * (qx * qy + qz * qw);
  R(1, 1) = 1 - 2 * (qx * qx + qz * qz);
  R(1, 2) = 2 * (qy * qz - qx * qw);

  R(2, 0) = 2 * (qx * qz - qy * qw);
  R(2, 1) = 2 * (qy * qz + qx * qw);
  R(2, 2) = 1 - 2 * (qx * qx + qy * qy);

  return R;
}

bool WristWrenchAggregator::transformWrench(
  const geometry_msgs::msg::Wrench & wrench,
  const std::string & source_frame,
  const std::string & target_frame,
  geometry_msgs::msg::Wrench & transformed_wrench)
{
  try {

    // Get transform from source frame to target frame
    geometry_msgs::msg::TransformStamped transform_stamped;
    transform_stamped = tf_buffer_->lookupTransform(
        target_frame,    // "left_hand_base_link" (손목)
        source_frame,    // "left_thumb_tip" (손가락 끝)
        tf2::TimePointZero,
        tf2::durationFromSec(0.0)
    );

    // Extract rotation and translation
    auto q = transform_stamped.transform.rotation;
    auto t = transform_stamped.transform.translation;

    // Convert quaternion to rotation matrix
    Eigen::Matrix3d R = quaternionToRotationMatrix(q);

    // Original force and torque
    Eigen::Vector3d f_orig(
      wrench.force.x,
      wrench.force.y,
      wrench.force.z
    );

    Eigen::Vector3d tau_orig(
      wrench.torque.x,
      wrench.torque.y,
      wrench.torque.z
    );

    // Transform force: f_new = R * f_orig
    Eigen::Vector3d f_new = R * f_orig;

    // Transform torque: tau_new = R * tau_orig + r × (R * f_orig)
    // where r is the position vector from target frame to source frame
    Eigen::Vector3d r(t.x, t.y, t.z);
    Eigen::Vector3d tau_new = R * tau_orig + r.cross(f_new);

    // Create transformed wrench
    transformed_wrench.force.x = f_new(0);
    transformed_wrench.force.y = f_new(1);
    transformed_wrench.force.z = f_new(2);

    transformed_wrench.torque.x = tau_new(0);
    transformed_wrench.torque.y = tau_new(1);
    transformed_wrench.torque.z = tau_new(2);

    return true;

  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(),
      *this->get_clock(),
      2000,  // 2 seconds
      "Could not transform wrench from %s to %s: %s",
      source_frame.c_str(),
      target_frame.c_str(),
      ex.what());
    return false;
  }
}

void WristWrenchAggregator::computeWristWrench()
{
  // NOTE: This function is called from ftSensorCallback which already holds the mutex
  // So we don't need to lock again here
  
  // Initialize total force and torque
  Eigen::Vector3d total_force = Eigen::Vector3d::Zero();
  Eigen::Vector3d total_torque = Eigen::Vector3d::Zero();

  int valid_count = 0;

  // Transform and aggregate wrenches ONLY from fingers received in current message
  for (const auto & finger_tip : received_fingers_) {
    const auto & wrench = finger_wrenches_[finger_tip];
    // Apply zero offset calibration if initialized
    geometry_msgs::msg::Wrench calibrated_wrench = wrench;
    if (zero_offset_initialized_ && zero_offsets_.find(finger_tip) != zero_offsets_.end()) {
      const auto & offset = zero_offsets_[finger_tip];
      calibrated_wrench.force.x -= offset.force.x;
      calibrated_wrench.force.y -= offset.force.y;
      calibrated_wrench.force.z -= offset.force.z;
      calibrated_wrench.torque.x -= offset.torque.x;
      calibrated_wrench.torque.y -= offset.torque.y;
      calibrated_wrench.torque.z -= offset.torque.z;
    }
    
    // Get frame_id for this finger
    std::string source_frame = finger_tip;  // Use finger_tip as frame_id
    if (finger_frame_ids_.find(finger_tip) != finger_frame_ids_.end()) {
      source_frame = finger_frame_ids_[finger_tip];
    }

    // Transform wrench to wrist frame
    geometry_msgs::msg::Wrench transformed_wrench;
    if (transformWrench(calibrated_wrench, source_frame, wrist_frame_, transformed_wrench)) {
      // Add to total force
      total_force(0) += transformed_wrench.force.x;
      total_force(1) += transformed_wrench.force.y;
      total_force(2) += transformed_wrench.force.z;

      // Add to total torque
      total_torque(0) += transformed_wrench.torque.x;
      total_torque(1) += transformed_wrench.torque.y;
      total_torque(2) += transformed_wrench.torque.z;

      valid_count++;
    }
  }

  // Publish aggregated wrench ONLY if we have ALL finger data
  // This prevents publishing incomplete data that causes value jumps
  if (valid_count != static_cast<int>(finger_tips_.size())) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(),
      *this->get_clock(),
      1000,  // 1 second
      "Cannot publish: valid_count=%d, expected=%zu (received_fingers=%zu)",
      valid_count, finger_tips_.size(), received_fingers_.size());
    return;
  }
  
  if (valid_count == static_cast<int>(finger_tips_.size())) {
    // RViz hand frame to robot wrist frame transformation matrix
    // This matrix aligns the RViz visualization with the actual robot wrist frame
    Eigen::Matrix3d rviz_to_robot_wrist;
    rviz_to_robot_wrist << -1,  0,  0,
                            0, -1,  0,
                            0,  0,  1;
    
    // Apply transformation to total force and torque
    Eigen::Vector3d final_force = rviz_to_robot_wrist * total_force;
    Eigen::Vector3d final_torque = rviz_to_robot_wrist * total_torque;
    
    sensor_msgs::msg::MultiDOFJointState aggregated_msg;
    aggregated_msg.header.stamp = this->now();
    aggregated_msg.header.frame_id = wrist_frame_;

    // Single joint name for aggregated wrench
    aggregated_msg.joint_names.push_back("wrist");

    // Create the aggregated wrench in robot wrist frame
    geometry_msgs::msg::Wrench aggregated_wrench;
    aggregated_wrench.force.x = final_force(0);
    aggregated_wrench.force.y = final_force(1);
    aggregated_wrench.force.z = final_force(2);

    aggregated_wrench.torque.x = final_torque(0);
    aggregated_wrench.torque.y = final_torque(1);
    aggregated_wrench.torque.z = final_torque(2);

    aggregated_msg.wrench.push_back(aggregated_wrench);

    aggregated_wrench_pub_->publish(aggregated_msg);

    // Log for debugging (throttled)
    double force_mag = final_force.norm();
    double torque_mag = final_torque.norm();
    
    RCLCPP_DEBUG_THROTTLE(
      this->get_logger(),
      *this->get_clock(),
      1000,  // 1 second
      "Wrist wrench - Force: %.3f N, Torque: %.3f Nm",
      force_mag, torque_mag);
  } else {
    // Log warning if not all fingers have valid transforms
    RCLCPP_WARN_THROTTLE(
      this->get_logger(),
      *this->get_clock(),
      5000,  // 5 seconds
      "Incomplete finger data: %d/%zu fingers valid. Skipping publish. (received_fingers: %zu)",
      valid_count, finger_tips_.size(), received_fingers_.size());
    
    // Also log which fingers failed
    for (const auto & finger_tip : received_fingers_) {
      if (finger_frame_ids_.find(finger_tip) == finger_frame_ids_.end()) {
        RCLCPP_DEBUG(this->get_logger(), "Missing frame_id for: %s", finger_tip.c_str());
      }
    }
  }
}

}  // namespace aidin_hand_description

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(aidin_hand_description::WristWrenchAggregator)
