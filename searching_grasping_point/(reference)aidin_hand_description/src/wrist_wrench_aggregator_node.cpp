#include "aidin_hand_description/wrist_wrench_aggregator.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  
  auto node = std::make_shared<aidin_hand_description::WristWrenchAggregator>();
  
  rclcpp::spin(node);
  
  rclcpp::shutdown();
  return 0;
}
