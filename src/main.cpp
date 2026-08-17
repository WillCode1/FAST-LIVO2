#include "LIVMapper.h"
#include "glogger.h"

#ifdef ROS1
DEFINE_string(camera_config, "config/camera_fisheye_HILTI22.yaml", "camera config file.");
#endif

int main(int argc, char **argv) {
  // 方便报错时找到出错的位置
#ifdef ROS1
  GLogger glogger(argc, argv, "", "");
  ros::init(argc, argv, "laserMapping");
  ros::NodeHandle nh;
#else
  GLogger glogger(1, argv, "", "");
  rclcpp::init(argc, argv);
  auto nh = std::make_shared<rclcpp::Node>("fast_livo2");
#endif
  image_transport::ImageTransport it(nh);
#ifdef ROS1
  LIVMapper mapper(nh,FLAGS_camera_config);
#else
  std::string camera_config;
  nh->declare_parameter("vio.camera_config", "");
  nh->get_parameter("vio.camera_config", camera_config);
  LIVMapper mapper(nh,camera_config);
#endif
  mapper.InitializeSubscribersAndPublishers(nh, it);
#ifdef ROS1
  mapper.Run();
#else
  mapper.Run(nh);
#endif
  return 0;
}