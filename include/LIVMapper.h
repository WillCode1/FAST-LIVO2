/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIV_MAPPER_H
#define LIV_MAPPER_H

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#ifdef ROS1
#include <nav_msgs/Path.h>
#else
#include <nav_msgs/msg/path.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#endif
#include <yaml_loader.h>

#include "IMU_Processing.h"
#include "preprocess.h"
#include "vio.h"

class LIVMapper {
 public:
#ifdef ROS1
  LIVMapper(ros::NodeHandle &nh, const std::string &camera_config);
  ~LIVMapper();
  void ReadParameters(ros::NodeHandle &nh);
  void InitializeSubscribersAndPublishers(ros::NodeHandle &nh,
                                          image_transport::ImageTransport &it);
  void Run();
#else
  LIVMapper(rclcpp::Node::SharedPtr &node, const std::string &camera_config);
  ~LIVMapper();
  void ReadParameters(rclcpp::Node::SharedPtr &node);
  void InitializeSubscribersAndPublishers(rclcpp::Node::SharedPtr &node,
                                          image_transport::ImageTransport &it);
  void Run(rclcpp::Node::SharedPtr &node);
#endif
  void InitializeComponents();
#ifdef ROS1
  void InitBackend(bool &save_globalmap_en);
#else
  void InitBackend(rclcpp::Node::SharedPtr &node, bool &save_globalmap_en);
#endif
  void InitializeFiles();
  void GravityAlignment();
  void HandleFirstFrame();
  void StateEstimationAndMapping();
  void HandleVIO();
  void HandleLIO();
  void SavePCD();
  void ProcessImu();

  bool SyncPackages(LidarMeasureGroup &meas);
  void PropImuOnce(StatesGroup &imu_prop_state, const double dt, V3D acc_avr,
                   V3D angvel_avr);
  void TransformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t,
                      const PointCloudXYZIN::Ptr &input_cloud,
                      PointCloudXYZIN::Ptr &trans_cloud);
  void PointBodyToWorld(const PointXYZIN &pi, PointXYZIN &po);
  void RGBpointBodyLidarToIMU(PointXYZIN const *const pi, PointXYZIN *const po);
  void RGBpointBodyToWorld(PointXYZIN const *const pi, PointXYZIN *const po);
#ifdef ROS1
  void ImuPropCallback(const ros::TimerEvent &e);
  void PointCloud2Cbk(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void LivoxCbk(const livox_ros_driver::CustomMsg::ConstPtr &msg_in);
  void ImuCbk(const sensor_msgs::Imu::ConstPtr &msg_in);
  void ImageCbk(const sensor_msgs::ImageConstPtr &msg_in);
  void PublishFrameWorld(const ros::Publisher &pubLaserCloudFullRes,
                         VIOManagerPtr vio_manager);
  void PublishVisualSubMap(const ros::Publisher &pubSubVisualMap);
  void PublishEffectWorld(const ros::Publisher &pubLaserCloudEffect,
                          const std::vector<PointToPlane> &ptpl_list);
  void PublishOdometry(const ros::Publisher &pubOdomAftMapped);
  void PublishMavros(const ros::Publisher &mavros_pose_publisher);
  void PublishPath(const ros::Publisher &pubPath);
  cv::Mat GetImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);
#else
  void ImuPropCallback();
  void PointCloud2Cbk(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void LivoxCbk(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg_in);
  void ImuCbk(const sensor_msgs::msg::Imu::SharedPtr msg_in);
  void ImageCbk(const sensor_msgs::msg::Image::SharedPtr msg_in);
  void PublishFrameWorld(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubLaserCloudFullRes,
                         VIOManagerPtr vio_manager);
  void PublishVisualSubMap(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubSubVisualMap);
  void PublishEffectWorld(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubLaserCloudEffect,
                          const std::vector<PointToPlane> &ptpl_list);
  void PublishOdometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr &pubOdomAftMapped);
  void PublishMavros(const rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr &mavros_pose_publisher);
  void PublishPath(const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr &pubPath);
  cv::Mat GetImageFromMsg(const sensor_msgs::msg::Image::SharedPtr &img_msg);
#endif
  void PublishImgRGB(const image_transport::Publisher &pubImage,
                     VIOManagerPtr vio_manager);
  template <typename T>
  void SetPosestamp(T &out);
  template <typename T>
  void PointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi,
                        Eigen::Matrix<T, 3, 1> &po);
  template <typename T>
  Eigen::Matrix<T, 3, 1> PointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi);

  std::mutex mtx_buffer_, mtx_buffer_imu_prop_;
  std::condition_variable sig_buffer_;

  SLAM_MODE slam_mode_;

  std::string root_dir_;
  std::string camera_config_;
  std::string lid_topic_, imu_topic_, seq_name_, img_topic_;
  V3D ext_t_;
  M3D ext_r_;

  int feats_down_size_ = 0, max_iterations_ = 0;

  double res_mean_last_ = 0.05;
  double gyr_cov_ = 0, acc_cov_ = 0, inv_expo_cov_ = 0;
  double blind_rgb_points_ = 0.0;
  double last_timestamp_lidar_ = -1.0, last_timestamp_imu_ = -1.0,
         last_timestamp_img_ = -1.0;
  double filter_size_surf_min_ = 0;
  double filter_size_pcd_ = 0;
  double first_lidar_time_ = 0.0;
  double match_time_ = 0, solve_time_ = 0, solve_const_H_time_ = 0;

  bool lidar_map_inited_ = false, pcd_save_en_ = false, img_save_en_ = false,
       pub_effect_point_en_ = false, pose_output_en_ = false,
       ros_driver_fix_en_ = false, hilti_en_ = false;
  int img_save_interval_ = 1,pcd_save_interval_ = -1, pcd_save_type_ = 0;
  int pub_scan_num_ = 1;

  StatesGroup imu_propagate_, latest_ekf_state_;

  bool new_imu_ = false, state_update_flg_ = false, imu_prop_enable_ = true,
       ekf_finish_once_ = false;
  double latest_ekf_time_;
#ifdef ROS1
  std::deque<sensor_msgs::Imu> prop_imu_buffer_;
  sensor_msgs::Imu newest_imu_;
  nav_msgs::Odometry imu_prop_odom_;
  std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer_;
#else
  std::deque<sensor_msgs::msg::Imu> prop_imu_buffer_;
  sensor_msgs::msg::Imu newest_imu_;
  nav_msgs::msg::Odometry imu_prop_odom_;
  std::deque<sensor_msgs::msg::Imu::SharedPtr> imu_buffer_;
#endif
  double imu_time_offset_ = 0.0;
  double lidar_time_offset_ = 0.0;

  bool gravity_align_en_ = false, gravity_align_finished_ = false;

  bool sync_jump_flag_ = false;

  bool lidar_pushed_ = false, gravity_est_en_, flg_reset_ = false,
       ba_bg_est_en_ = true;
  bool dense_map_en = false;
  int img_en_ = 1, imu_int_frame_ = 3;
  bool normal_en_ = true;
  bool exposure_estimate_en_ = false;
  double exposure_time_init_ = 0.0;
  bool inverse_composition_en_ = false;
  bool raycast_en_ = false;
  bool first_frame_finished_ = false;
  int grid_size_, patch_size_, patch_pyrimid_level_;
  double outlier_threshold_;
  double plot_time_;
  int frame_cnt_;
  double img_time_offset_ = 0.0;
  std::deque<PointCloudXYZIN::Ptr> lid_raw_data_buffer_;
  std::deque<double> lid_header_time_buffer_;
  std::deque<cv::Mat> img_buffer_;
  std::deque<double> img_time_buffer_;
  std::vector<pointWithVar> pv_list_;
  std::vector<double> extrin_t_;
  std::vector<double> extrin_r_;
  std::vector<double> cameraextrin_t_;
  std::vector<double> cameraextrin_r_;
  double img_point_cov_;

  PointCloudXYZIN::Ptr visual_sub_map_;
  PointCloudXYZIN::Ptr feats_undistort_;
  PointCloudXYZIN::Ptr feats_down_body_;
  PointCloudXYZIN::Ptr feats_down_world_;
  PointCloudXYZIN::Ptr pcl_w_wait_pub_;
  PointCloudXYZIN::Ptr pcl_wait_pub_;
  PointCloudXYZRGB::Ptr pcl_wait_save_;
  PointCloudXYZIN::Ptr pcl_wait_save_intensity_;

  std::ofstream fout_pre_, fout_out_, fout_visual_pos_, fout_lidar_pos_, fout_points_;

  pcl::VoxelGrid<PointXYZIN> downSize_filter_surf_;

  V3D euler_cur_;

  LidarMeasureGroup lidar_measures_;
  StatesGroup state_;
  StatesGroup state_propagat_;

#ifdef ROS1
  nav_msgs::Path path_;
  nav_msgs::Odometry odom_aft_mapped_;
  geometry_msgs::Quaternion geo_quat_;
  geometry_msgs::PoseStamped msg_body_pose_;
#else
  nav_msgs::msg::Path path_;
  nav_msgs::msg::Odometry odom_aft_mapped_;
  geometry_msgs::msg::Quaternion geo_quat_;
  geometry_msgs::msg::PoseStamped msg_body_pose_;
#endif

  PreprocessPtr p_pre_;
  ImuProcessPtr p_imu_;
  VoxelMapManagerPtr voxel_map_manager_;
  VIOManagerPtr vio_manager_;

#ifdef ROS1
  ros::Publisher plane_pub_;
  ros::Publisher voxel_pub_;
  ros::Subscriber sub_pcl_;
  ros::Subscriber sub_imu_;
  ros::Subscriber sub_img_;
  ros::Publisher pubLaser_cloud_full_res_;
  ros::Publisher pub_normal_;
  ros::Publisher pub_sub_visual_map_;
  ros::Publisher pub_laser_cloud_effect_;
  ros::Publisher pub_laser_cloud_map_;
  ros::Publisher pub_odom_aft_mapped_;
  ros::Publisher pub_path_;
  ros::Publisher pub_laser_cloud_dyn_;
  ros::Publisher pub_laser_cloud_dyn_rmed_;
  ros::Publisher pub_laser_cloud_dyn_dbg_;
  image_transport::Publisher pub_image_;
  ros::Publisher mavros_pose_publisher_;
  ros::Timer imu_prop_timer_;
  ros::Publisher pub_imu_prop_odom_;
#else
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr plane_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr voxel_pub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl1_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl2_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaser_cloud_full_res_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_normal_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_sub_visual_map_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_laser_cloud_effect_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_laser_cloud_map_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_aft_mapped_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_laser_cloud_dyn_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_laser_cloud_dyn_rmed_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_laser_cloud_dyn_dbg_;
  image_transport::Publisher pub_image_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mavros_pose_publisher_;
  rclcpp::TimerBase::SharedPtr imu_prop_timer_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_imu_prop_odom_;
#endif

  int frame_num_ = 0;
  double aver_time_consu_ = 0;
  double aver_time_icp_ = 0;
  double aver_time_map_inre_ = 0;
  bool colmap_output_en_ = false;
};
#endif