/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "voxel_map.h"

#include <glog/logging.h>

void CalcBodyCov(Eigen::Vector3d &pb, const float range_inc,
                 const float degree_inc, Eigen::Matrix3d &cov)
{
  if (pb[2] == 0)
    pb[2] = 0.0001;
  float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
  float range_var = range_inc * range_inc;
  Eigen::Matrix2d direction_var;
  direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0,
      pow(sin(DEG2RAD(degree_inc)), 2);
  Eigen::Vector3d direction(pb);
  direction.normalize();
  Eigen::Matrix3d direction_hat;
  direction_hat << 0, -direction(2), direction(1), direction(2), 0,
      -direction(0), -direction(1), direction(0), 0;
  Eigen::Vector3d base_vector1(1, 1,
                               -(direction(0) + direction(1)) / direction(2));
  base_vector1.normalize();
  Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
  base_vector2.normalize();
  Eigen::Matrix<double, 3, 2> N;
  N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1),
      base_vector1(2), base_vector2(2);
  Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
  cov = direction * range_var * direction.transpose() +
        A * direction_var * A.transpose();
}

// Compute the variance of the each point
void VarInit(M3D &extR, V3D &extT, PointCloudXYZIN &pl_cur, std::vector<pointWithVar> &pvs, double dept_err, double beam_err)
{
  int plsize = pl_cur.size();
  pvs.clear();
  pvs.resize(plsize);
  for (int i = 0; i < plsize; i++)
  {
    PointXYZIN &ap = pl_cur[i];
    pointWithVar &pv = pvs.at(i);
    pv.point_i << ap.x, ap.y, ap.z;
    CalcBodyCov(pv.point_i, dept_err, beam_err, pv.var);
    pv.point_i = extR * pv.point_i + extT;
    pv.var = extR * pv.var * extR.transpose();
  }
}

void Var2World(std::vector<pointWithVar> &pvs, StatesGroup &state)
{
  Eigen::Matrix3d rot_var = state.cov.block<3, 3>(0, 0);
  Eigen::Matrix3d tsl_var = state.cov.block<3, 3>(3, 3);

  for (pointWithVar &pv : pvs)
  {
    Eigen::Matrix3d phat;
    phat << SKEW_SYM_MATRX(pv.point_i);
    // pv.var = state.rot_end * pv.var * state.rot_end.transpose() + phat * rot_var * phat.transpose() + tsl_var;
    pv.var = state.rot_end * pv.var * state.rot_end.transpose() + state.rot_end * phat * rot_var * phat.transpose() * state.rot_end.transpose() + tsl_var;
    pv.point_w = state.rot_end * pv.point_i + state.pos_end;
    // pwld.push_back(pv.point_w);
  }
}

#ifdef ROS1
void loadVoxelConfig(ros::NodeHandle &nh, VoxelMapConfig &voxel_config) {
  nh.param<bool>("publish/pub_plane_en", voxel_config.is_pub_plane_map_, false);

  nh.param<int>("lio/max_layer", voxel_config.max_layer_, 1);
  nh.param<double>("lio/voxel_size", voxel_config.max_voxel_size_, 0.5);
  nh.param<double>("lio/min_eigen_value", voxel_config.planner_threshold_,
                   0.01);
  nh.param<double>("lio/sigma_num", voxel_config.sigma_num_, 3);
  nh.param<double>("lio/beam_err", voxel_config.beam_err_, 0.02);
  nh.param<double>("lio/dept_err", voxel_config.dept_err_, 0.05);
  nh.param<std::vector<int>>("lio/layer_init_num", voxel_config.layer_init_num_,
                             std::vector<int>{5, 5, 5, 5, 5});
  nh.param<int>("lio/max_points_num", voxel_config.max_points_num_, 50);
  nh.param<int>("lio/max_iterations", voxel_config.max_iterations_, 5);

  nh.param<bool>("local_map/map_sliding_en", voxel_config.map_sliding_en,
                 false);
  nh.param<int>("local_map/half_map_size", voxel_config.half_map_size, 100);
  nh.param<double>("local_map/sliding_thresh", voxel_config.sliding_thresh, 8);
}
#else
void loadVoxelConfig(rclcpp::Node::SharedPtr &node, VoxelMapConfig &voxel_config) {
  std::vector<int64_t> layer_init_num;
  node->declare_parameter("publish.pub_plane_en", false);
  node->declare_parameter("lio.max_layer", 1);
  node->declare_parameter("lio.voxel_size", 0.5);
  node->declare_parameter("lio.min_eigen_value", 0.01);
  node->declare_parameter("lio.sigma_num", 3.0);
  node->declare_parameter("lio.beam_err", 0.02);
  node->declare_parameter("lio.dept_err", 0.05);
  node->declare_parameter("lio.layer_init_num", std::vector<int64_t>({5, 5, 5, 5, 5}));
  node->declare_parameter("lio.max_points_num", 50);
  node->declare_parameter("lio.max_iterations", 5);
  node->declare_parameter("local_map.map_sliding_en", false);
  node->declare_parameter("local_map.half_map_size", 100);
  node->declare_parameter("local_map.sliding_thresh", 8.0);

  node->get_parameter("publish.pub_plane_en", voxel_config.is_pub_plane_map_);
  node->get_parameter("lio.max_layer", voxel_config.max_layer_);
  node->get_parameter("lio.voxel_size", voxel_config.max_voxel_size_);
  node->get_parameter("lio.min_eigen_value", voxel_config.planner_threshold_);
  node->get_parameter("lio.sigma_num", voxel_config.sigma_num_);
  node->get_parameter("lio.beam_err", voxel_config.beam_err_);
  node->get_parameter("lio.dept_err", voxel_config.dept_err_);
  node->get_parameter("lio.layer_init_num", layer_init_num);
  node->get_parameter("lio.max_points_num", voxel_config.max_points_num_);
  node->get_parameter("lio.max_iterations", voxel_config.max_iterations_);
  node->get_parameter("local_map.map_sliding_en", voxel_config.map_sliding_en);
  node->get_parameter("local_map.half_map_size", voxel_config.half_map_size);
  node->get_parameter("local_map.sliding_thresh", voxel_config.sliding_thresh);

  for (const auto &val : layer_init_num)
  {
    voxel_config.layer_init_num_.push_back(static_cast<int>(val));
  }
}
#endif

void VoxelOctoTree::InitPlane(const std::vector<pointWithVar> &points,
                              VoxelPlane *plane) {
  plane->plane_var_ = Eigen::Matrix<double, 6, 6>::Zero();
  plane->covariance_ = Eigen::Matrix3d::Zero();
  plane->center_ = Eigen::Vector3d::Zero();
  plane->normal_ = Eigen::Vector3d::Zero();
  plane->points_size_ = points.size();
  plane->radius_ = 0;
  for (auto pv : points) {
    plane->covariance_ += pv.point_w * pv.point_w.transpose();
    plane->center_ += pv.point_w;
  }
  plane->center_ = plane->center_ / plane->points_size_;
  plane->covariance_ = plane->covariance_ / plane->points_size_ -
                       plane->center_ * plane->center_.transpose();
  Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance_);
  Eigen::Matrix3cd evecs = es.eigenvectors();
  Eigen::Vector3cd evals = es.eigenvalues();
  Eigen::Vector3d evalsReal;
  evalsReal = evals.real();
  Eigen::Matrix3f::Index evalsMin, evalsMax;
  evalsReal.rowwise().sum().minCoeff(&evalsMin);
  evalsReal.rowwise().sum().maxCoeff(&evalsMax);
  int evalsMid = 3 - evalsMin - evalsMax;
  Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
  Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
  Eigen::Vector3d evecMax = evecs.real().col(evalsMax);
  Eigen::Matrix3d J_Q;
  J_Q << 1.0 / plane->points_size_, 0, 0, 0, 1.0 / plane->points_size_, 0, 0, 0,
      1.0 / plane->points_size_;
  // && evalsReal(evalsMid) > 0.05
  //&& evalsReal(evalsMid) > 0.01
  if (evalsReal(evalsMin) < planer_threshold_) {
    for (int i = 0; i < points.size(); i++) {
      Eigen::Matrix<double, 6, 3> J;
      Eigen::Matrix3d F;
      for (int m = 0; m < 3; m++) {
        if (m != (int)evalsMin) {
          Eigen::Matrix<double, 1, 3> F_m =
              (points[i].point_w - plane->center_).transpose() /
              ((plane->points_size_) * (evalsReal[evalsMin] - evalsReal[m])) *
              (evecs.real().col(m) * evecs.real().col(evalsMin).transpose() +
               evecs.real().col(evalsMin) * evecs.real().col(m).transpose());
          F.row(m) = F_m;
        } else {
          Eigen::Matrix<double, 1, 3> F_m;
          F_m << 0, 0, 0;
          F.row(m) = F_m;
        }
      }
      J.block<3, 3>(0, 0) = evecs.real() * F;
      J.block<3, 3>(3, 0) = J_Q;
      plane->plane_var_ += J * points[i].var * J.transpose();
    }

    plane->normal_ << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
        evecs.real()(2, evalsMin);
    plane->y_normal_ << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid),
        evecs.real()(2, evalsMid);
    plane->x_normal_ << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax),
        evecs.real()(2, evalsMax);
    plane->min_eigen_value_ = evalsReal(evalsMin);
    plane->mid_eigen_value_ = evalsReal(evalsMid);
    plane->max_eigen_value_ = evalsReal(evalsMax);
    plane->radius_ = sqrt(evalsReal(evalsMax));
    plane->d_ = -(plane->normal_(0) * plane->center_(0) +
                  plane->normal_(1) * plane->center_(1) +
                  plane->normal_(2) * plane->center_(2));
    plane->is_plane_ = true;
    plane->is_update_ = true;
    if (!plane->is_init_) {
      plane->id_ = voxel_plane_id;
      voxel_plane_id++;
      plane->is_init_ = true;
    }
  } else {
    plane->is_update_ = true;
    plane->is_plane_ = false;
  }
}

void VoxelOctoTree::InitOctoTree() {
  if (temp_points_.size() > points_size_threshold_) {
    InitPlane(temp_points_, plane_ptr_);
    if (plane_ptr_->is_plane_ == true) {
      octo_state_ = 0;
      // new added
      if (temp_points_.size() > max_points_num_) {
        update_enable_ = false;
        std::vector<pointWithVar>().swap(temp_points_);
        new_points_ = 0;
      }
    } else {
      octo_state_ = 1;
      CutOctoTree();
    }
    init_octo_ = true;
    new_points_ = 0;
  }
}

void VoxelOctoTree::CutOctoTree() {
  if (layer_ >= max_layer_) {
    octo_state_ = 0;
    return;
  }
  for (size_t i = 0; i < temp_points_.size(); i++) {
    int xyz[3] = {0, 0, 0};
    if (temp_points_[i].point_w[0] > voxel_center_[0]) {
      xyz[0] = 1;
    }
    if (temp_points_[i].point_w[1] > voxel_center_[1]) {
      xyz[1] = 1;
    }
    if (temp_points_[i].point_w[2] > voxel_center_[2]) {
      xyz[2] = 1;
    }
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves_[leafnum] == nullptr) {
      leaves_[leafnum] =
          new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1],
                            max_points_num_, planer_threshold_);
      leaves_[leafnum]->layer_init_num_ = layer_init_num_;
      leaves_[leafnum]->voxel_center_[0] =
          voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[1] =
          voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[2] =
          voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
      leaves_[leafnum]->quater_length_ = quater_length_ / 2;
    }
    leaves_[leafnum]->temp_points_.push_back(temp_points_[i]);
    leaves_[leafnum]->new_points_++;
  }
  for (uint i = 0; i < 8; i++) {
    if (leaves_[i] != nullptr) {
      if (leaves_[i]->temp_points_.size() >
          leaves_[i]->points_size_threshold_) {
        InitPlane(leaves_[i]->temp_points_, leaves_[i]->plane_ptr_);
        if (leaves_[i]->plane_ptr_->is_plane_) {
          leaves_[i]->octo_state_ = 0;
          // new added
          if (leaves_[i]->temp_points_.size() > leaves_[i]->max_points_num_) {
            leaves_[i]->update_enable_ = false;
            std::vector<pointWithVar>().swap(leaves_[i]->temp_points_);
            new_points_ = 0;
          }
        } else {
          leaves_[i]->octo_state_ = 1;
          leaves_[i]->CutOctoTree();
        }
        leaves_[i]->init_octo_ = true;
        leaves_[i]->new_points_ = 0;
      }
    }
  }
}

void VoxelOctoTree::UpdateOctoTree(const pointWithVar &pv) {
  if (!init_octo_) {
    new_points_++;
    temp_points_.push_back(pv);
    if (temp_points_.size() > points_size_threshold_) {
      InitOctoTree();
    }
  } else {
    if (plane_ptr_->is_plane_) {
      if (update_enable_) {
        new_points_++;
        temp_points_.push_back(pv);
        if (new_points_ > update_size_threshold_) {
          InitPlane(temp_points_, plane_ptr_);
          new_points_ = 0;
        }
        if (temp_points_.size() >= max_points_num_) {
          update_enable_ = false;
          std::vector<pointWithVar>().swap(temp_points_);
          new_points_ = 0;
        }
      }
    } else {
      if (layer_ < max_layer_) {
        int xyz[3] = {0, 0, 0};
        if (pv.point_w[0] > voxel_center_[0]) {
          xyz[0] = 1;
        }
        if (pv.point_w[1] > voxel_center_[1]) {
          xyz[1] = 1;
        }
        if (pv.point_w[2] > voxel_center_[2]) {
          xyz[2] = 1;
        }
        int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
        if (leaves_[leafnum] != nullptr) {
          leaves_[leafnum]->UpdateOctoTree(pv);
        } else {
          leaves_[leafnum] = new VoxelOctoTree(
              max_layer_, layer_ + 1, layer_init_num_[layer_ + 1],
              max_points_num_, planer_threshold_);
          leaves_[leafnum]->layer_init_num_ = layer_init_num_;
          leaves_[leafnum]->voxel_center_[0] =
              voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
          leaves_[leafnum]->voxel_center_[1] =
              voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
          leaves_[leafnum]->voxel_center_[2] =
              voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
          leaves_[leafnum]->quater_length_ = quater_length_ / 2;
          leaves_[leafnum]->UpdateOctoTree(pv);
        }
      } else {
        if (update_enable_) {
          new_points_++;
          temp_points_.push_back(pv);
          if (new_points_ > update_size_threshold_) {
            InitPlane(temp_points_, plane_ptr_);
            new_points_ = 0;
          }
          if (temp_points_.size() > max_points_num_) {
            update_enable_ = false;
            std::vector<pointWithVar>().swap(temp_points_);
            new_points_ = 0;
          }
        }
      }
    }
  }
}

VoxelOctoTree *VoxelOctoTree::FindCorrespond(Eigen::Vector3d pw) {
  if (!init_octo_ || plane_ptr_->is_plane_ || (layer_ >= max_layer_))
    return this;

  int xyz[3] = {0, 0, 0};
  xyz[0] = pw[0] > voxel_center_[0] ? 1 : 0;
  xyz[1] = pw[1] > voxel_center_[1] ? 1 : 0;
  xyz[2] = pw[2] > voxel_center_[2] ? 1 : 0;
  int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];

  // printf("leafnum: %d. \n", leafnum);

  return (leaves_[leafnum] != nullptr) ? leaves_[leafnum]->FindCorrespond(pw)
                                       : this;
}

bool VoxelOctoTree::inside(Eigen::Vector3d &wld)
{
  double hl = quater_length_ * 2;
  return (wld[0] >= voxel_center_[0] - hl &&
          wld[0] <= voxel_center_[0] + hl &&
          wld[1] >= voxel_center_[1] - hl &&
          wld[1] <= voxel_center_[1] + hl &&
          wld[2] >= voxel_center_[2] - hl &&
          wld[2] <= voxel_center_[2] + hl);
}

// EKF matching: find plane at world point, compute point-to-plane distance with probabilistic gating (3-sigma).
int VoxelOctoTree::Match(Eigen::Vector3d &wld, VoxelPlane *&pla, double &max_prob, Eigen::Matrix3d &var_wld, double &sigma_d, VoxelOctoTree *&oc)
{
  int flag = 0;
  if (octo_state_ == 0)
  {
    if (plane_ptr_->is_plane_)
    {
      float dis_to_plane = fabs(plane_ptr_->normal_.dot(wld - plane_ptr_->center_));
      float dis_to_center = (plane_ptr_->center_ - wld).squaredNorm();
      float range_dis = (dis_to_center - dis_to_plane * dis_to_plane);
      if (range_dis <= 3 * 3 * plane_ptr_->radius_)
      {
        Eigen::Matrix<double, 1, 6> J_nq;
        J_nq.block<1, 3>(0, 0) = wld - plane_ptr_->center_;
        J_nq.block<1, 3>(0, 3) = -plane_ptr_->normal_;
        double sigma_l = J_nq * plane_ptr_->plane_var_ * J_nq.transpose();
        sigma_l += plane_ptr_->normal_.transpose() * var_wld * plane_ptr_->normal_;
        if (dis_to_plane < 3 * sqrt(sigma_l))
        {
          float prob = 1 / (sqrt(sigma_l)) * exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
          if(prob > max_prob)
          {
            oc = this;
            sigma_d = sigma_l;
            max_prob = prob;
            pla = plane_ptr_;
          }

          flag = 1;
        }
      }
    }
  }
  else
  {
    int xyz[3] = {0, 0, 0};
    for (int k = 0; k < 3; k++)
      if (wld[k] > voxel_center_[k])
        xyz[k] = 1;
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];

    if (leaves_[leafnum] != nullptr)
      flag = leaves_[leafnum]->Match(wld, pla, max_prob, var_wld, sigma_d, oc);
  }

  return flag;
}

VoxelMapManager::VoxelMapManager(VoxelMapConfig &config_setting)
    : config_setting_(config_setting) {
};

// Match the point with the plane in the voxel map
int VoxelMapManager::Match(Eigen::Vector3d &wld, VoxelPlane *&plane, Eigen::Matrix3d &var_wld, double &sigma_d, VoxelOctoTree *&oc)
{
  int flag = 0;
  float loc[3];
  double voxel_size = config_setting_.max_voxel_size_;
  for (int j = 0; j < 3; j++)
  {
    loc[j] = wld[j] / voxel_size;
    if (loc[j] < 0)
      loc[j] -= 1;
  }
  VOXEL_LOCATION position(loc[0], loc[1], loc[2]);
  auto iter = vm_map_.find(position);
  if (iter != vm_map_.end())
  {
    double max_prob = 0;
    flag = iter->second->second->Match(wld, plane, max_prob, var_wld, sigma_d, oc);
    if (flag && plane == nullptr)
    {
      printf("plane null max_prob: %lf %ld %ld %ld\n", max_prob, position.x, position.y, position.z);
    }
  }

  return flag;
}

bool VoxelMapManager::StateEstimation(StatesGroup &state_propagat, const PointCloudXYZIN::Ptr &cloud_body)
{
  int feats_down_size = cloud_body->points.size();
  std::vector<pointWithVar>().swap(pv_list_);
  pv_list_.resize(feats_down_size);
  VarInit(extR_, extT_, *cloud_body, pv_list_, config_setting_.dept_err_, config_setting_.beam_err_);

  int rematch_num = 0;
  Mat19d G, H_T_H, I_STATE;
  G.setZero();
  H_T_H.setZero();
  I_STATE.setIdentity();

  bool flg_EKF_inited, flg_EKF_converged, EKF_stop_flg = 0;
  std::vector<VoxelOctoTree *> octos;
  octos.resize(feats_down_size, nullptr);

  Eigen::Matrix3d nnt;
  for (int iterCount = 0; iterCount < config_setting_.max_iterations_; iterCount++)
  {
    Eigen::Matrix<double, 6, 6> HTH;
    HTH.setZero();
    Eigen::Matrix<double, 6, 1> HTz;
    HTz.setZero();
    Eigen::Matrix3d rot_var = state_.cov.block<3, 3>(0, 0);
    Eigen::Matrix3d tsl_var = state_.cov.block<3, 3>(3, 3);
    nnt.setZero();

    for (int i = 0; i < feats_down_size; i++)
    {
      pointWithVar &pv = pv_list_.at(i);
      Eigen::Matrix3d phat;
      phat << SKEW_SYM_MATRX(pv.point_i);
      // Eigen::Matrix3d var_world = state_.rot_end * pv.var * state_.rot_end.transpose() + phat * rot_var * phat.transpose() + tsl_var;
      Eigen::Matrix3d var_world = state_.rot_end * pv.var * state_.rot_end.transpose() + state_.rot_end * phat * rot_var * phat.transpose() * state_.rot_end.transpose() + tsl_var;
      Eigen::Vector3d wld = state_.rot_end * pv.point_i + state_.pos_end;

      double sigma_d = 0;
      VoxelPlane *pla = nullptr;
      int flag = 0;

      if (octos[i] != nullptr && octos[i]->inside(wld))
      {
        double max_prob = 0;
        flag = octos[i]->Match(wld, pla, max_prob, var_world, sigma_d, octos[i]);
      }
      else
      {
        flag = Match(wld, pla, var_world, sigma_d, octos[i]);
      }

      if (flag)
      {
        VoxelPlane &pp = *pla;
        double R_inv = 1.0 / (0.0005 + sigma_d);
        double resi = pp.normal_.dot(wld - pp.center_);

        Eigen::Matrix<double, 6, 1> jac;
        jac.head(3) = phat * state_.rot_end.transpose() * pp.normal_;
        jac.tail(3) = pp.normal_;
        HTH += R_inv * jac * jac.transpose();
        HTz -= R_inv * jac * resi;
        nnt += pp.normal_ * pp.normal_.transpose();
      }
    }

    H_T_H.block<6, 6>(0, 0) = HTH;
    Mat19d &&K_1 = (H_T_H + state_.cov.inverse()).inverse();
    G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
    auto vec = state_propagat - state_;
    Vec19d solution = K_1.block<DIM_STATE, 6>(0, 0) * HTz +
                      vec.block<DIM_STATE, 1>(0, 0) -
                      G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);

    state_ += solution;
    auto rot_add = solution.block<3, 1>(0, 0);
    auto t_add = solution.block<3, 1>(3, 0);

    EKF_stop_flg = false;
    flg_EKF_converged = false;

    if ((rot_add.norm() * 57.3 < 0.01) && (t_add.norm() * 100 < 0.015))
    {
      flg_EKF_converged = true;
    }

    /*** Rematch Judgement ***/
    if (flg_EKF_converged || ((rematch_num == 0) && (iterCount == (config_setting_.max_iterations_ - 2))))
    {
      rematch_num++;
    }

    /*** Convergence Judgements and Covariance Update ***/
    if (!EKF_stop_flg && (rematch_num >= 2 ||
                          (iterCount == config_setting_.max_iterations_ - 1)))
    {
      /*** Covariance Update ***/
      // P_k+1 = (I - KH)P_k
      state_.cov = (I_STATE - G) * state_.cov;
      EKF_stop_flg = true;
    }

    if (EKF_stop_flg)
      break;
  }

  // Degeneration detection: eigenvalue threshold = 14
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(nnt);
  Eigen::Vector3d evalue = saes.eigenvalues();
  // printf("eva %d: %lf\n", match_num, evalue[0]);

  if (evalue[0] >= degrade_eigval_)
  {
    if (degrade_cnt_ > 0)
      degrade_cnt_--;
  }
  else
    degrade_cnt_++;

  if (degrade_cnt_ > degrade_bound_)
    return false;
  else
    return true;
}

#if 1
void VoxelMapManager::UpdateVoxelMapLRU(std::vector<pointWithVar> &input_points)
{
  float voxel_size = config_setting_.max_voxel_size_;
  float planer_threshold = config_setting_.planner_threshold_;
  int max_layer = config_setting_.max_layer_;
  int max_points_num = config_setting_.max_points_num_;
  std::vector<int> layer_init_num = config_setting_.layer_init_num_;

  Var2World(pv_list_, state_);

  uint plsize = input_points.size();
  for (uint i = 0; i < plsize; i++) {
    const pointWithVar p_v = input_points[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_v.point_w[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                            (int64_t)loc_xyz[2]);
    auto iter = vm_map_.find(position);
    if (iter != vm_map_.end()) {
      vm_map_[position]->second->UpdateOctoTree(p_v);
      // 更新的放至最前
      vm_data_.splice(vm_data_.begin(), vm_data_, iter->second);
      iter->second = vm_data_.begin();
    } else {
      VoxelOctoTree *octo_tree = new VoxelOctoTree(
          max_layer, 0, layer_init_num[0], max_points_num, planer_threshold);
      vm_data_.push_front({position, {octo_tree}});
      vm_map_.insert({position, vm_data_.begin()});

      // LRU
      if (vm_data_.size() >= lru_size_) {
        // 删除一个尾部的数据
        vm_map_.erase(vm_data_.back().first);
        delete vm_data_.back().second;
        vm_data_.pop_back();
      }

      vm_map_[position]->second->quater_length_ = voxel_size / 4;
      vm_map_[position]->second->voxel_center_[0] =
          (0.5 + position.x) * voxel_size;
      vm_map_[position]->second->voxel_center_[1] =
          (0.5 + position.y) * voxel_size;
      vm_map_[position]->second->voxel_center_[2] =
          (0.5 + position.z) * voxel_size;
      vm_map_[position]->second->layer_init_num_ = layer_init_num;
      vm_map_[position]->second->UpdateOctoTree(p_v);
    }
  }
}

void VoxelMapManager::BuildVoxelMapLRU(const PointCloudXYZIN::Ptr &cloud_body)
{
  float voxel_size = config_setting_.max_voxel_size_;
  float planer_threshold = config_setting_.planner_threshold_;
  int max_layer = config_setting_.max_layer_;
  int max_points_num = config_setting_.max_points_num_;
  std::vector<int> layer_init_num = config_setting_.layer_init_num_;

  std::vector<pointWithVar> input_points;
  VarInit(extR_, extT_, *cloud_body, input_points, config_setting_.dept_err_, config_setting_.beam_err_);
  Var2World(input_points, state_);

  uint plsize = input_points.size();
  for (uint i = 0; i < plsize; i++) {
    const pointWithVar p_v = input_points[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_v.point_w[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                            (int64_t)loc_xyz[2]);
    auto iter = vm_map_.find(position);
    if (iter != vm_map_.end()) {
      // 体素已存在
      vm_map_[position]->second->temp_points_.push_back(p_v);
      vm_map_[position]->second->new_points_++;
      // 更新的放至最前
      vm_data_.splice(vm_data_.begin(), vm_data_, iter->second);
      iter->second = vm_data_.begin();
    } else {
      // 体素不存在
      VoxelOctoTree *octo_tree = new VoxelOctoTree(
          max_layer, 0, layer_init_num[0], max_points_num, planer_threshold);
      vm_data_.push_front({position, {octo_tree}});
      vm_map_.insert({position, vm_data_.begin()});
      vm_map_[position]->second->quater_length_ = voxel_size / 4;
      vm_map_[position]->second->voxel_center_[0] =
          (0.5 + position.x) * voxel_size;
      vm_map_[position]->second->voxel_center_[1] =
          (0.5 + position.y) * voxel_size;
      vm_map_[position]->second->voxel_center_[2] =
          (0.5 + position.z) * voxel_size;
      vm_map_[position]->second->temp_points_.push_back(p_v);
      vm_map_[position]->second->new_points_++;
      vm_map_[position]->second->layer_init_num_ = layer_init_num;

      // LRU
      if (vm_data_.size() >= lru_size_) {
        // 删除一个尾部的数据
        vm_map_.erase(vm_data_.back().first);
        delete vm_data_.back().second;
        vm_data_.pop_back();
      }
    }
  }

  for (auto iter = vm_map_.begin(); iter != vm_map_.end(); ++iter) {
    iter->second->second->InitOctoTree();
  }
}
#endif

void VoxelMapManager::RebuildVoxelMapLRU(const PointCloudXYZIN::Ptr &cloud_body)
{
  // reset
  for (auto &pair : vm_map_)
    delete pair.second->second;
  vm_map_.clear();
  vm_data_.clear();

  BuildVoxelMapLRU(cloud_body);
}
