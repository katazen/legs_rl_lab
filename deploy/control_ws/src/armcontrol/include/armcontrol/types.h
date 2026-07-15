#pragma once

#include <Eigen/Geometry>
#include <Eigen/LU>
// #include <hpp/fcl/collision_object.h>
#include <iostream>

namespace manual_ctrl {

using JointVector6f = Eigen::Matrix<float, 6, 1>;
using JointVector7f = Eigen::Matrix<float, 7, 1>;
using RotMatrix = Eigen::Matrix3d;
using Point = Eigen::Vector3d;
using Euler = Eigen::Vector3d;
using Quaternion = Eigen::Quaterniond;

inline Quaternion euler2Quaternion(const Euler &euler) {
  Eigen::AngleAxisd rollAngle(
      Eigen::AngleAxisd(euler(2), Eigen::Vector3d::UnitX()));
  Eigen::AngleAxisd pitchAngle(
      Eigen::AngleAxisd(euler(1), Eigen::Vector3d::UnitY()));
  Eigen::AngleAxisd yawAngle(
      Eigen::AngleAxisd(euler(0), Eigen::Vector3d::UnitZ()));
  Quaternion quaternion = yawAngle * pitchAngle * rollAngle;
  return quaternion;
}
enum Hand { LEFT_HAND, RIGHT_HAND };
struct Pose {
public:
  Point position;
  Quaternion orientation;

  Pose() : position(Point::Zero()), orientation(RotMatrix::Identity()) {}
  Pose(const Point &position_, const Quaternion &orientation_)
      : position(position_), orientation(orientation_) {}
  Pose(const Point &position_, const Euler &euler_) : position(position_) {
    Eigen::AngleAxisd rollAngle(
        Eigen::AngleAxisd(euler_(2), Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd pitchAngle(
        Eigen::AngleAxisd(euler_(1), Eigen::Vector3d::UnitY()));
    Eigen::AngleAxisd yawAngle(
        Eigen::AngleAxisd(euler_(0), Eigen::Vector3d::UnitZ()));
    orientation = yawAngle * pitchAngle * rollAngle;
  }
  Pose(const Pose &pose_)
      : position(pose_.position), orientation(pose_.orientation) {}
  const Pose operator=(const Pose &ref) {
    position = ref.position;
    orientation = ref.orientation;
    return *this;
  }

  friend std::ostream &operator<<(std::ostream &os, const Pose &obj) {
    os << obj.position.transpose() << "; "
       << obj.orientation.matrix().eulerAngles(2, 1, 0).transpose();
    return os;
  }
};

using Translation = Point;
struct Transform {
public:
  Translation translation;
  Quaternion rotation;
  Euler euler;

  Transform()
      : translation(Translation::Zero()), rotation(RotMatrix::Identity()),
        euler(Euler::Zero()) {}

  Transform(const Translation &translation_, const Quaternion &rotation_)
      : translation(translation_), rotation(rotation_) {
    euler = rotation.matrix().eulerAngles(2, 1, 0);
  }

  Transform(const Translation &translation_, const RotMatrix &rotation_)
      : translation(translation_), rotation(rotation_) {
    euler = rotation.matrix().eulerAngles(2, 1, 0);
  }
  Transform(const Translation &translation_, const Euler &euler_)
      : translation(translation_), euler(euler_) {
    rotation = euler2Quaternion(euler_);
  }
  Transform(const Transform &transform_)
      : translation(transform_.translation), rotation(transform_.rotation),
        euler(transform_.euler) {}
  const Transform operator=(const Transform &ref) {
    translation = ref.translation;
    rotation = ref.rotation;
    euler = ref.euler;
    return *this;
  }

  friend std::ostream &operator<<(std::ostream &os, const Transform &obj) {
    os << obj.translation.transpose() << "; " << obj.rotation << "; "
       << obj.euler.transpose();
    return os;
  }
};

  // 碰撞检测结果结构体，jingyi
 struct CollisionResult {
    bool has_collision = false;
    bool self_collision_left = false;
    bool env_collision_left = false;
    bool self_collision_right = false;
    bool env_collision_right = false;
    bool arm_body_collision_left = false;
    bool arm_body_collision_right = false;
    bool arm_to_arm_collision = false; // 双臂之间的碰撞
    // CollisionInfo env_collision_info;
    uint64_t timestamp = 0;
    
    // 便于判断
    operator bool() const { return has_collision; }
};

} // namespace manual_ctrl