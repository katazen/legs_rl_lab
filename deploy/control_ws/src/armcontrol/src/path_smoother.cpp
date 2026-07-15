#include "armcontrol/path_smoother.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

QuinticPolynomial::QuinticPolynomial(double x0, double v0, double a0_acc,
                                     double x1, double v1, double a1_acc,
                                     double T) {
  // 边界条件：
  // x(0) = x0, x'(0) = v0, x''(0) = a0_acc
  // x(T) = x1, x'(T) = v1, x''(T) = a1_acc

  this->a0 = x0;
  this->a1 = v0;
  this->a2 = a0_acc / 2.0;

  // 求解剩余系数
  double T2 = T * T;
  double T3 = T2 * T;
  double T4 = T3 * T;
  double T5 = T4 * T;

  // 系数矩阵求解
  double h1 = x1 - x0 - v0 * T - 0.5 * a0_acc * T2;
  double h2 = v1 - v0 - a0_acc * T;
  double h3 = a1_acc - a0_acc;

  this->a3 = (20 * h1 - 8 * h2 * T - 3 * h3 * T2) / (2 * T3);
  this->a4 = (-30 * h1 + 14 * h2 * T + 3 * h3 * T2) / (2 * T4);
  this->a5 = (12 * h1 - 6 * h2 * T - h3 * T2) / (2 * T5);
}

// 计算位置
double QuinticPolynomial::position(double t) const {
  return a0 + a1 * t + a2 * t * t + a3 * t * t * t + a4 * t * t * t * t +
         a5 * t * t * t * t * t;
}

// 计算速度
double QuinticPolynomial::velocity(double t) const {
  return a1 + 2 * a2 * t + 3 * a3 * t * t + 4 * a4 * t * t * t +
         5 * a5 * t * t * t * t;
}

// 计算加速度
double QuinticPolynomial::acceleration(double t) const {
  return 2 * a2 + 6 * a3 * t + 12 * a4 * t * t + 20 * a5 * t * t * t;
}

// 添加路径点
void PathSmoother::addWaypoint(double position, double velocity,
                               double acceleration, double time) {
  waypoints.push_back(PathPoint(position, velocity, acceleration, time));
}

// 生成平滑路径
void PathSmoother::generateSmoothPath() {
  if (waypoints.size() < 2) {
    std::cerr << "至少需要2个路径点" << std::endl;
    return;
  }

  polynomials.clear();

  // 为每两个相邻路径点生成五次多项式
  for (size_t i = 0; i < waypoints.size() - 1; i++) {
    double deltaT = waypoints[i + 1].time - waypoints[i].time;
    if (deltaT <= 0) {
      deltaT = 1.0; // 默认时间间隔
    }

    QuinticPolynomial poly(waypoints[i].position, waypoints[i].velocity,
                           waypoints[i].acceleration, waypoints[i + 1].position,
                           waypoints[i + 1].velocity,
                           waypoints[i + 1].acceleration, deltaT);

    polynomials.push_back(poly);
  }
}

// 计算指定时间的位置、速度、加速度
PathPoint PathSmoother::interpolate(double t) const {
  if (polynomials.empty()) {
    std::cerr << "请先生成平滑路径" << std::endl;
    return PathPoint();
  }

  // 找到对应的时间段
  size_t segment = 0;
  double localTime = t;

  for (size_t i = 0; i < waypoints.size() - 1; i++) {
    double segmentDuration = waypoints[i + 1].time - waypoints[i].time;
    if (segmentDuration <= 0)
      segmentDuration = 1.0;

    if (localTime <= segmentDuration) {
      segment = i;
      break;
    }
    localTime -= segmentDuration;
  }

  if (segment >= polynomials.size()) {
    segment = polynomials.size() - 1;
    localTime = waypoints.back().time - waypoints[waypoints.size() - 2].time;
  }

  const QuinticPolynomial &poly = polynomials[segment];
  return PathPoint(poly.position(localTime), poly.velocity(localTime),
                   poly.acceleration(localTime), t);
}

double PathSmoother::calculateSCurveTime(double distance, double max_vel,
                                         double max_acc, double max_j) {
  distance = std::abs(distance);
  if (distance <= 0.01) {
    return 0.1;
  }
  // 加加速度阶段时间
  double t_jerk = max_acc / max_j;
  // 加速度达到最大值时的速度
  double v_jerk = 0.5 * max_j * t_jerk * t_jerk;
  // 如果最大速度太小，无法达到最大加速度
  if (max_vel <= 2.0 * v_jerk) {
    // 重新计算加加速度时间
    t_jerk = std::sqrt(max_vel / max_j);
    double total_time = 4.0 * t_jerk;
    double s_total = 2.0 * max_vel * t_jerk;
    if (distance <= s_total) {
      // 进一步缩短时间
      double scale = std::sqrt(distance / s_total);
      return total_time * scale;
    }
    return total_time;
  }
  // 标准七段式S曲线计算
  // 第一阶段：加加速度 (0 -> max_jerk)
  double t1 = t_jerk;
  double s1 = max_j * t1 * t1 * t1 / 6.0;
  double v1 = 0.5 * max_j * t1 * t1;
  // 第三阶段：减加速度 (max_jerk -> 0)
  double t3 = t_jerk;
  double s3 = v1 * t3 + 0.5 * max_acc * t3 * t3 - max_j * t3 * t3 * t3 / 6.0;
  double v3 = v1 + max_acc * t3 - 0.5 * max_j * t3 * t3;
  // 检查是否能达到最大速度
  double v_at_end_of_accel = v3;
  if (v_at_end_of_accel >= max_vel) {
    // 无法达到最大速度，需要重新计算
    // 使用二次方程求解
    double a = max_j;
    double b = 2.0 * max_acc;
    double c = -2.0 * max_vel;
    double discriminant = b * b - 4.0 * a * c;
    if (discriminant > 0) {
      t1 = (-b + std::sqrt(discriminant)) / (2.0 * a);
      t3 = t1;
      s1 = max_j * t1 * t1 * t1 / 6.0;
      v1 = 0.5 * max_j * t1 * t1;
      s3 = v1 * t3 + 0.5 * max_acc * t3 * t3 - max_j * t3 * t3 * t3 / 6.0;
      double t2 = (max_vel - v1) / max_acc - t3;
      t2 = std::max(0.0, t2);
      double s2 = v1 * t2 + 0.5 * max_acc * t2 * t2;
      // 加速段总距离
      double s_accel = s1 + s2 + s3;
      // 减速段距离相同
      double s_decel = s_accel;
      if (distance <= s_accel + s_decel) {
        // 无法达到最大速度，使用缩放
        double scale = std::sqrt(distance / (s_accel + s_decel));
        return 2.0 * (t1 + t2 + t3) * scale;
      } else {
        // 匀速段时间
        double t_uniform = (distance - s_accel - s_decel) / max_vel;
        return 2.0 * (t1 + t2 + t3) + t_uniform;
      }
    }
  }
  // 第二阶段：匀加速 (保持 max_acc)
  double t2 = (max_vel - v3) / max_acc;
  t2 = std::max(0.0, t2);
  double s2 = v3 * t2 + 0.5 * max_acc * t2 * t2;
  // 加速段总距离和时间
  double s_accel = s1 + s2 + s3;
  double t_accel = t1 + t2 + t3;
  // 减速段距离和时间相同
  double s_decel = s_accel;
  double t_decel = t_accel;
  if (distance <= s_accel + s_decel) {
    // 无法达到最大速度
    double tmp = distance / (s_accel + s_decel);
    if (tmp < 0.001) {
      tmp = 0.001;
    }
    double scale = std::sqrt(tmp);
    return (t_accel + t_decel) * scale;
  } else {
    // 可以达到最大速度，计算匀速段时间
    double s_uniform = distance - s_accel - s_decel;
    double t_uniform = s_uniform / max_vel;
    return t_accel + t_uniform + t_decel;
  }
}

// 清空路径点
void PathSmoother::clear() {
  waypoints.clear();
  polynomials.clear();
}
