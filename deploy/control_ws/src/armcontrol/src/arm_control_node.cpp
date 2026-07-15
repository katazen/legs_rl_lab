#ifdef USE_CAN_COMMUNICATION
#include "armcontrol/damiao_can.h"  // 使用CAN通信的达妙电机库头文件
#endif

#ifdef USE_SERIAL_COMMUNICATION
#include "armcontrol/damiao.h"
#endif

#include "armcontrol/path_smoother.h"
#include "armcontrol/serial_port.h"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/int32.hpp"
#include "std_msgs/msg/u_int8.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/bool.hpp"
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <chrono>
#include <cstring>
#include "sensor_msgs/msg/joint_state.hpp"
#include <nlohmann/json.hpp>
#include <future>
#include <queue>
#include <thread>
#include <mutex>
#include <random>
#include <condition_variable>
#include <atomic>
#include <cmath>
#include <sstream>
#include <fstream>
#include "std_msgs/msg/bool.hpp"//jingyi
#include <visualization_msgs/msg/marker_array.hpp>//gripper test,jingyi
#include <tf2_ros/transform_broadcaster.h> // TF广播器,jingyi
#include <tf2_ros/static_transform_broadcaster.h>  // 静态TF广播器,jingyi
#include <geometry_msgs/msg/transform_stamped.hpp>  // 变换消息,jingyi
#include <tf2/LinearMath/Quaternion.h>              // 四元数计算,jingyi
#include <tf2/LinearMath/Transform.h>               // Transform类,jingyi
#include <tf2/convert.h>                            // tf2转换函数,jingyi
#include <ctime>                                    // 诊断日志运行时间戳
#include <cstdlib>                                  // std::getenv (诊断目录用 $HOME)
#include <Eigen/Dense>                              // Eigen (原经 robot_model.h 间接引入, 现直接包含)

namespace {
// 诊断 CSV 落盘目录（$HOME 下，便于跨机迁移；需提前存在）
inline const std::string & diag_dir() {
  static const std::string d = []{
    const char * h = std::getenv("HOME");
    return std::string(h ? h : ".") + "/dual_leg_diag/";
  }();
  return d;
}
// 进程级运行时间戳：整次运行只计算一次，保证 cmd/state 两个文件成对
inline const std::string & diag_run_tag() {
  static const std::string tag = [] {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&t));
    return std::string(buf);
  }();
  return tag;
}
}  // namespace

#define USE_POS_VEL_MODE 0 // 使用位置+速度控制模式
#define ONLY_PINOCCHIO_IK 1 // 仅使用Pinocchio逆解
#define IGNORE_COLLISION 0 // 忽略碰撞检测
#define DTOF 12
#define TIME_STEP 0.01
// #define MAX_VEL 21.0 // 最大速度
#define MAX_VEL 1.5 // 最大速度 song0327
#define MAX_ACC (MAX_VEL / 5.0)
#define MAX_JERK (MAX_ACC / 3.0) // 最大加加速度

// #define PADDING_LEFT 0.0f // 夹爪整体的预留间隙,jingyi
// #define PADDING_RIGHT 0.0f // 夹爪整体的预留间隙,jingyi

// 机械臂状态
enum class ArmState {
  IDLE,    // 空闲状态
  RUNNING, // 运行状态（遥操模式）
  AGING,   // 老化状态
  ERROR    // 错误状态（无法操作）
};

// using namespace std::chrono_literals;
// 从臂
class ArmControlNode : public rclcpp::Node {
public:
  explicit ArmControlNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("armcontrol_node", options) {
    current_state_ = ArmState::IDLE;
    start_ = false;
    restore_ = false;
    // 初始化最大力矩数组
    left_max_tau_.resize(DTOF, 0.0);
    right_max_tau_.resize(DTOF, 0.0);

    // 初始化线程池
    size_t thread_count = std::thread::hardware_concurrency(); // 获取系统支持的线程数
    if (thread_count == 0) {
        thread_count = thread_pool_size_; // 如果无法获取，则使用默认值
    }
    else if (thread_count > thread_pool_size_) {
        thread_count = thread_pool_size_; // 限制线程数不超过预设值
    }
    for (size_t i = 0; i < thread_count; ++i) {
        thread_pool_.emplace_back([this]() {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    condition_.wait(lock, [this]() { return stop_threads_ || !task_queue_.empty(); });
                    if (stop_threads_ && task_queue_.empty()) {
                        return;
                    }
                    task = std::move(task_queue_.front());
                    task_queue_.pop();
                }
                task();
            }
        });
    }

    auto left_arm_device =
        this->declare_parameter<std::string>("left_arm_device", "/dev/ttyCANL");
    auto right_arm_device = this->declare_parameter<std::string>(
        "right_arm_device", "/dev/ttyCANR");
    auto joystick_topic = this->declare_parameter<std::string>(
        "joystick_topic", "/joystick_info");

    auto left_arm_can_name = this->declare_parameter<std::string>(
        "left_arm_can_name", "can2");
    auto right_arm_can_name = this->declare_parameter<std::string>(
        "right_arm_can_name", "can1");
    dual_leg_ = this->declare_parameter<bool>("dual_leg", true);

    config_max_vel_ = this->declare_parameter<double>("max_vel", MAX_VEL);
    config_max_acc_ = this->declare_parameter<double>("max_acc", MAX_ACC);
    config_max_jerk_ = this->declare_parameter<double>("max_jerk", MAX_JERK);
    config_time_step_ = this->declare_parameter<double>("time_step", TIME_STEP);
    
    auto repeat_to_size = [](const std::vector<double> &base, size_t size) {
      std::vector<double> out;
      out.reserve(size);
      while (out.size() < size) {
        size_t remaining = size - out.size();
        size_t copy_count = std::min(remaining, base.size());
        out.insert(out.end(), base.begin(), base.begin() + static_cast<long>(copy_count));
      }
      return out;
    };

    if(DTOF == 7)
    {
      config_kps_ = this->declare_parameter<std::vector<double>>(
          "kps", {200.0, 200.0, 150.0, 100.0, 20.0, 20.0, 20.0});
      config_kds_ = this->declare_parameter<std::vector<double>>(
          "kds", {8.0, 8.0, 5.0, 2.0, 0.2, 0.2, 0.2});
      auto legacy_kis_7 = this->declare_parameter<std::vector<double>>(
          "kis", std::vector<double>{});
      if (!legacy_kis_7.empty()) {
        config_kds_ = legacy_kis_7;
        RCLCPP_WARN(this->get_logger(), "Parameter 'kis' is deprecated. Please use 'kds'.");
      }

      // lift_default_height = 1.25;
      lift_default_height = 0.87;

      config_default_joints_left_ = this->declare_parameter<std::vector<double>>(
          // "default_joints_left", {0.0, 0.0, 0.0, -0.5, 0.0, 1.3, 1.57});
          // "default_joints_left", {-1.75, 0.78, 1.57, -1.05, -1.57, -0.75, 1.57});
          "default_joints_left", {-3.02, 0.031, 0.97, 0.98, -0.0188, -1.26, 1.78});
      config_default_joints_right_ = this->declare_parameter<std::vector<double>>(
          // "default_joints_right", {0.0, 0.0, 0.0, 0.5, 0.0, -1.3, -1.57});
          // "default_joints_right", {1.75, -0.78, -1.57, 1.05, 1.57, 0.75, -1.57});
          "default_joints_right", {3.02, -0.031, -0.97, -0.98, 0.0188, 1.26, -1.78});
      config_default_joints_zero_ = this->declare_parameter<std::vector<double>>(
          "default_joints_zero", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

      // TODO:song1120，我感觉模型读入了urdf，这可以从urdf中获取，不用再重新写
      max_joints_left_ = this->declare_parameter<std::vector<double>>(
          "max_joints_left", {1.05, 3.58, 2.76, 1.946, 2.234, 2.094, 2.87});
      min_joints_left_ = this->declare_parameter<std::vector<double>>(
          "min_joints_left", {-3.14, -0.44, -2.76, -1.946, -2.234, -2.094, -2.87});
      max_joints_right_ = this->declare_parameter<std::vector<double>>(
          "max_joints_right", {3.14, 0.44, 2.76, 1.946, 2.234, 2.094, 2.87});
      min_joints_right_ = this->declare_parameter<std::vector<double>>(
          "min_joints_right", {-1.05, -3.58, -2.76, -1.946, -2.234, -2.094, -2.87});

      max_delta_left_ = this->declare_parameter<std::vector<double>>(
          "max_delta_left", {0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035}); // 角度为2°
      min_delta_left_ = this->declare_parameter<std::vector<double>>(
          "min_delta_left", {-0.035, -0.035, -0.035, -0.035, -0.035, -0.035, -0.035}); // 角度为2°
      max_delta_right_ = this->declare_parameter<std::vector<double>>(
          "max_delta_right", {0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035}); // 角度为2°
      min_delta_right_ = this->declare_parameter<std::vector<double>>(
          "min_delta_right", {-0.035, -0.035, -0.035, -0.035, -0.035, -0.035, -0.035}); // 角度为2°
    }
    else if(DTOF == 6)
    {
      config_kps_ = this->declare_parameter<std::vector<double>>(
          "kps", {200.0, 200.0, 100.0, 30.0, 50.0, 20.0});
      config_kds_ = this->declare_parameter<std::vector<double>>(
          "kds", {8.0, 8.0, 5.0, 2.0, 2.0, 0.2});
      auto legacy_kis_6 = this->declare_parameter<std::vector<double>>(
          "kis", std::vector<double>{});
      if (!legacy_kis_6.empty()) {
        config_kds_ = legacy_kis_6;
        RCLCPP_WARN(this->get_logger(), "Parameter 'kis' is deprecated. Please use 'kds'.");
      }

      config_default_joints_left_ = this->declare_parameter<std::vector<double>>(
          "default_joints_left", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
      config_default_joints_right_ = this->declare_parameter<std::vector<double>>(
          "default_joints_right", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
      config_default_joints_zero_ = this->declare_parameter<std::vector<double>>(
          "default_joints_zero", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
      max_joints_left_ = this->declare_parameter<std::vector<double>>(
          "max_joints_left", {1.05, 3.58, 2.76, 1.946, 2.234, 2.094});
      min_joints_left_ = this->declare_parameter<std::vector<double>>(
          "min_joints_left", {-3.14, -0.44, -2.76, -1.946, -2.234, -2.094});
      max_joints_right_ = this->declare_parameter<std::vector<double>>(
          "max_joints_right", {3.14, 0.44, 2.76, 1.946, 2.234, 2.094});
      min_joints_right_ = this->declare_parameter<std::vector<double>>(
          "min_joints_right", {-1.05, -3.58, -2.76, -1.946, -2.234, -2.094});

      max_delta_left_ = this->declare_parameter<std::vector<double>>(
          "max_delta_left", {0.035, 0.035, 0.035, 0.035, 0.035, 0.035}); // 角度为2°
      min_delta_left_ = this->declare_parameter<std::vector<double>>(
          "min_delta_left", {-0.035, -0.035, -0.035, -0.035, -0.035, -0.035}); // 角度为2°
      max_delta_right_ = this->declare_parameter<std::vector<double>>(
          "max_delta_right", {0.035, 0.035, 0.035, 0.035, 0.035, 0.035}); // 角度为2°
      min_delta_right_ = this->declare_parameter<std::vector<double>>(
          "min_delta_right", {-0.035, -0.035, -0.035, -0.035, -0.035, -0.035}); // 角度为2°
    }
      else if (DTOF == 12)
      {
        std::vector<double> base_kps = {8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0};
        std::vector<double> base_kds = {0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2};
        std::vector<double> base_max_left = {1.05, 3.58, 2.76, 1.946, 2.234, 2.094};
        std::vector<double> base_min_left = {-3.14, -0.44, -2.76, -1.946, -2.234, -2.094};
        std::vector<double> base_max_right = {3.14, 0.44, 2.76, 1.946, 2.234, 2.094};
        std::vector<double> base_min_right = {-1.05, -3.58, -2.76, -1.946, -2.234, -2.094};
        std::vector<double> base_max_delta = {0.035, 0.035, 0.035, 0.035, 0.035, 0.035};
        std::vector<double> base_min_delta = {-0.035, -0.035, -0.035, -0.035, -0.035, -0.035};

        config_kps_ = this->declare_parameter<std::vector<double>>(
          "kps", repeat_to_size(base_kps, DTOF));
        config_kds_ = this->declare_parameter<std::vector<double>>(
          "kds", repeat_to_size(base_kds, DTOF));
        auto legacy_kis_12 = this->declare_parameter<std::vector<double>>(
          "kis", std::vector<double>{});

          {
            std::ostringstream kps_ss;
            kps_ss << "[";
            for (size_t i = 0; i < config_kps_.size(); ++i) {
              kps_ss << config_kps_[i];
              if (i + 1 < config_kps_.size()) {
                kps_ss << ", ";
              }
            }
            kps_ss << "]";

            std::ostringstream kds_ss;
            kds_ss << "[";
            for (size_t i = 0; i < config_kds_.size(); ++i) {
              kds_ss << config_kds_[i];
              if (i + 1 < config_kds_.size()) {
                kds_ss << ", ";
              }
            }
            kds_ss << "]";

            RCLCPP_INFO(this->get_logger(), "base_kps=%s base_kds=%s", kps_ss.str().c_str(), kds_ss.str().c_str());
          }

        if (!legacy_kis_12.empty()) {
          config_kds_ = legacy_kis_12;
          RCLCPP_WARN(this->get_logger(), "Parameter 'kis' is deprecated. Please use 'kds'.");
        }

        config_default_joints_left_ = this->declare_parameter<std::vector<double>>(
          "default_joints_left", std::vector<double>(DTOF, 0.0));
        config_default_joints_right_ = this->declare_parameter<std::vector<double>>(
          "default_joints_right", std::vector<double>(DTOF, 0.0));
        config_default_joints_zero_ = this->declare_parameter<std::vector<double>>(
          "default_joints_zero", std::vector<double>(DTOF, 0.0));

        max_joints_left_ = this->declare_parameter<std::vector<double>>(
          "max_joints_left", repeat_to_size(base_max_left, DTOF));
        min_joints_left_ = this->declare_parameter<std::vector<double>>(
          "min_joints_left", repeat_to_size(base_min_left, DTOF));
        max_joints_right_ = this->declare_parameter<std::vector<double>>(
          "max_joints_right", repeat_to_size(base_max_right, DTOF));
        min_joints_right_ = this->declare_parameter<std::vector<double>>(
          "min_joints_right", repeat_to_size(base_min_right, DTOF));

        max_delta_left_ = this->declare_parameter<std::vector<double>>(
          "max_delta_left", repeat_to_size(base_max_delta, DTOF));
        min_delta_left_ = this->declare_parameter<std::vector<double>>(
          "min_delta_left", repeat_to_size(base_min_delta, DTOF));
        max_delta_right_ = this->declare_parameter<std::vector<double>>(
          "max_delta_right", repeat_to_size(base_max_delta, DTOF));
        min_delta_right_ = this->declare_parameter<std::vector<double>>(
          "min_delta_right", repeat_to_size(base_min_delta, DTOF));
      }
    control_position_left_.resize(DTOF);
    control_position_left_[0] = -1000;
    control_position_right_.resize(DTOF);
    control_position_right_[0] = -1000;

    // 1110 byf 电机故障码单话题
    motors_err_pub_ = this->create_publisher<std_msgs::msg::String>("/motor_warn", 10);
    last_left_err_.fill(-1);
    last_right_err_.fill(-1);

    // 发布机械臂动作完成信号的publisher
    arm_move_state_pub_ = this->create_publisher<std_msgs::msg::Bool>("/arm_move_state", 10);

    // 双腿: 订阅 dog_joint_pos, QoS深度1只取最新(防积压老命令飞车)
    dog_joint_pos_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      "dog_joint_pos", rclcpp::QoS(1),
      std::bind(&ArmControlNode::joint_pos_dual_leg_callback, this, std::placeholders::_1));


    auto arm_move_topic = this->declare_parameter<std::string>("arm_move_topic", "/arm_move_info");
    arm_move_publisher_ = this->create_publisher<std_msgs::msg::Int32>(arm_move_topic, 10);
    body_general_control_pub_ = this->create_publisher<std_msgs::msg::Int32>("/body_general_control", 10);

    lift_cmd_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("/lift/joint_states/update", 1);
    joint_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);
    left_joint_max_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("/left_joint_max_states", 10);
    right_joint_max_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("/right_joint_max_states", 10);
    left_joint_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("/left_joint_states", 10);
    right_joint_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("/right_joint_states", 10);
    left_gripper_pub_ = this->create_publisher<std_msgs::msg::UInt8>("/left_gripper_state", 10);
    right_gripper_pub_ = this->create_publisher<std_msgs::msg::UInt8>("/right_gripper_state", 10);
    // 订阅 joint_states 以获取升降真实位置
    lift_joint_states_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/lift/joint_states", rclcpp::QoS(50),
      std::bind(&ArmControlNode::liftJointStateCallback, this, std::placeholders::_1));


    // //gripper test,jingyi
    // auto gripperLeft_marker_topic =
    //     this->declare_parameter<std::string>("gripperLeft_marker_topic", "/gripperLeft_envelopes");
    // gripperLeft_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(gripperLeft_marker_topic, 10);
    // // RCLCPP_INFO(this->get_logger(), "Created gripperLeft_marker_pub_ for topic: %s", gripperLeft_marker_topic.c_str());
    
    // auto gripperRight_marker_topic =
    //     this->declare_parameter<std::string>("gripperRight_marker_topic", "/gripperRight_envelopes");
    // gripperRight_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(gripperRight_marker_topic, 10);
    // // RCLCPP_INFO(this->get_logger(), "Created gripperRight_marker_pub_ for topic: %s", gripperRight_marker_topic.c_str());
    
    // //link test，jingyi
    // auto linkLeft_marker_topic =
    //     this->declare_parameter<std::string>("linkLeft_marker_topic", "/linkLeft_envelopes");
    // linkLeft_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(linkLeft_marker_topic, 10);
    // // RCLCPP_INFO(this->get_logger(), "Created linkLeft_marker_pub_ for topic: %s", linkLeft_marker_topic.c_str());
    
    // auto linkRight_marker_topic =
    //     this->declare_parameter<std::string>("linkRight_marker_topic", "/linkRight_envelopes");
    // linkRight_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(linkRight_marker_topic, 10);
    // // RCLCPP_INFO(this->get_logger(), "Created linkRight_marker_pub_ for topic: %s", linkRight_marker_topic.c_str());
    
    // //link test，jingyi
    // auto body_marker_topic =
    //     this->declare_parameter<std::string>("body_marker_topic", "/body_envelopes");
    // body_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(body_marker_topic, 10);
    // // RCLCPP_INFO(this->get_logger(), "Created body_marker_pub_ for topic: %s", body_marker_topic.c_str());

    // //TF test，jingyi
    // static_tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    // publishArmConnectionTF();
    // 用宏定义来区分使用串口通信还是CAN通信
#ifdef USE_SERIAL_COMMUNICATION
    RCLCPP_INFO(this->get_logger(), "Initializing Damiao motor control via Serial Communication");
    // 用串口通信初始化达妙电机控制对象 
    auto left_arm_serial =
        std::make_shared<SerialPort>(left_arm_device, B921600);
    auto right_arm_serial =
        std::make_shared<SerialPort>(right_arm_device, B921600);
    left_arm_mc_ = std::make_unique<damiao::Motor_Control>(left_arm_serial);
    right_arm_mc_ = std::make_unique<damiao::Motor_Control>(right_arm_serial);
#endif

#ifdef USE_CAN_COMMUNICATION
    // 直接用CAN通信初始化达妙电机控制对象
    RCLCPP_INFO(this->get_logger(), "Initializing Damiao motor control via CAN Communication");
    left_arm_mc_ = std::make_unique<damiao::Motor_Control>(left_arm_can_name);
    right_arm_mc_ = std::make_unique<damiao::Motor_Control>(right_arm_can_name);

    if (!left_arm_mc_->init()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to initialize left arm %s interface", left_arm_can_name.c_str());
      return;
    }
    if (!right_arm_mc_->init()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to initialize right arm %s interface", right_arm_can_name.c_str());
      return;
    }
    RCLCPP_INFO(this->get_logger(), "CAN interfaces initialized successfully");
#endif

    left_arm_motors_ = std::make_unique<damiao::Motor[]>(DTOF);
    right_arm_motors_ = std::make_unique<damiao::Motor[]>(DTOF);
    has_gripper_ = false;
    if (DTOF == 6)
    {
      left_gripper_ = std::make_unique<damiao::Gripper>(0x07);
      right_gripper_ = std::make_unique<damiao::Gripper>(0x07);
      has_gripper_ = true;

    }
    else if (DTOF == 7)
    {
      left_gripper_ = std::make_unique<damiao::Gripper>(0x08);
      right_gripper_ = std::make_unique<damiao::Gripper>(0x08);
      has_gripper_ = true;
    }

    left_arm_path_smoothers_ = std::make_unique<PathSmoother[]>(DTOF);
    right_arm_path_smoothers_ = std::make_unique<PathSmoother[]>(DTOF);
    for (int i = 0; i < DTOF; i++) {
      left_arm_path_smoothers_[i] = PathSmoother();
      right_arm_path_smoothers_[i] = PathSmoother();
    }

    if (DTOF == 12 && dual_leg_)
    {
      init_dual_leg_motors();
    }
    else if (DTOF == 12)
    {
      for (int i = 0; i < DTOF; i++) {
        auto slave_id = static_cast<Motor_id>(0x01 + i);
        auto master_id = static_cast<Motor_id>(0x11 + i);
        left_arm_motors_[i] = damiao::Motor(damiao::DM4340, slave_id, master_id);
        right_arm_motors_[i] = damiao::Motor(damiao::DM4340, slave_id, master_id);
      }
    }
    else
    {
      left_arm_motors_[0] = damiao::Motor(damiao::DM4340, 0x01, 0x11);
      left_arm_motors_[1] = damiao::Motor(damiao::DM4340, 0x02, 0x12);
      left_arm_motors_[2] = damiao::Motor(damiao::DM4340, 0x03, 0x13);
      left_arm_motors_[3] = damiao::Motor(damiao::DM4340, 0x04, 0x14);
      left_arm_motors_[4] = damiao::Motor(damiao::DM4310, 0x05, 0x15);
      left_arm_motors_[5] = damiao::Motor(damiao::DM4310, 0x06, 0x16);
      if (DTOF == 7)
      {
        left_arm_motors_[6] = damiao::Motor(damiao::DM4310, 0x07, 0x17);
      }

      right_arm_motors_[0] = damiao::Motor(damiao::DM4340, 0x01, 0x11);
      right_arm_motors_[1] = damiao::Motor(damiao::DM4340, 0x02, 0x12);
      right_arm_motors_[2] = damiao::Motor(damiao::DM4340, 0x03, 0x13);
      right_arm_motors_[3] = damiao::Motor(damiao::DM4340, 0x04, 0x14);
      right_arm_motors_[4] = damiao::Motor(damiao::DM4310, 0x05, 0x15);
      right_arm_motors_[5] = damiao::Motor(damiao::DM4310, 0x06, 0x16);
      if (DTOF == 7)
      {
        right_arm_motors_[6] = damiao::Motor(damiao::DM4310, 0x07, 0x17);
      }
    }
    const int n_motors_per_arm = (DTOF == 12 && dual_leg_) ? 6 : DTOF;
    for (int i = 0; i < n_motors_per_arm; i++) {
      left_arm_mc_->addMotor(&left_arm_motors_[i]);
      right_arm_mc_->addMotor(&right_arm_motors_[i]);
    }

    if (has_gripper_) {
      left_arm_mc_->addGripper(left_gripper_.get());
      right_arm_mc_->addGripper(right_gripper_.get());
    }

    const bool need_pos_vel_mode = (DTOF == 12);
    {
      std::unique_lock<std::mutex> lock(writeLocker_);
      const int n = (DTOF == 12 && dual_leg_) ? 6 : DTOF;
      for (int i = 0; i < n; i++) {
        left_arm_mc_->enable(left_arm_motors_[i]);
        right_arm_mc_->enable(right_arm_motors_[i]);

        if (need_pos_vel_mode) {
#if USE_POS_VEL_MODE
          if (!left_arm_mc_->switchControlMode(left_arm_motors_[i], damiao::POS_VEL_MODE)) {
            RCLCPP_WARN(this->get_logger(), "Failed to switch left motor %d to POS_VEL mode", i);
          }
          // if (!right_arm_mc_->switchControlMode(right_arm_motors_[i], damiao::POS_VEL_MODE)) {
          //   RCLCPP_WARN(this->get_logger(), "Failed to switch right motor %d to POS_VEL mode", i);
          // }
#endif
        }

      }
    }

    usleep(100000); // 等待电机稳定

    RCLCPP_INFO(this->get_logger(), "Motors enabled successfully");

    // restore_arm(left_arm_model_, left_arm_mc_, left_arm_motors_,
    //             left_arm_path_smoothers_);
    // restore_arm(right_arm_model_, right_arm_mc_, right_arm_motors_,
    //             right_arm_path_smoothers_);
    // restore_ = true;
    // RCLCPP_INFO(this->get_logger(), "Arms restored to default positions");
    // usleep(100000); // 等待电机稳定
    std::thread loop_thread(&ArmControlNode::loop, this);
    loop_thread.detach();

    RCLCPP_INFO(this->get_logger(), "Ready to Run...");
  }

  ~ArmControlNode() {
      // 停止老化线程
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (current_state_ == ArmState::AGING) {
          stop_aging_left_ = true;
          stop_aging_right_ = true;
          current_state_ = ArmState::ERROR;
        }
      }
      if (aging_thread_.joinable()) {
        aging_thread_.join();
      }
      
      {
          std::lock_guard<std::mutex> lock(queue_mutex_);
          stop_threads_ = true;
      }
      condition_.notify_all();
      for (std::thread &thread : thread_pool_) {
          if (thread.joinable()) {
              thread.join();
          }
      }
  }

private:
  // 订阅器与状态变量（升降）
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr lift_joint_states_sub_;
  double lift_target_ {0.0};
  double last_lift_pos_ {0.0};
  bool lift_reached_ {false};
  // 升降状态监控
  void liftJointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
    for (size_t i = 0; i < msg->name.size(); ++i) {
      if (msg->name[i] == "lift_joint") {
        if (i < msg->position.size()) {
          last_lift_pos_ = msg->position[i];
          // 只在设置了目标时才判断到位
          if (lift_target_ > 0.0) {
            lift_reached_ = std::fabs(last_lift_pos_ - lift_target_) <= 0.01; // 1cm 容差
          }
        }
        break;
      }
    }
  }


  void normalizeJoints(Eigen::VectorXd &joints) const
  {
    for (int i = 0; i < joints.size(); ++i)
    {
      joints(i) = std::fmod(joints(i) + M_PI, 2.0 * M_PI);
      if (joints[i] < 0.0)
      {
        joints(i) += 2.0 * M_PI;
      }
      joints(i) -= M_PI;
    }
  }

  inline double correctedMotorPosition(const double raw_position, const int joint_index) const
  {
    // 电机安装正反差异修正：仅对 5、6、7、10、11、12 号电机读值取反（joint_index 为 0-based）
    switch (joint_index) {
      case 4:
      case 5:
      case 6:
      case 9:
      case 10:
      case 11:
        return -raw_position;
      default:
        return raw_position;
    }
  }

  Eigen::VectorXd getCurrentPosition(const std::unique_ptr<damiao::Motor_Control> &dm,
                                     const std::unique_ptr<damiao::Motor[]> &motors) const
  {
    Eigen::VectorXd current_position = Eigen::VectorXd::Zero(DTOF);
    {
      std::unique_lock<std::mutex> lock(writeLocker_);
      for (int i = 0; i < DTOF; i++) {
        dm->refresh_motor_status(motors[i]);
      }
      for (int i = 0; i < DTOF; ++i) {
        current_position[i] = correctedMotorPosition(motors[i].Get_Position(), i);
      }
    }
    normalizeJoints(current_position);
    return current_position;
  }

  void loop() {
    std::vector<std::string> left_joint_names;
    std::vector<std::string> right_joint_names;
    left_joint_names.reserve(DTOF);
    right_joint_names.reserve(DTOF);
    for (int i = 1; i <= DTOF; ++i) {
      left_joint_names.push_back("joint" + std::to_string(i) + "-l");
      right_joint_names.push_back("joint" + std::to_string(i) + "-r");
    }
    while (rclcpp::ok()) {
      // song1120，以下为获取当前位置并计算正运动学
      // 诊断计时（仅日志，不改 refresh 逻辑）
      double loop_lock_wait_us = 0.0;
      double refresh_us = 0.0;
      {
        const auto wait_lock_start = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(writeLocker_);
        const auto lock_acquired = std::chrono::steady_clock::now();
        loop_lock_wait_us =
            std::chrono::duration<double, std::micro>(lock_acquired - wait_lock_start).count();
        const int n_refresh = (DTOF == 12 && dual_leg_) ? 6 : DTOF;
        for (int i = 0; i < n_refresh; i++) {
          right_arm_mc_->refresh_motor_status(right_arm_motors_[i]);
          left_arm_mc_->refresh_motor_status(left_arm_motors_[i]);
        }
        refresh_us =
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - lock_acquired).count();
      }

      //每30秒打印一次各个关节状态
      static auto last_print_time = std::chrono::steady_clock::now();
      if (std::chrono::steady_clock::now() - last_print_time > std::chrono::seconds(30)) {
          last_print_time = std::chrono::steady_clock::now();
          // RCLCPP_INFO(this->get_logger(), "Left Arm Joint States:");
          // for (int i = 0; i < DTOF; i++) {
          //     RCLCPP_INFO(this->get_logger(), "  Joint %d: Position: %.2f, Velocity: %.2f, Effort: %.2f",
          //                 i + 1, left_arm_motors_[i].Get_Position(), left_arm_motors_[i].Get_Velocity(), left_arm_motors_[i].Get_tau());
          // }
          // RCLCPP_INFO(this->get_logger(), "Right Arm Joint States:");
          // for (int i = 0; i < DTOF; i++) {
          //     RCLCPP_INFO(this->get_logger(), "  Joint %d: Position: %.2f, Velocity: %.2f, Effort: %.2f",
          //                 i + 1, right_arm_motors_[i].Get_Position(), right_arm_motors_[i].Get_Velocity(), right_arm_motors_[i].Get_tau());
          // }
      }
      
   
      if (DTOF == 12 && dual_leg_) {
        publish_dual_leg_joint_states(left_joint_names);
      } else {
      auto current = getCurrentPosition(left_arm_mc_, left_arm_motors_); // Initialize current position
      auto joint_state_msg = sensor_msgs::msg::JointState();
      auto joint_max_state_msg = sensor_msgs::msg::JointState();
      joint_state_msg.header.frame_id = "left_arm";
      joint_state_msg.header.stamp = this->now();
      joint_state_msg.name = left_joint_names;
      joint_state_msg.position = std::vector<double>(current.data(), current.data() + current.size());
      joint_max_state_msg = joint_state_msg;
      for (int i = 0; i < DTOF; i++)
      {
        joint_state_msg.velocity.push_back(left_arm_motors_[i].Get_Velocity());
        float left_tau = left_arm_motors_[i].Get_tau();
        joint_state_msg.effort.push_back(left_tau);
        // 更新最大力矩
        {
          std::lock_guard<std::mutex> lock(max_tau_mutex_);
          if (std::fabs(left_tau) > std::fabs(left_max_tau_[i])) {
            left_max_tau_[i] = left_tau;
          }
          joint_max_state_msg.effort.push_back(left_max_tau_[i]);
        }
        if (left_arm_motors_[i].GetErrCode() > 1 && left_arm_motors_[i].GetErrCode() != prev_left_err_code_ )
        {
          prev_left_err_code_ = left_arm_motors_[i].GetErrCode();
          RCLCPP_INFO(this->get_logger(), "Left Arm Motor ID %u Error: %s",
                        static_cast<unsigned int>(left_arm_motors_[i].GetSlaveId()),
                        damiao::Motor::error_code_to_string(left_arm_motors_[i].GetErrCode()));
        }
      }
      for (int i = 0; i < DTOF; i++)
      {
        joint_state_msg.velocity[i] = correctedMotorPosition(joint_state_msg.velocity[i], i);
      }
      joint_state_pub_->publish(joint_state_msg);
      left_joint_state_pub_->publish(joint_state_msg);
      left_joint_max_state_pub_->publish(joint_max_state_msg);
      }

      // //DEBUG:jingyi
      //  //link test，gripper test, body test,jingyi
      // PaddingValues current_padding = padding_values_.load();
      // // 使用互斥锁保护pino_model的访问，避免与checkCollision()的竞态条件
      // std::vector<Envelope> links_l_copy;
      // std::vector<Envelope> grippers_l_copy;
      // std::vector<Envelope> body_copy;
      // {
      //   std::lock_guard<std::mutex> lock(pino_model_mutex_);
        
      //   pino_model_left_->setScale(current_padding.link_scale); //link比例缩放
      //   pino_model_left_->updateLinkEnvelopes(current);
      //   pino_model_left_->setPadding(current_padding.left); 
      //   pino_model_left_->updateGripperEnvelopes(current);
      //   pino_model_left_->setPadding(current_padding.body);
      //   pino_model_left_->initBodyEnvelopeBox();

      //   // 在锁的保护下做拷贝
      //   links_l_copy = pino_model_left_->getLinkEnvelopes();
      //   grippers_l_copy = pino_model_left_->getGripperEnvelopes();
      //   body_copy = pino_model_left_->getBodyEnvelopes();
      // }  // ← 锁释放

      // // 发布操作在锁外进行
      // publishLinkEnvelopes(links_l_copy, 1);
      // publishGripperEnvelopes(grippers_l_copy, 1);
      // publishBodyEnvelopes(body_copy, 1);
      
      // current = getCurrentPosition(right_arm_mc_, right_arm_motors_); // Initialize current position
      // auto joint_state_msg2 = sensor_msgs::msg::JointState();
      // joint_state_msg2.header.frame_id = "right_arm";
      // joint_state_msg2.header.stamp = this->now();
      // joint_state_msg2.name = right_joint_names;
      // joint_state_msg2.position = std::vector<double>(current.data(), current.data() + current.size());
      // joint_max_state_msg = joint_state_msg2;
      // joint_max_state_msg.effort.clear();
      // for (int i = 0; i < DTOF; i++)
      // {
      //   joint_state_msg2.velocity.push_back(right_arm_motors_[i].Get_Velocity());
      //   float right_tau = right_arm_motors_[i].Get_tau();
      //   joint_state_msg2.effort.push_back(right_tau);
      //   // 更新最大力矩
      //   {
      //     std::lock_guard<std::mutex> lock(max_tau_mutex_);
      //     if (std::fabs(right_tau) > std::fabs(right_max_tau_[i])) {
      //       right_max_tau_[i] = right_tau;
      //     }
      //     joint_max_state_msg.effort.push_back(right_max_tau_[i]);
      //   }
      //   if (right_arm_motors_[i].GetErrCode() > 1 && right_arm_motors_[i].GetErrCode() != prev_right_err_code_)
      //   {
      //     prev_right_err_code_ = right_arm_motors_[i].GetErrCode();
      //     RCLCPP_INFO(this->get_logger(), "Right Arm Motor ID %u Error: %s",
      //                   static_cast<unsigned int>(right_arm_motors_[i].GetSlaveId()),
      //                   damiao::Motor::error_code_to_string(right_arm_motors_[i].GetErrCode()));
      //   }
      // }
      // joint_state_pub_->publish(joint_state_msg2);
      // right_joint_state_pub_->publish(joint_state_msg2);
      // right_joint_max_state_pub_->publish(joint_max_state_msg);

      // //DEBUG:jingyi
      //  //link test，gripper test, body test(TODO),jingyi
      // // 使用互斥锁保护pino_model的访问，避免与checkCollision()的竞态条件
      // std::vector<Envelope> links_r_copy;
      // std::vector<Envelope> grippers_r_copy;
      // {
      //   std::lock_guard<std::mutex> lock(pino_model_mutex_);
        
      //   pino_model_right_->setScale(current_padding.link_scale); //link比例缩放
      //   pino_model_right_->updateLinkEnvelopes(current);
      //   pino_model_right_->setPadding(current_padding.right); // 设置夹爪预留间隙
      //   pino_model_right_->updateGripperEnvelopes(current);

      //   // 在锁的保护下做拷贝
      //   links_r_copy = pino_model_right_->getLinkEnvelopes();
      //   grippers_r_copy = pino_model_right_->getGripperEnvelopes();
      // }  // ← 锁释放
      
      // // 发布操作在锁外进行
      // publishLinkEnvelopes(links_r_copy, 0);
      // publishGripperEnvelopes(grippers_r_copy, 0);

 
      //1110 byf 组装左臂错误 (motor_id 从 1 开始)
      std::vector<std::pair<int,int>> left_errors;
      left_errors.reserve(DTOF);
      for (int i = 0; i < DTOF; ++i) {
        int err = static_cast<int>(left_arm_motors_[i].GetErrCode());
        left_errors.emplace_back(i+1, err);
      }
      publishMotorErrors("left", left_errors);

      // 组装右臂错误
      std::vector<std::pair<int,int>> right_errors;
      right_errors.reserve(DTOF);
      for (int i = 0; i < DTOF; ++i) {
        int err = static_cast<int>(right_arm_motors_[i].GetErrCode());
        right_errors.emplace_back(i+1, err);
      }
      publishMotorErrors("Right", right_errors);

      //发布碰撞状态及同步状态，jingyi
      publishArmErrors();

      if (has_gripper_) {
        auto left_gripper_state = std_msgs::msg::UInt8();
        auto right_gripper_state = std_msgs::msg::UInt8();
        left_gripper_state.data = left_gripper_->GetPosition();
        right_gripper_state.data = right_gripper_->GetPosition();
        left_gripper_pub_->publish(left_gripper_state);
        right_gripper_pub_->publish(right_gripper_state);

        {
          std::unique_lock<std::mutex> lock(writeLocker_);
          if (left_percent_ != left_gripper_state.data)
          {
            left_arm_mc_->control_gripper(left_gripper_.get(), left_percent_, 255, 255);
          }

          if (right_percent_ != right_gripper_state.data)
          {
            right_arm_mc_->control_gripper(right_gripper_.get(), right_percent_, 255, 255);
          }
        }
      }
      // 诊断：双腿模式下把每关节状态(q/v/tau/err)+loop 计时写入本地 CSV（终端不再刷 perf）
      // 文件：dual_leg_diag/dual_leg_state_<时间戳>.csv —— 列：时间、loop 周期、refresh 耗时、loop 锁等待、12 关节(q,v,tau,err)
      // 关节顺序：j0..5=左腿 left_arm_motors_[0..5]，j6..11=右腿 right_arm_motors_[0..5]
      if (DTOF == 12 && dual_leg_) {
        static std::ofstream state_log = [] {
          std::ofstream f(diag_dir() + "dual_leg_state_" + diag_run_tag() + ".csv", std::ios::out | std::ios::trunc);
          f << "t_ns,loop_dt_ms,refresh_us,loop_lock_wait_us";
          for (int j = 0; j < 12; ++j) f << ",q" << j << ",v" << j << ",tau" << j << ",err" << j;
          f << "\n";
          return f;
        }();
        static std::chrono::steady_clock::time_point last_iter = std::chrono::steady_clock::now();
        const auto iter_now = std::chrono::steady_clock::now();
        const double loop_dt_ms = std::chrono::duration<double, std::milli>(iter_now - last_iter).count();
        last_iter = iter_now;
        if (state_log.is_open()) {
          state_log.precision(6);
          state_log << this->now().nanoseconds()
                    << "," << loop_dt_ms << "," << refresh_us << "," << loop_lock_wait_us;
          for (int j = 0; j < 12; ++j) {
            damiao::Motor & m = (j < 6) ? left_arm_motors_[j] : right_arm_motors_[j - 6];
            state_log << "," << m.Get_Position()
                      << "," << m.Get_Velocity()
                      << "," << m.Get_tau()
                      << "," << static_cast<unsigned>(m.GetErrCode());
          }
          state_log << "\n";
          state_log.flush();
        }
      }
      // usleep(500000); // Sleep for 500 milliseconds
      usleep(10000); // Sleep for 10 milliseconds //SLAM要求不能低于100HZ频率发布
      // if(!start_)
      // {
      //   for (int i = 0; i < DTOF; i++) {
      //     left_arm_mc_->refresh_motor_status(left_arm_motors_[i]);
      //     right_arm_mc_->refresh_motor_status(right_arm_motors_[i]);
      //   }
      // }
    }
  }

//   // DEBUG:发布连接变换(gripper test,body test,link test)，jingyi
// void publishArmConnectionTF()
// {
//     geometry_msgs::msg::TransformStamped transform;
//     transform.header.stamp = this->now();
//     transform.header.frame_id = "base_link-L";
//     transform.child_frame_id = "base_link-r";
    
//     transform.transform.translation.x = 0.0;
//     transform.transform.translation.y = 0.0;
//     transform.transform.translation.z = -0.113; // 根据实际连接偏移设置10.8cm
    
//     // 绕Y轴180度，绕Z轴180度
//     tf2::Quaternion q;
//     q.setRPY(0.0, 180.0 * M_PI / 180.0, 182.0 * M_PI / 180.0);
//     transform.transform.rotation.x = q.x();
//     transform.transform.rotation.y = q.y();
//     transform.transform.rotation.z = q.z();
//     transform.transform.rotation.w = q.w();
    
//     static_tf_broadcaster_->sendTransform(transform);
// }

  // 双腿模式电机初始化：每腿 6 电机，CAN ID 1,2,3,4,6,7（跳过 5），与原 12 单 CAN 模式区分
  void init_dual_leg_motors()
  {
    left_arm_motors_[0]  = damiao::Motor(damiao::DM4340, 0x01, 0x11);
    left_arm_motors_[1]  = damiao::Motor(damiao::DM4340, 0x02, 0x12);
    left_arm_motors_[2]  = damiao::Motor(damiao::DM4340, 0x03, 0x13);
    left_arm_motors_[3]  = damiao::Motor(damiao::DM4340, 0x04, 0x14);
    left_arm_motors_[4]  = damiao::Motor(damiao::DM4340, 0x06, 0x16);
    left_arm_motors_[5]  = damiao::Motor(damiao::DM4310, 0x07, 0x17);
    right_arm_motors_[0] = damiao::Motor(damiao::DM4340, 0x01, 0x11);
    right_arm_motors_[1] = damiao::Motor(damiao::DM4340, 0x02, 0x12);
    right_arm_motors_[2] = damiao::Motor(damiao::DM4340, 0x03, 0x13);
    right_arm_motors_[3] = damiao::Motor(damiao::DM4340, 0x04, 0x14);
    right_arm_motors_[4] = damiao::Motor(damiao::DM4340, 0x06, 0x16);
    right_arm_motors_[5] = damiao::Motor(damiao::DM4310, 0x07, 0x17);
  }





  
  // 双腿模式控制回调：12 维输入，前 6 给 left_arm_mc_，后 6 给 right_arm_mc_
  void joint_pos_dual_leg_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    using steady_clock = std::chrono::steady_clock;
    const auto cb_enter = steady_clock::now();  // 诊断：回调进入时刻（用于计算到达间隔/频率）

    if (msg->data.size() != 12) {
      RCLCPP_WARN(this->get_logger(),
                  "joint_pos_dual_leg: invalid data size %zu (expect 12)",
                  msg->data.size());
      return;
    }

    // 双腿命令方向修正表（idx 0..5=左腿1..6，6..11=右腿1..6；+1 不变，-1 取反）。
    // 起点等价于旧的 correctedMotorPosition（取反 {4,5,6,9,10,11}），并按实测把执行方向反了的
    // 右腿 3/5/6（idx 8/10/11）翻过来。要再调某关节执行方向，只改对应这一项即可。
    // 注意：与 publish_dual_leg_joint_states 的 kLegFbSign 配合——改这里的命令方向，会同时改变该
    //      关节裸编码器反馈相对目标的符号，若返回值随之反了，请同步调那边对应项。
    static constexpr double kLegCmdSign[12] = {
        +1, +1, -1, +1, -1, +1,   // 左腿 1..6
        -1, +1, -1, -1, +1, +1    // 右腿 1..6（idx 8/10/11 已按右腿3/5/6执行反向修正）
    };
    for (int i = 0; i < 12; i++) {
      msg->data[i] = kLegCmdSign[i] * msg->data[i];
    }

    std::vector<double> vel_cmd(12, 0.0);
    {
      std::lock_guard<std::mutex> lock(dog_joint_pos_mutex_);
      const auto now = this->now();
      if (dog_joint_pos_has_last_) {
        double dt = (now - dog_joint_pos_last_stamp_).seconds();
        if (dt < 0.001) dt = 0.001;
        for (int i = 0; i < 12; ++i) {
          double v = (msg->data[i] - dog_joint_pos_last_cmd_[i]) / dt;
          if (v >  config_max_vel_) v =  config_max_vel_;
          if (v < -config_max_vel_) v = -config_max_vel_;
          vel_cmd[i] = v;
        }
      } else {
        // 首帧没有上一帧可差分，前馈速度必须为 0：MIT 模式下 dq_des 是前馈项
        // (τ=kp·(q_des−q)+kd·(dq_des−dq))，给非 0 会留下 kd·dq_des/kp 的稳态位置偏置；
        // 且 --once 单帧下永不被刷新，导致首发跑到错误角度、要再发一遍才正。
        for (int i = 0; i < 12; ++i) {
          vel_cmd[i] = 0.0;
        }
      }
      dog_joint_pos_last_cmd_ = msg->data;
      dog_joint_pos_last_stamp_ = now;
      dog_joint_pos_has_last_ = true;
    }

    // 诊断计时变量（不改任何下发逻辑）
    double lock_wait_us = 0.0;
    double left_send_us = 0.0;
    double right_send_us = 0.0;
    try {
      const auto wait_lock_start = steady_clock::now();
      std::unique_lock<std::mutex> lock(writeLocker_);
      const auto lock_acquired = steady_clock::now();
      lock_wait_us = std::chrono::duration<double, std::micro>(lock_acquired - wait_lock_start).count();

      if (!dog_joint_pos_enabled_) {
        const auto enable_start = steady_clock::now();
        for (int i = 0; i < 6; ++i) {
          left_arm_mc_->enable(left_arm_motors_[i]);
          right_arm_mc_->enable(right_arm_motors_[i]);
        }
        dog_joint_pos_enabled_ = true;
        const double enable_ms =
            std::chrono::duration<double, std::milli>(steady_clock::now() - enable_start).count();
        RCLCPP_INFO(this->get_logger(),
                    "joint_pos_dual_leg: 首帧 enable 12 个电机耗时 %.1f ms (持 writeLocker_)", enable_ms);
      }

      for (int i = 0; i < 6; ++i) {
        const double dq_l = std::clamp(vel_cmd[i],     -config_max_vel_, config_max_vel_);
        const double dq_r = std::clamp(vel_cmd[i + 6], -config_max_vel_, config_max_vel_);
        const auto t0 = steady_clock::now();
        // std::cout << msg->data[i] << " left position " << i << std::endl;
        // std::cout << msg->data[i+6] << " right position " << i+6 << std::endl;
        left_arm_mc_->control_mit(left_arm_motors_[i],
                                  config_kps_[i], config_kds_[i],
                                  msg->data[i], dq_l, 0.0);
        const auto t1 = steady_clock::now();
        right_arm_mc_->control_mit(right_arm_motors_[i],
                                   config_kps_[i + 6], config_kds_[i + 6],
                                   msg->data[i + 6], dq_r, 0.0);
        const auto t2 = steady_clock::now();
        left_send_us  += std::chrono::duration<double, std::micro>(t1 - t0).count();
        right_send_us += std::chrono::duration<double, std::micro>(t2 - t1).count();
      }
    } catch (const std::exception &e) {
      RCLCPP_WARN(this->get_logger(),
                  "joint_pos_dual_leg: control_mit exception: %s", e.what());
      return;
    }

    // 诊断：命令路径每帧写入本地 CSV（终端不再刷 perf，原 dual_leg cmd 行保留）
    // 文件：dual_leg_diag/dual_leg_cmd_<时间戳>.csv —— 列：时间、到达间隔、锁等待、左右下发耗时、12 个命令位置 q、12 个下发速度 dq
    {
      static std::ofstream cmd_log = [] {
        std::ofstream f(diag_dir() + "dual_leg_cmd_" + diag_run_tag() + ".csv", std::ios::out | std::ios::trunc);
        f << "t_ns,inter_arrival_ms,lock_wait_us,left_send_us,right_send_us";
        for (int i = 0; i < 12; ++i) f << ",q" << i;
        for (int i = 0; i < 12; ++i) f << ",dq" << i;
        f << "\n";
        return f;
      }();
      static steady_clock::time_point last_enter = cb_enter;
      const double gap_ms = std::chrono::duration<double, std::milli>(cb_enter - last_enter).count();
      last_enter = cb_enter;
      if (cmd_log.is_open()) {
        cmd_log.precision(6);
        cmd_log << this->now().nanoseconds()
                << "," << gap_ms << "," << lock_wait_us
                << "," << left_send_us << "," << right_send_us;
        for (int i = 0; i < 12; ++i) cmd_log << "," << msg->data[i];
        for (int i = 0; i < 12; ++i) cmd_log << "," << std::clamp(vel_cmd[i], -config_max_vel_, config_max_vel_);
        cmd_log << "\n";
        cmd_log.flush();
      }
    }

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
        "dual_leg cmd L[0..5]=%.3f,%.3f,%.3f,%.3f,%.3f,%.3f R[0..5]=%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
        msg->data[0], msg->data[1], msg->data[2], msg->data[3], msg->data[4], msg->data[5],
        msg->data[6], msg->data[7], msg->data[8], msg->data[9], msg->data[10], msg->data[11]);
  }

  // 双腿模式 JointState 发布：12 维，前 6 = 左腿，后 6 = 右腿
  void publish_dual_leg_joint_states(const std::vector<std::string> & joint_names)
  {
    auto joint_state_msg = sensor_msgs::msg::JointState();
    auto joint_max_state_msg = sensor_msgs::msg::JointState();
    joint_state_msg.header.frame_id = "dual_leg";
    joint_state_msg.header.stamp = this->now();
    joint_state_msg.name = joint_names;
    joint_state_msg.position.resize(12);
    joint_state_msg.velocity.reserve(12);
    joint_state_msg.effort.reserve(12);

    // 双腿反馈方向修正表（idx 0..5=左腿1..6，6..11=右腿1..6；+1 不变，-1 取反）。
    // 按 /left_joint_states 实测取反：左3(idx2)、左5(idx4)、右1(idx6)、右3(idx8)、右4(idx9)。位置与速度一并取反。
    static constexpr double kLegFbSign[12] = {
        +1, +1, -1, +1, -1, +1,   // 左腿 1..6（idx2=左3、idx4=左5 取反）
        -1, +1, -1, -1, +1, +1    // 右腿 1..6（idx6=右1、idx8=右3、idx9=右4 取反）
    };

    for (int i = 0; i < 6; ++i) {
      joint_state_msg.position[i]     = kLegFbSign[i]     * left_arm_motors_[i].Get_Position();
      joint_state_msg.position[i + 6] = kLegFbSign[i + 6] * right_arm_motors_[i].Get_Position();
    }
    joint_max_state_msg = joint_state_msg;
    joint_max_state_msg.effort.clear();

    for (int i = 0; i < 6; ++i) {
      joint_state_msg.velocity.push_back(kLegFbSign[i] * left_arm_motors_[i].Get_Velocity());
      const float tau_l = left_arm_motors_[i].Get_tau();
      joint_state_msg.effort.push_back(tau_l);
      {
        std::lock_guard<std::mutex> lock(max_tau_mutex_);
        if (std::fabs(tau_l) > std::fabs(left_max_tau_[i])) {
          left_max_tau_[i] = tau_l;
        }
        joint_max_state_msg.effort.push_back(left_max_tau_[i]);
      }
    }
    for (int i = 0; i < 6; ++i) {
      joint_state_msg.velocity.push_back(kLegFbSign[i + 6] * right_arm_motors_[i].Get_Velocity());
      const float tau_r = right_arm_motors_[i].Get_tau();
      joint_state_msg.effort.push_back(tau_r);
      {
        std::lock_guard<std::mutex> lock(max_tau_mutex_);
        if (std::fabs(tau_r) > std::fabs(right_max_tau_[i])) {
          right_max_tau_[i] = tau_r;
        }
        joint_max_state_msg.effort.push_back(right_max_tau_[i]);
      }
    }

    joint_state_pub_->publish(joint_state_msg);
    left_joint_state_pub_->publish(joint_state_msg);
    left_joint_max_state_pub_->publish(joint_max_state_msg);
  }

  // 新增：movep 回调，输入末端位姿与可选速度，做笛卡尔直线插补后逐段关节空间跟随
  // 发布电机变化到全使能和变化到全失能的状态（只在状态变化时发布一次），jingyi
  void publishMotorErrors(const std::string& arm_name,
                          const std::vector<std::pair<int,int>>& id_errs) {
    if (!motors_err_pub_) return;

    const bool is_left = (arm_name == "left" || arm_name == "Left");
    std::vector<std::pair<int,int>> events;

    bool now_enabled_all = true;
    bool now_disabled_all = true;

    // 获取左右手对应的上一次状态
    bool& last_enabled_all = is_left ? last_left_enabled_all_ : last_right_enabled_all_;
    bool& last_disabled_all = is_left ? last_left_disabled_all_ : last_right_disabled_all_;

    // ========== 第一轮：收集故障变化和计算全状态 ==========
    for (const auto& p : id_errs) {
      int motor_id = p.first;
      int now_code = p.second;
      int idx = motor_id - 1;
      if (idx < 0 || idx >= DTOF) continue;

      int& prev_code = is_left ? last_left_err_[idx] : last_right_err_[idx];

      // 首次仅建缓存；若想启动时就上报已有故障，可在此加: if (now_code > 1) events.emplace_back(motor_id, now_code);
      if (prev_code == -1) {
        prev_code = now_code;
        // 也参与全状态计算
        if (now_code != 0) now_disabled_all = false;
        if (now_code != 1) now_enabled_all = false;
        continue;
      }

      bool prev_fault = (prev_code > 1);
      bool now_fault  = (now_code > 1);

      // 检查当前全状态
      if (now_code != 0) now_disabled_all = false;
      if (now_code != 1) now_enabled_all = false;

      // 触发三种事件
      if ((!prev_fault && now_fault) ||                      // 新故障 0/1 -> >1
          (prev_fault && now_fault && now_code != prev_code) || // 故障码变化 >1 -> >1
          (prev_fault && !now_fault)) {                      // 故障清除 >1 -> 0/1
        events.emplace_back(motor_id, now_code);             // 清除时携带实际0或1
      }

      // 更新缓存
      prev_code = now_code;
    }

    // ========== 第二轮：检测全状态变化并发布（优先级最高，直接 return） ==========
    
    // 从非全失能 -> 全失能
    if (now_disabled_all && !last_disabled_all) {
      std::ostringstream oss;
      oss << "{\"master_or_slave\":\"slave\",\"arm\":\"" << (is_left ? "left" : "Right") << "\",\"errors\":[";
      for (size_t i = 0; i < id_errs.size(); ++i) {
        oss << "{\"id\":" << id_errs[i].first << ",\"err\":\"" << damiao::Motor::error_code_to_string(id_errs[i].second) << "\"}";
        if (i + 1 < id_errs.size()) oss << ",";
      }
      oss << "]}";

      std_msgs::msg::String msg;
      msg.data = oss.str();
      motors_err_pub_->publish(msg);

      RCLCPP_INFO(this->get_logger(), "Arm %s all motors disabled", is_left ? "left" : "right");

      last_disabled_all = true;
      last_enabled_all = false;
      return;
    }

    // 从非全使能 -> 全使能
    if (now_enabled_all && !last_enabled_all) {
      std::ostringstream oss;
      oss << "{\"master_or_slave\":\"slave\",\"arm\":\"" << (is_left ? "left" : "Right") << "\",\"errors\":[";
      for (size_t i = 0; i < id_errs.size(); ++i) {
        oss << "{\"id\":" << id_errs[i].first << ",\"err\":\"" << damiao::Motor::error_code_to_string(id_errs[i].second) << "\"}";
        if (i + 1 < id_errs.size()) oss << ",";
      }
      oss << "]}";

      std_msgs::msg::String msg;
      msg.data = oss.str();
      motors_err_pub_->publish(msg);

      RCLCPP_INFO(this->get_logger(), "Arm %s all motors enabled", is_left ? "left" : "right");

      last_disabled_all = false;
      last_enabled_all = true;
      return;
    }

    // ========== 第三轮：更新全状态缓存（不符合上面的转移时） ==========
    last_disabled_all = now_disabled_all;
    last_enabled_all = now_enabled_all;

    // ========== 第四轮：如有故障变化事件，发布（只在不是全状态变化时） ==========
    if (events.empty()) return;

    // {"arm":"left|Right","errors":[{"id":X,"err":Y},...]}
    std::ostringstream oss;
    oss << "{\"master_or_slave\":\"slave\",\"arm\":\"" << (is_left ? "left" : "Right") << "\",\"errors\":[";
    for (size_t i = 0; i < events.size(); ++i) {
      if((events[i].second >1 && events[i].second < 7) || events[i].second > 14) events[i].second = 99; // 将电机错误码 2-6 和大于14的码统一为99，方便前端显示
      oss << "{\"id\":" << events[i].first << ",\"err\":\"" << damiao::Motor::error_code_to_string(events[i].second) << "\"}";
      if (i + 1 < events.size()) oss << ",";
    }
    oss << "]}";


    std_msgs::msg::String msg;
    msg.data = oss.str();
    motors_err_pub_->publish(msg);
  }

  void publishArmErrors() {
    if (!motors_err_pub_) return;  // ← 改用独立的 arm_err_pub_

    // 读取当前状态
    int current_collision_sync = collide_sync_;

    // 检查是否有状态变化（任何一个flag变化都发布）
    bool state_changed = (current_collision_sync != last_collision_sync_state_);

    if (!state_changed) {
      return; // 没有状态变化，不发布
    }

    // 构建并发布消息
    std::ostringstream oss;
    oss << "{\"collide_and_sync\":\"" << collide_sync_to_string(collide_sync_) << "\"}";

    std_msgs::msg::String msg;
    msg.data = oss.str();
    motors_err_pub_->publish(msg);  // ← 改用独立的 arm_err_pub_

    // 更新上一次状态
    last_collision_sync_state_ = current_collision_sync;
  }

  static const char* collide_sync_to_string(int code)
        {
            switch (code)
            {
                case 10: return "无碰撞且从臂已经与主臂同步";
                case 11: return "从臂的左臂与身体碰撞，需要移动主臂到无碰撞的位置";
                case 12: return "从臂的左臂自碰撞，需要移动主臂到无碰撞的位置";
                case 13: return "从臂的右臂与身体碰撞，需要移动主臂到无碰撞的位置";
                case 14: return "从臂的右臂自碰撞，需要移动主臂到无碰撞的位置";
                case 15: return "双臂发生碰撞，需要移动主臂到无碰撞的位置";
                case 16: return "从臂没有跟踪上主臂，且从臂正在跟踪中";
            }
        }

  // 新增：对关节限位进行裁剪，超过则使用 min/max 替换
  void enforceJointLimits(Eigen::VectorXd &joints, const std::vector<double> &min_joints, const std::vector<double> &max_joints) const
  {
    if (min_joints.size() != max_joints.size()) {
      RCLCPP_WARN(this->get_logger(), "Joint limits size mismatch: min=%zu max=%zu", min_joints.size(), max_joints.size());
    }
    if (joints.size() != static_cast<Eigen::Index>(min_joints.size())) {
      // 仅提示，不强制报错，实际裁剪仍按可用范围
      RCLCPP_WARN(this->get_logger(), "Joint vector size %ld != limits size %zu", long(joints.size()), min_joints.size());
    }
    for (Eigen::Index i = 0; i < joints.size(); ++i) {
      if (i < static_cast<Eigen::Index>(min_joints.size()) && i < static_cast<Eigen::Index>(max_joints.size())) {
        if (!std::isfinite(joints[i])) {
          RCLCPP_WARN(this->get_logger(), "Joint %ld not finite before clamp: %f", long(i), joints[i]);
          continue; // 跳过不合法值，交给后续检查
        }
        joints[i] = std::clamp(joints[i], min_joints[i], max_joints[i]);
      }
    }
  }


  // // 笛卡尔空间运动, pos - 末端执行器位置, quat - 末端执行器方向四元数, speed -
  // // 速度
  // 笛卡尔空间运动（模仿示例）：根据目标末端位姿一次IK求解目标关节，随后做轨迹执行
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr dog_joint_pos_sub_;
  // rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr gripperLeft_marker_pub_;//gripper test,jingyi
  // rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr gripperRight_marker_pub_;//gripper test,jingyi
  // rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr linkLeft_marker_pub_;//link envelope test,jingyi
  // rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr linkRight_marker_pub_;//link envelope test,jingyi
  // rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr body_marker_pub_;//body envelope test,jingyi
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr arm_move_state_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr arm_move_publisher_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr body_general_control_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr left_joint_max_state_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr right_joint_max_state_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr left_joint_state_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr right_joint_state_pub_;
  rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr left_gripper_pub_;
  rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr right_gripper_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr motors_err_pub_; // 电机错误，碰撞检测，主从臂同步监测发布器
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr lift_cmd_pub_; // 新增：升降绝对位置发布器
  std::array<int, DTOF> last_left_err_;
  std::array<int, DTOF> last_right_err_;
  int prev_left_err_code_ = 1;
  int prev_right_err_code_ = 1;


  // std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;//静态tf广播器

  rclcpp::TimerBase::SharedPtr timer_;
  std::unique_ptr<damiao::Motor_Control> left_arm_mc_;
  std::unique_ptr<PathSmoother[]> left_arm_path_smoothers_;
  std::unique_ptr<damiao::Motor[]> left_arm_motors_;
  std::unique_ptr<damiao::Gripper> left_gripper_;

  std::unique_ptr<damiao::Motor_Control> right_arm_mc_;
  std::unique_ptr<PathSmoother[]> right_arm_path_smoothers_;
  std::unique_ptr<damiao::Motor[]> right_arm_motors_;
  std::unique_ptr<damiao::Gripper> right_gripper_;
  bool has_gripper_ = false;

  double config_max_vel_;                           // 最大速度
  double config_max_acc_;                           // 最大加速度
  double config_max_jerk_;                          // 最大加加速度
  double config_time_step_;                         // 时间步长
  std::vector<double> config_kps_;                  // 比例系数
  std::vector<double> config_kds_;                  // 微分系数
  std::vector<double> config_default_joints_zero_;  // 默认姿态
  std::vector<double> config_default_joints_left_;  // 左臂默认姿态
  std::vector<double> config_default_joints_right_; // 右臂默认姿态

  double lift_default_height = 1.02;

  
  // 用于碰撞与同步信息的发布,jingyi
  int collide_sync_ = 0; // 碰撞与同步状态标志
  int last_collision_sync_state_ = 0;
  
  // 左右手电机全状态标志（分别维护，避免冲突）,jingyi
  bool last_left_enabled_all_{false};
  bool last_left_disabled_all_{false};
  bool last_right_enabled_all_{false};
  bool last_right_disabled_all_{false};


  //----线程池相关----
  const int thread_pool_size_ = 2; // 线程池大小
  std::vector<std::thread> thread_pool_;
  std::queue<std::function<void()>> task_queue_;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_threads_ = false;

  bool start_ = false;
  bool restore_ = false;

  // 机械臂状态与老化测试相关
  ArmState current_state_;
  std::mutex state_mutex_;
  std::atomic<bool> stop_aging_left_{false};
  std::atomic<bool> stop_aging_right_{false};
  std::thread aging_thread_;

  const double walk_l1_ = 0.15;
  const double walk_l2_ = 0.15;
  const double walk_xs_ = -0.05;
  const double walk_xf_ = 0.03;
  const double walk_h0_ = 0.28;
  const double walk_h_ = 0.04;


  std::vector<double> control_position_left_; // 用于模拟位置更新
  std::vector<double> control_position_right_; // 用于模拟位置更新

  std::mutex dog_joint_pos_mutex_;
  std::vector<double> dog_joint_pos_last_cmd_ = std::vector<double>(DTOF, 0.0);
  rclcpp::Time dog_joint_pos_last_stamp_;
  bool dog_joint_pos_has_last_ = false;
  bool dog_joint_pos_enabled_ = false;
  bool dual_leg_{true};
  mutable std::mutex mutexPosition_left_;
  mutable std::mutex mutexPosition_right_;
  mutable std::mutex writeLocker_;  // 夹爪控制锁
  // 存储左右臂每个关节的最大力矩
  mutable std::mutex max_tau_mutex_;
  std::vector<float> left_max_tau_;
  std::vector<float> right_max_tau_;
  bool delta_mode_ = false;

  std::vector<double> max_joints_left_;  // 左臂最大姿态
  std::vector<double> max_joints_right_; // 右臂最大姿态
  std::vector<double> min_joints_left_;  // 左臂最小姿态
  std::vector<double> min_joints_right_; // 右臂最小姿态

  std::vector<double> max_delta_left_;  // 左臂上限增量
  std::vector<double> max_delta_right_;  // 右臂上限增量
  std::vector<double> min_delta_left_; // 左臂下限增量
  std::vector<double> min_delta_right_; // 右臂下限增量

  int left_percent_ = 0; // 左夹爪控制位置
  int right_percent_ = 0; // 右夹爪控制位置
  std::vector<std::unique_ptr<PathSmoother>> smoother_; // Array of PathSmoother for each joint

  std::atomic<bool> ignore_gripper_collision_{true}; // 是否忽略夹爪碰撞检测
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;
  try {
    const auto share_dir = ament_index_cpp::get_package_share_directory("armcontrol");
    const auto params_file = share_dir + std::string("/config/arm_control_node.yaml");
    std::ifstream ifs(params_file);
    if (ifs.good()) {
      options.arguments({"--ros-args", "--params-file", params_file});
      RCLCPP_INFO(rclcpp::get_logger("armcontrol_node_main"),
                  "Default params file loaded: %s", params_file.c_str());
    } else {
      RCLCPP_WARN(rclcpp::get_logger("armcontrol_node_main"),
                  "Default params file not found, continue with declared defaults: %s",
                  params_file.c_str());
    }
  } catch (const std::exception &e) {
    RCLCPP_WARN(rclcpp::get_logger("armcontrol_node_main"),
                "Failed to resolve default params file, continue with declared defaults: %s",
                e.what());
  }

  rclcpp::spin(std::make_shared<ArmControlNode>(options));
  rclcpp::shutdown();
  return 0;
}