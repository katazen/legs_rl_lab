#include "armcontrol/motor_protocol.h"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <yaml-cpp/yaml.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>

namespace {
const std::string& diag_dir() {
  static const std::string path = std::string(std::getenv("HOME") ? std::getenv("HOME") : ".") + "/dual_leg_diag/";
  return path;
}
const std::string& diag_run_tag() {
  static const std::string tag = [] {
    const auto now = std::time(nullptr);
    char value[32];
    std::strftime(value, sizeof(value), "%Y%m%d_%H%M%S", std::localtime(&now));
    return std::string(value);
  }();
  return tag;
}
}

class ArmControlNode : public rclcpp::Node {
public:
  explicit ArmControlNode(const rclcpp::NodeOptions& options) : Node("armcontrol_node", options) {
    config_kps_ = declare_parameter<std::vector<double>>("kps", std::vector<double>(12, 8.0));
    config_kds_ = declare_parameter<std::vector<double>>("kds", std::vector<double>(12, 0.2));
    config_max_vel_ = declare_parameter<double>("max_vel", 0.0);
    enable_dual_leg_diag_ = declare_parameter<bool>("enable_dual_leg_diag", false);
    if (config_kps_.size() != 12 || config_kds_.size() != 12 || config_max_vel_ != 0)
      throw std::invalid_argument("需要 12 组 PD，max_vel 必须为 0");
    for (int i = 0; i < 12; ++i) {
      if (!std::isfinite(config_kps_[i]) || !std::isfinite(config_kds_[i]) ||
          config_kps_[i] < 0 || config_kps_[i] > 500 || config_kds_[i] < 0 || config_kds_[i] > 5)
        throw std::invalid_argument("PD 超出协议范围");
    }
    const auto file = declare_parameter<std::string>("motor_config_file",
      ament_index_cpp::get_package_share_directory("armcontrol") + "/config/motors.yaml");
    const auto cfg = YAML::LoadFile(file);
    if (cfg["models"].size() != 2 || !cfg["models"]["large"] || !cfg["models"]["small"] || cfg["joints"].size() != 12)
      throw std::invalid_argument("需要 large/small 两款自研电机和 12 个关节");
    left_arm_motors_ = std::make_unique<robot_motor::Motor[]>(6);
    right_arm_motors_ = std::make_unique<robot_motor::Motor[]>(6);
    const std::array<int, 6> ids{1, 2, 3, 4, 6, 7};
    for (int i = 0; i < 12; ++i) {
      const auto j = cfg["joints"][i];
      const auto name = std::string(i < 6 ? "L" : "R") + std::to_string(i % 6 + 1);
      const auto type = j["model"].as<std::string>();
      const auto limits = cfg["models"][type];
      direction_[i] = j["direction"].as<int>();
      if (j["name"].as<std::string>() != name || type != (i % 6 == 5 ? "small" : "large") ||
          std::abs(direction_[i]) != 1 || j["slave_id"].as<int>() != ids[i % 6] ||
          j["master_id"].as<int>() != ids[i % 6] + 16)
        throw std::invalid_argument("关节顺序/型号/CAN ID/方向无效");
      robot_motor::Limits ranges{limits["position_range"].as<float>(),
        limits["velocity_range"].as<float>(), limits["effort_range"].as<float>()};
      if (!std::isfinite(ranges.position) || ranges.position <= 0 ||
          !std::isfinite(ranges.velocity) || ranges.velocity <= 0 ||
          !std::isfinite(ranges.effort) || ranges.effort <= 0)
        throw std::invalid_argument("协议范围必须为有限正数");
      auto& motor = i < 6 ? left_arm_motors_[i] : right_arm_motors_[i - 6];
      motor.configure(ids[i % 6], ids[i % 6] + 16, ranges);
    }
    if (declare_parameter<bool>("preflight_only", false)) {
      RCLCPP_INFO(get_logger(), "Motor configuration PASS; no ports, publishers or motor enable");
      return;
    }
#ifdef USE_CAN_COMMUNICATION
    left_arm_mc_ = std::make_unique<robot_motor::MotorControl>(declare_parameter<std::string>("left_arm_can_name", "can2"));
    right_arm_mc_ = std::make_unique<robot_motor::MotorControl>(declare_parameter<std::string>("right_arm_can_name", "can1"));
#else
    left_arm_mc_ = std::make_unique<robot_motor::MotorControl>(std::make_shared<SerialPort>(
      declare_parameter<std::string>("left_arm_device", "/dev/ttyCANL"), B921600, 0));
    right_arm_mc_ = std::make_unique<robot_motor::MotorControl>(std::make_shared<SerialPort>(
      declare_parameter<std::string>("right_arm_device", "/dev/ttyCANR"), B921600, 0));
#endif
    for (int i = 0; i < 6; ++i) {
      left_arm_mc_->addMotor(&left_arm_motors_[i]);
      right_arm_mc_->addMotor(&right_arm_motors_[i]);
    }
#ifdef USE_CAN_COMMUNICATION
    if (!left_arm_mc_->init() || !right_arm_mc_->init()) throw std::runtime_error("CAN initialization failed");
#endif
    motors_err_pub_ = create_publisher<std_msgs::msg::String>("/motor_warn", 10);
    motor_health_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/sysid_motor_health", 1);
    joint_state_pub_ = create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);
    left_joint_state_pub_ = create_publisher<sensor_msgs::msg::JointState>("/left_joint_states", 10);
    left_joint_max_state_pub_ = create_publisher<sensor_msgs::msg::JointState>("/left_joint_max_states", 10);
    dog_joint_pos_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>("dog_joint_pos", rclcpp::QoS(1),
      std::bind(&ArmControlNode::joint_pos_dual_leg_callback, this, std::placeholders::_1));
    last_left_err_.fill(-1);
    last_right_err_.fill(-1);
    for (int i = 0; i < 6; ++i) {
      left_arm_mc_->enable(left_arm_motors_[i]);
      right_arm_mc_->enable(right_arm_motors_[i]);
    }
    usleep(100000);
    loop_thread_ = std::thread(&ArmControlNode::loop, this);
    RCLCPP_INFO(get_logger(), "Self-developed dual-leg motors enabled; ready");
  }
  ~ArmControlNode() override {
    stop_ = true;
    if (loop_thread_.joinable()) loop_thread_.join();
  }
private:
  void loop() {
    // Match the shared hardware order in motors.yaml.
    std::vector<std::string> names;
    for (int i = 0; i < 12; ++i) names.push_back(std::string(i < 6 ? "L" : "R") + std::to_string(i % 6 + 1));
    while (rclcpp::ok() && !stop_) {
      const auto wait_start = std::chrono::steady_clock::now();
      std::unique_lock<std::mutex> lock(writeLocker_);
      const auto refresh_start = std::chrono::steady_clock::now();
      const double loop_lock_wait_us = std::chrono::duration<double, std::micro>(refresh_start - wait_start).count();
      for (int i = 0; i < 6; ++i) {
        right_arm_mc_->refresh_motor_status(right_arm_motors_[i]);
        left_arm_mc_->refresh_motor_status(left_arm_motors_[i]);
      }
      const double refresh_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - refresh_start).count();
      publish_dual_leg_joint_states(names);
      std::vector<std::pair<int, int>> left_errors, right_errors;
      for (int i = 0; i < 6; ++i) {
        left_errors.emplace_back(i + 1, left_arm_motors_[i].GetErrCode());
        right_errors.emplace_back(i + 1, right_arm_motors_[i].GetErrCode());
      }
      publishMotorErrors("left", left_errors);
      publishMotorErrors("Right", right_errors);
      if (enable_dual_leg_diag_) {
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
        static long state_rows = 0;  // 诊断CSV总行数上限, 满即停止写入(防长跑写满磁盘; 200Hz下~16min, ~90MB封顶)
        if (state_log.is_open() && state_rows < 200000) {
          state_log.precision(6);
          state_log << this->now().nanoseconds()
                    << "," << loop_dt_ms << "," << refresh_us << "," << loop_lock_wait_us;
          for (int j = 0; j < 12; ++j) {
            robot_motor::Motor & m = (j < 6) ? left_arm_motors_[j] : right_arm_motors_[j - 6];
            state_log << "," << m.Get_Position()
                      << "," << m.Get_Velocity()
                      << "," << m.Get_effort()
                      << "," << static_cast<unsigned>(m.GetErrCode());
          }
          state_log << "\n";
          if (++state_rows == 200000)
            RCLCPP_WARN(this->get_logger(), "dual_leg state 诊断CSV达20万行上限, 停止写入(防写满磁盘)");
        }
      }
      lock.unlock();
      usleep(4700);  // Preserve the existing feedback polling interval.
    }
  }
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

    if (!std::all_of(msg->data.begin(), msg->data.end(), [](double q) { return std::isfinite(q); })) {
      RCLCPP_WARN(this->get_logger(), "拒绝非有限电机位置目标");
      return;
    }
    for (int i = 0; i < 12; ++i) {
      const auto& motor = i < 6 ? left_arm_motors_[i] : right_arm_motors_[i - 6];
      if (std::abs(msg->data[i]) > motor.limits().position) {
        RCLCPP_WARN(get_logger(), "拒绝超出协议范围的关节目标 %d", i);
        return;
      }
    }
    for (int i = 0; i < 12; ++i) msg->data[i] *= direction_[i];
    const std::vector<double> vel_cmd(12, 0.0);
    double lock_wait_us = 0.0;
    double left_send_us = 0.0;
    double right_send_us = 0.0;
    try {
      const auto wait_lock_start = steady_clock::now();
      std::unique_lock<std::mutex> lock(writeLocker_);
      const auto lock_acquired = steady_clock::now();
      lock_wait_us = std::chrono::duration<double, std::micro>(lock_acquired - wait_lock_start).count();


      for (int i = 0; i < 6; ++i) {
        const double dq_l = std::clamp(vel_cmd[i],     -config_max_vel_, config_max_vel_);
        const double dq_r = std::clamp(vel_cmd[i + 6], -config_max_vel_, config_max_vel_);
        const auto t0 = steady_clock::now();
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

    if (enable_dual_leg_diag_) {
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
      static long cmd_rows = 0;  // 诊断CSV总行数上限, 满即停止写入(防长跑写满磁盘; 250Hz下~13min, ~90MB封顶)
      if (cmd_log.is_open() && cmd_rows < 200000) {
        cmd_log.precision(6);
        cmd_log << this->now().nanoseconds()
                << "," << gap_ms << "," << lock_wait_us
                << "," << left_send_us << "," << right_send_us;
        for (int i = 0; i < 12; ++i) cmd_log << "," << msg->data[i];
        for (int i = 0; i < 12; ++i) cmd_log << "," << std::clamp(vel_cmd[i], -config_max_vel_, config_max_vel_);
        cmd_log << "\n";
        if (++cmd_rows == 200000)
          RCLCPP_WARN(this->get_logger(), "dual_leg cmd 诊断CSV达20万行上限, 停止写入(防写满磁盘)");
      }
    }

    if (enable_dual_leg_diag_) {
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
          "dual_leg cmd L[0..5]=%.3f,%.3f,%.3f,%.3f,%.3f,%.3f R[0..5]=%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
          msg->data[0], msg->data[1], msg->data[2], msg->data[3], msg->data[4], msg->data[5],
          msg->data[6], msg->data[7], msg->data[8], msg->data[9], msg->data[10], msg->data[11]);
    }
  }
  // Caller holds writeLocker_ throughout refresh and publication.
  void publish_dual_leg_joint_states(const std::vector<std::string> & joint_names)
  {
    std_msgs::msg::Float64MultiArray health;
    for (auto *motors : {left_arm_motors_.get(), right_arm_motors_.get()}) {
      for (int i = 0; i < 6; ++i) {
        const auto h = motors[i].feedback_health();
        health.data.insert(health.data.end(), h.begin(), h.end());
      }
    }
    motor_health_pub_->publish(health);
    auto joint_state_msg = sensor_msgs::msg::JointState();
    auto joint_max_state_msg = sensor_msgs::msg::JointState();
    joint_state_msg.header.frame_id = "dual_leg";
    joint_state_msg.header.stamp = this->now();
    joint_state_msg.name = joint_names;
    joint_state_msg.position.resize(12);
    joint_state_msg.velocity.reserve(12);
    joint_state_msg.effort.reserve(12);

    for (int i = 0; i < 6; ++i) {
      joint_state_msg.position[i]     = direction_[i]     * left_arm_motors_[i].Get_Position();
      joint_state_msg.position[i + 6] = direction_[i + 6] * right_arm_motors_[i].Get_Position();
    }
    joint_max_state_msg = joint_state_msg;
    joint_max_state_msg.effort.clear();

    for (int i = 0; i < 6; ++i) {
      joint_state_msg.velocity.push_back(direction_[i] * left_arm_motors_[i].Get_Velocity());
      const float tau_l = left_arm_motors_[i].Get_effort();   // 电机帧裸值
      joint_state_msg.effort.push_back(direction_[i] * tau_l);
      {
        std::lock_guard<std::mutex> lock(max_tau_mutex_);
        if (std::fabs(tau_l) > std::fabs(left_max_tau_[i])) {
          left_max_tau_[i] = tau_l;
        }
        joint_max_state_msg.effort.push_back(direction_[i] * left_max_tau_[i]);
      }
    }
    for (int i = 0; i < 6; ++i) {
      joint_state_msg.velocity.push_back(direction_[i + 6] * right_arm_motors_[i].Get_Velocity());
      const float tau_r = right_arm_motors_[i].Get_effort();  // 电机帧裸值
      joint_state_msg.effort.push_back(direction_[i + 6] * tau_r);
      {
        std::lock_guard<std::mutex> lock(max_tau_mutex_);
        if (std::fabs(tau_r) > std::fabs(right_max_tau_[i])) {
          right_max_tau_[i] = tau_r;
        }
        joint_max_state_msg.effort.push_back(direction_[i + 6] * right_max_tau_[i]);
      }
    }

    joint_state_pub_->publish(joint_state_msg);
    left_joint_state_pub_->publish(joint_state_msg);
    left_joint_max_state_pub_->publish(joint_max_state_msg);
  }
  void publishMotorErrors(const std::string& arm_name,
                          const std::vector<std::pair<int,int>>& id_errs) {
    if (!motors_err_pub_) return;

    const bool is_left = (arm_name == "left" || arm_name == "Left");
    std::vector<std::pair<int,int>> events;

    bool now_enabled_all = true;
    bool now_disabled_all = true;

    bool& last_enabled_all = is_left ? last_left_enabled_all_ : last_right_enabled_all_;
    bool& last_disabled_all = is_left ? last_left_disabled_all_ : last_right_disabled_all_;

    for (const auto& p : id_errs) {
      int motor_id = p.first;
      int now_code = p.second;
      int idx = motor_id - 1;
      if (idx < 0 || idx >= 6) continue;

      int& prev_code = is_left ? last_left_err_[idx] : last_right_err_[idx];

      if (prev_code == -1) {
        prev_code = now_code;
        if (now_code != 0) now_disabled_all = false;
        if (now_code != 1) now_enabled_all = false;
        continue;
      }

      bool prev_fault = (prev_code > 1);
      bool now_fault  = (now_code > 1);

      if (now_code != 0) now_disabled_all = false;
      if (now_code != 1) now_enabled_all = false;

      if ((!prev_fault && now_fault) ||                      // 新故障 0/1 -> >1
          (prev_fault && now_fault && now_code != prev_code) || // 故障码变化 >1 -> >1
          (prev_fault && !now_fault)) {                      // 故障清除 >1 -> 0/1
        events.emplace_back(motor_id, now_code);             // 清除时携带实际0或1
      }

      prev_code = now_code;
    }


    if (now_disabled_all && !last_disabled_all) {
      std::ostringstream oss;
      oss << "{\"master_or_slave\":\"slave\",\"arm\":\"" << (is_left ? "left" : "Right") << "\",\"errors\":[";
      for (size_t i = 0; i < id_errs.size(); ++i) {
        oss << "{\"id\":" << id_errs[i].first << ",\"err\":\"" << robot_motor::Motor::error_code_to_string(id_errs[i].second) << "\"}";
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

    if (now_enabled_all && !last_enabled_all) {
      std::ostringstream oss;
      oss << "{\"master_or_slave\":\"slave\",\"arm\":\"" << (is_left ? "left" : "Right") << "\",\"errors\":[";
      for (size_t i = 0; i < id_errs.size(); ++i) {
        oss << "{\"id\":" << id_errs[i].first << ",\"err\":\"" << robot_motor::Motor::error_code_to_string(id_errs[i].second) << "\"}";
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

    last_disabled_all = now_disabled_all;
    last_enabled_all = now_enabled_all;

    if (events.empty()) return;

    std::ostringstream oss;
    oss << "{\"master_or_slave\":\"slave\",\"arm\":\"" << (is_left ? "left" : "Right") << "\",\"errors\":[";
    for (size_t i = 0; i < events.size(); ++i) {
      if((events[i].second >1 && events[i].second < 7) || events[i].second > 14) events[i].second = 99; // 将电机错误码 2-6 和大于14的码统一为99，方便前端显示
      oss << "{\"id\":" << events[i].first << ",\"err\":\"" << robot_motor::Motor::error_code_to_string(events[i].second) << "\"}";
      if (i + 1 < events.size()) oss << ",";
    }
    oss << "]}";


    std_msgs::msg::String msg;
    msg.data = oss.str();
    motors_err_pub_->publish(msg);
  }

  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr dog_joint_pos_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr motor_health_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_, left_joint_state_pub_, left_joint_max_state_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr motors_err_pub_;
  // Motors outlive transport callbacks: controllers are destroyed first.
  std::unique_ptr<robot_motor::Motor[]> left_arm_motors_, right_arm_motors_;
  std::unique_ptr<robot_motor::MotorControl> left_arm_mc_, right_arm_mc_;
  std::array<double, 12> direction_{};
  std::vector<double> config_kps_, config_kds_;
  double config_max_vel_ = 0;
  bool enable_dual_leg_diag_ = false;
  std::array<int, 6> last_left_err_{}, last_right_err_{};
  bool last_left_enabled_all_ = false, last_left_disabled_all_ = false;
  bool last_right_enabled_all_ = false, last_right_disabled_all_ = false;
  std::mutex writeLocker_, max_tau_mutex_;
  std::vector<float> left_max_tau_ = std::vector<float>(6, 0), right_max_tau_ = std::vector<float>(6, 0);
  std::atomic<bool> stop_{false};
  std::thread loop_thread_;
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::NodeOptions options;
    bool explicit_params = false;
    for (int i = 1; i < argc; ++i) if (std::strcmp(argv[i], "--params-file") == 0) explicit_params = true;
    if (!explicit_params) {
      const auto file = ament_index_cpp::get_package_share_directory("armcontrol") + "/config/arm_control_node.yaml";
      options.arguments({"--ros-args", "--params-file", file});
    }
    auto node = std::make_shared<ArmControlNode>(options);
    if (!node->get_parameter("preflight_only").as_bool()) rclcpp::spin(node);
    node.reset();
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception& error) {
    RCLCPP_ERROR(rclcpp::get_logger("armcontrol_node"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
}
