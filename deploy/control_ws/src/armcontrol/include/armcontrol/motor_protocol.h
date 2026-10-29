#pragma once

#ifdef USE_CAN_COMMUNICATION
#include "socketcan.h"
#else
#include "serial_port.h"
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unistd.h>

namespace robot_motor {
struct Limits {
  float position, velocity, effort;
};

// Preserve the existing USB-CAN wire format; no motor-vendor assumption.
#pragma pack(push, 1)
struct ReceiveFrame {
  uint8_t header, command, flags;
  uint32_t can_id;
  uint8_t data[8], end;
};
struct SendFrame {
  uint8_t header[2] = {0x55, 0xAA};
  uint8_t length = 0x1e, command = 0x03;
  uint32_t count = 1, interval = 10;
  uint8_t id_type = 0;
  uint32_t can_id = 0;
  uint8_t frame_type = 0, length_data = 8, id_acc = 0, data_acc = 0;
  uint8_t data[8] = {}, crc = 0;
};
#pragma pack(pop)
static_assert(sizeof(ReceiveFrame) == 16 && sizeof(SendFrame) == 30);

class Motor {
public:
  void configure(uint32_t slave, uint32_t master, Limits limits) {
    slave_ = slave;
    master_ = master;
    limits_ = limits;
  }
  uint32_t GetSlaveId() const { return slave_; }
  uint32_t GetMasterId() const { return master_; }
  Limits limits() const { return limits_; }
  void receive_data(float q, float dq, float effort, uint8_t err, uint8_t mos, uint8_t rotor) {
    std::lock_guard<std::mutex> lock(mutex_);
    q_ = q; dq_ = dq; effort_ = effort; err_ = err;
    mos_ = mos; rotor_ = rotor;
    stamp_ = std::chrono::steady_clock::now();
  }
  void receive_error(uint8_t err) {
    std::lock_guard<std::mutex> lock(mutex_);
    err_ = err & 0x0f;
  }
  float Get_Position() const { std::lock_guard<std::mutex> lock(mutex_); return q_; }
  float Get_Velocity() const { std::lock_guard<std::mutex> lock(mutex_); return dq_; }
  // Driver feedback in protocol units; not calibrated joint-output torque.
  float Get_effort() const { std::lock_guard<std::mutex> lock(mutex_); return effort_; }
  uint8_t GetErrCode() const { std::lock_guard<std::mutex> lock(mutex_); return err_; }
  std::array<double, 4> feedback_health() const {
    std::lock_guard<std::mutex> lock(mutex_);
    const double age = stamp_.time_since_epoch().count() == 0 ? 1e9 :
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stamp_).count();
    return {age, double(err_), double(mos_), double(rotor_)};
  }
  static const char* error_code_to_string(uint8_t code) {
    switch (code) {
      case 0: return "失能";
      case 1: return "使能";
      case 8: return "超压";
      case 9: return "欠压";
      case 10: return "过电流";
      case 11: return "MOS 过温";
      case 12: return "电机线圈过温";
      case 13: return "通讯丢失";
      case 14: return "过载";
      default: return "未知错误";
    }
  }
private:
  uint32_t slave_ = 0, master_ = 0;
  Limits limits_{};
  mutable std::mutex mutex_;
  float q_ = 0, dq_ = 0, effort_ = 0;
  uint8_t err_ = 0, mos_ = 0, rotor_ = 0;
  std::chrono::steady_clock::time_point stamp_{};
};

class MotorControl {
public:
#ifdef USE_CAN_COMMUNICATION
  explicit MotorControl(std::string bus) : bus_(std::move(bus)) {}
  bool init() {
    socket_ = std::make_shared<SocketCAN>();
    return socket_->open(bus_, [this](const can_frame& frame) {
      if (frame.can_dlc == 8 && !(frame.can_id & (CAN_RTR_FLAG | CAN_ERR_FLAG)))
        receive_feedback(frame.can_id & CAN_SFF_MASK, frame.data);
    }, 95);
  }
#else
  explicit MotorControl(SerialPort::SharedPtr serial) : serial_(std::move(serial)) {}
#endif
  void addMotor(Motor* motor) {
    motors_.emplace(motor->GetSlaveId(), motor);
    if (motor->GetMasterId()) motors_.emplace(motor->GetMasterId(), motor);
  }
  void enable(const Motor& motor) {
    control_cmd(motor.GetSlaveId(), 0xFC);
    usleep(100000);
    receive();
  }
  void refresh_motor_status(const Motor& motor) {
    const uint32_t id = motor.GetSlaveId();
    send(0x7FF, {uint8_t(id), uint8_t(id >> 8), 0xCC, 0, 0, 0, 0, 0});
    receive();
  }
  static std::array<uint8_t, 8> encode_mit(Limits limits, float kp, float kd,
                                         float q, float dq, float effort) {
    if (!std::isfinite(kp) || !std::isfinite(kd) || kp < 0 || kp > 500 || kd < 0 || kd > 5)
      throw std::invalid_argument("MIT gains: Kp must be 0..500, Kd 0..5");
    const auto encode = [](float value, float lo, float hi, int bits) -> uint16_t {
      if (!std::isfinite(value) || value < lo || value > hi)
        throw std::invalid_argument("MIT value outside protocol range");
      return (value - lo) / (hi - lo) * ((1 << bits) - 1);
    };
    const auto p = encode(q, -limits.position, limits.position, 16);
    const auto v = encode(dq, -limits.velocity, limits.velocity, 12);
    const auto t = encode(effort, -limits.effort, limits.effort, 12);
    const auto k = encode(kp, 0, 500, 12), d = encode(kd, 0, 5, 12);
    return {uint8_t(p >> 8), uint8_t(p), uint8_t(v >> 4),
            uint8_t((v & 0xf) << 4 | k >> 8), uint8_t(k), uint8_t(d >> 4),
            uint8_t((d & 0xf) << 4 | t >> 8), uint8_t(t)};
  }
  void control_mit(const Motor& motor, float kp, float kd, float q, float dq, float effort) {
    send(motor.GetSlaveId(), encode_mit(motor.limits(), kp, kd, q, dq, effort));
    receive();
  }
  void receive_feedback(uint32_t id, const uint8_t* data) {
    const auto found = motors_.find(id ? id : data[0] & 0x0f);
    if (found == motors_.end()) return;
    auto& motor = *found->second;
    const auto limits = motor.limits();
    const auto decode = [](uint16_t value, float range, int bits) {
      return float(value) / ((1 << bits) - 1) * (2 * range) - range;
    };
    motor.receive_data(
      decode(uint16_t(data[1]) << 8 | data[2], limits.position, 16),
      decode(uint16_t(data[3]) << 4 | data[4] >> 4, limits.velocity, 12),
      decode(uint16_t(data[4] & 0xf) << 8 | data[5], limits.effort, 12),
      data[0] >> 4, data[6], data[7]);
  }
private:
  void receive() {
#ifndef USE_CAN_COMMUNICATION
    ReceiveFrame frame{};
    if (!serial_->recv(reinterpret_cast<uint8_t*>(&frame), 0xAA, sizeof(frame))) return;
    if (frame.end != 0x55 || (frame.flags & 0x3f) != 8) return;
    if (frame.command == 0x11) receive_feedback(frame.can_id, frame.data);
    else if (frame.command == 0xEE) {
      const auto found = motors_.find(frame.can_id);
      if (found != motors_.end()) found->second->receive_error(frame.data[0] >> 4);
    }
#endif
  }
  void control_cmd(uint32_t id, uint8_t cmd) {
    send(id, {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, cmd});
  }
  void send(uint32_t id, const std::array<uint8_t, 8>& data) {
#ifdef USE_CAN_COMMUNICATION
    socket_->sendFrame(id, data.data(), data.size());
#else
    SendFrame frame;
    frame.can_id = id;
    std::copy(data.begin(), data.end(), frame.data);
    if (serial_->send(reinterpret_cast<const uint8_t*>(&frame), sizeof(frame)) != sizeof(frame))
      throw std::runtime_error("Incomplete USB-CAN command write");
#endif
  }
  std::unordered_map<uint32_t, Motor*> motors_;
#ifdef USE_CAN_COMMUNICATION
  std::string bus_;
  std::shared_ptr<SocketCAN> socket_;
#else
  SerialPort::SharedPtr serial_;
#endif
};
}  // namespace robot_motor
