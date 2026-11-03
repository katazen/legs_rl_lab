#pragma once
#include <linux/can.h>
#include <atomic>
#include <functional>
#include <string>
#include <thread>
namespace robot_motor {
class SocketCAN {
public:
  ~SocketCAN();
  bool open(const std::string& interface, std::function<void(const can_frame&)> handler, int priority);
  void close();
  void sendFrame(uint32_t id, const uint8_t* data, uint8_t size);
private:
  int fd_ = -1;
  std::atomic<bool> stop_{false};
  std::thread receiver_;
};
}
