/*******************************************************************************
 * BSD 3-Clause License
 *
 * Copyright (c) 2021, Qiayuan Liao
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *******************************************************************************/

#include "armcontrol/socketcan.h"
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace robot_motor {
SocketCAN::~SocketCAN() { close(); }

bool SocketCAN::open(const std::string& interface,
                     std::function<void(const can_frame&)> handler, int priority) {
  if (interface.size() >= IFNAMSIZ) return false;
  fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
  if (fd_ < 0) return false;
  ifreq request{};
  std::memcpy(request.ifr_name, interface.c_str(), interface.size() + 1);
  sockaddr_can address{};
  address.can_family = AF_CAN;
  if (ioctl(fd_, SIOCGIFINDEX, &request) < 0) { close(); return false; }
  address.can_ifindex = request.ifr_ifindex;
  if (bind(fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) { close(); return false; }
  stop_ = false;
  receiver_ = std::thread([this, handler = std::move(handler)] {
    while (!stop_) {
      fd_set set;
      FD_ZERO(&set);
      FD_SET(fd_, &set);
      timeval timeout{0, 100000};
      if (select(fd_ + 1, &set, nullptr, nullptr, &timeout) <= 0) continue;
      can_frame frame{};
      if (read(fd_, &frame, sizeof(frame)) == sizeof(frame)) handler(frame);
    }
  });
  sched_param schedule{};
  schedule.sched_priority = priority;
  pthread_setschedparam(receiver_.native_handle(), SCHED_FIFO, &schedule);
  return true;
}

void SocketCAN::close() {
  stop_ = true;
  if (receiver_.joinable()) receiver_.join();
  if (fd_ >= 0) ::close(fd_);
  fd_ = -1;
}

void SocketCAN::sendFrame(uint32_t id, const uint8_t* data, uint8_t size) {
  if (fd_ < 0 || size > 8) throw std::runtime_error("Invalid CAN send");
  can_frame frame{};
  frame.can_id = id;
  frame.can_dlc = size;
  std::memcpy(frame.data, data, size);
  if (write(fd_, &frame, sizeof(frame)) != sizeof(frame)) throw std::runtime_error("CAN write failed");
}
}
