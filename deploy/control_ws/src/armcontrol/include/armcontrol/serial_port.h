#ifndef SERIAL_PORT_H
#define SERIAL_PORT_H

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <mutex>
#include <stdexcept>

class SerialPort
{
public:
  using SharedPtr = std::shared_ptr<SerialPort>;

  SerialPort(std::string port, speed_t baudrate, int timeout_ms = 2)
  {
    set_timeout(timeout_ms);
    Init(port, baudrate);
  }

  ~SerialPort()
  {
    close(fd_);
  }

  ssize_t send(const uint8_t* data, size_t len)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ssize_t ret = ::write(fd_, data, len);
    return ret;
  }

  ssize_t recv(uint8_t* data, size_t len)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    FD_ZERO(&rSet_);
    FD_SET(fd_, &rSet_);
    ssize_t recv_len = 0;

    auto timeout = timeout_;  // select mutates its timeout; preserve it for the next read.
    switch (select(fd_ + 1, &rSet_, NULL, NULL, &timeout))
    {
    case -1: // error
      break;
    case 0: // timeout
      break;
    default:
      recv_len = ::read(fd_, data, len);
      break;
    }

    return recv_len;
  }

  bool recv(uint8_t* data, uint8_t head, ssize_t len)
  {
    ssize_t recv_len = this->recv(recv_buf.data(), len);
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < recv_len; i++)
    {
      recv_queue.push(recv_buf[i]);
    }

    while (recv_queue.size() >= static_cast<size_t>(len))
    {
      if(recv_queue.front() != head)
      {
        recv_queue.pop();
        continue;
      }
      break;
    }

    if(recv_queue.size() < static_cast<size_t>(len)) return false;

    for(int i = 0; i < len; i++)
    {
      data[i] = recv_queue.front();
      recv_queue.pop();
    }
    return true;
  }

  void set_timeout(int timeout_ms)
  {
    timeout_.tv_sec = timeout_ms / 1000;
    timeout_.tv_usec = (timeout_ms % 1000) * 1000;
  }

private:
  void Init(std::string port, speed_t baudrate)
  {
    fd_ = open(port.c_str(), O_RDWR | O_NOCTTY);
    if (fd_ < 0)
    {
      printf("Open serial port %s failed\n", port.c_str());
      throw std::runtime_error("Unable to open USB-CAN serial port: " + port);
    }

    struct termios option;
    memset(&option, 0, sizeof(option));
    if (tcgetattr(fd_, &option) < 0) { close(fd_); throw std::runtime_error("tcgetattr failed"); }

    option.c_oflag = 0;
    option.c_lflag = 0;
    option.c_iflag = 0;

    cfsetispeed(&option, baudrate);
    cfsetospeed(&option, baudrate);

    option.c_cflag &= ~CSIZE;
    option.c_cflag |= CS8; // 8
    option.c_cflag &= ~PARENB; // no parity
    option.c_iflag &= ~INPCK; // no parity
    option.c_cflag &= ~CSTOPB; // 1 stop bit

    option.c_cc[VTIME] = 0;
    option.c_cc[VMIN] = 0;
    option.c_lflag |= CBAUDEX;

    tcflush(fd_, TCIFLUSH);
    if (tcsetattr(fd_, TCSANOW, &option) < 0) { close(fd_); throw std::runtime_error("tcsetattr failed"); }
  }

  int fd_;
	fd_set rSet_;
  timeval timeout_;

  std::queue<uint8_t> recv_queue;
  std::array<uint8_t, 1024> recv_buf;
  std::mutex mutex_;
};

#endif // SERIAL_PORT_H
