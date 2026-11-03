#include "armcontrol/motor_protocol.h"
#include <pty.h>
#include <limits>
#include <iostream>

using namespace robot_motor;
void require(bool ok) { if (!ok) throw std::runtime_error("Motor protocol regression failed"); }
int main() {
  const Limits limits{12.5, 45, 30};
  // Golden packet from the original 16/12-bit mapping, including midpoint truncation.
  require(MotorControl::encode_mit(limits, 200, 5, 0, 0, 0) ==
          std::array<uint8_t, 8>{0x7f, 0xff, 0x7f, 0xf6, 0x66, 0xff, 0xf7, 0xff});
  for (const auto bad : {std::numeric_limits<float>::quiet_NaN(), 13.0f}) {
    bool rejected = false;
    try { MotorControl::encode_mit(limits, 200, 5, bad, 0, 0); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
  }
  int master, slave;
  char name[128];
  require(openpty(&master, &slave, name, nullptr, nullptr) == 0);
  close(slave);
  auto serial = std::make_shared<SerialPort>(name, B921600, 0);
  MotorControl control(serial);
  Motor motor;
  motor.configure(6, 22, limits);
  control.addMotor(&motor);
  // Deliberately fragment a feedback frame across receives; unknown IDs must be ignored.
  ReceiveFrame feedback{0xaa, 0x11, 8, 22, {0x16, 0xbf, 0xff, 0xbf, 0xfb, 0xff, 30, 31}, 0x55};
  require(::write(master, &feedback, 7) == 7);
  control.refresh_motor_status(motor);
  require(motor.feedback_health()[0] == 1e9);
  require(::write(master, reinterpret_cast<const uint8_t*>(&feedback) + 7, 9) == 9);
  usleep(1000);
  control.refresh_motor_status(motor);
  require(std::abs(motor.Get_Position() - 6.25) < .001);
  require(std::abs(motor.Get_Velocity() - 22.5) < .03);
  require(std::abs(motor.Get_effort() - 15) < .03);
  require(motor.GetErrCode() == 1 && motor.feedback_health()[2] == 30);
  const auto q = motor.Get_Position();
  control.receive_feedback(99, feedback.data);
  require(motor.Get_Position() == q);
  feedback.data[4] = 3;
  feedback.data[5] = 0xff;
  control.receive_feedback(0, feedback.data);  // Embedded slave ID fallback.
  require(motor.Get_effort() < 0);
  SendFrame sent{};
  require(::read(master, &sent, sizeof(sent)) == sizeof(sent));
  require(sent.can_id == 0x7ff && sent.data[0] == 6 && sent.data[2] == 0xcc);
  close(master);
  std::cout << "Motor packet encode/decode and fragmented virtual serial PASS\n";
}
