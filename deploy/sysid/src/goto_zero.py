#!/usr/bin/env python3
"""把所有关节平滑 ramp 到指定姿态(默认全 0 位), 并保持, 便于观察实机。

前提: 先停掉 rl_real(否则两个节点抢发 /dog_joint_pos)。
     armcontrol 保持运行(它订阅 /dog_joint_pos 做 PD)。

用法(实机, 已 source ROS + control_ws):
  python3 sysid/src/goto_zero.py --ramp 4 --hold 10
  python3 sysid/src/goto_zero.py --target 0,0,0,0,0,0,0,0,0,0,0,0   # 自定义 12 维(实机序 L1..L6,R1..R6)
Ctrl-C 结束(结束后 armcontrol 会保持最后目标)。
"""
import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState


class GoTo(Node):
    def __init__(self, target, ramp, hold, rate):
        super().__init__("goto_zero")
        self.target = target
        self.ramp = ramp
        self.hold = hold
        self.rate = rate
        self.q0 = None
        self.pub = self.create_publisher(Float64MultiArray, "/dog_joint_pos", 1)
        self.create_subscription(JointState, "/left_joint_states", self._on_joint, 5)

    def _on_joint(self, msg):
        if self.q0 is None and len(msg.position) >= 12:
            self.q0 = np.array(msg.position[:12], np.float32)

    def _send(self, q):
        m = Float64MultiArray()
        m.data = [float(x) for x in q]
        self.pub.publish(m)

    def run(self):
        print("等待当前关节位置 ...")
        while rclpy.ok() and self.q0 is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        print(f"起点(实机序): {np.round(self.q0,3)}")
        print(f"目标        : {np.round(self.target,3)}")
        print(f"最大单关节位移: {np.abs(self.target-self.q0).max()*1000:.0f} mrad; {self.ramp}s 平滑移动 ...")
        dt = 1.0 / self.rate
        t0 = time.monotonic()
        # ramp: smoothstep
        while rclpy.ok():
            a = (time.monotonic() - t0) / max(self.ramp, 1e-3)
            if a >= 1.0:
                break
            s = a * a * (3.0 - 2.0 * a)
            self._send(self.q0 + s * (self.target - self.q0))
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(dt)
        print(f"已到目标, 保持 {self.hold}s (Ctrl-C 提前停) ...")
        t1 = time.monotonic()
        while rclpy.ok() and (time.monotonic() - t1) < self.hold:
            self._send(self.target)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(dt)
        print("结束(armcontrol 保持最后目标)。")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=str, default=",".join(["0"] * 12),
                   help="12 维目标(实机序 L1..L6,R1..R6), 逗号分隔; 缺省=全 0")
    p.add_argument("--ramp", type=float, default=4.0, help="平滑移动时长 s")
    p.add_argument("--hold", type=float, default=15.0, help="到位后保持 s")
    p.add_argument("--rate", type=float, default=200.0)
    args = p.parse_args()
    target = np.array([float(x) for x in args.target.split(",")], np.float32)
    assert len(target) == 12, "target 必须 12 维"

    rclpy.init()
    GoTo(target, args.ramp, args.hold, args.rate).run()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
