#!/usr/bin/env python3
"""静态零位检查: 机器人站在平地、保持 default 站姿不动时, 采集关节读数 + IMU,
判断机体是否倾斜、往哪倾, 反推可能有标0偏移的关节/侧。

前提(很重要):
  - 机器人【站在平地、用自身 PD 保持 default 站姿】, 双脚承重, 撒手不扶;
    (吊着测无效——身体朝向由吊点决定, 反映不出腿部零偏)
  - 先起 armcontrol + imu, 再起 rl_real 让它缓入到 default 并 hold(别按行走),
    撒手稳定后再跑本脚本。

用法(在实机、已 source ROS 与 control_ws):
  python3 sysid/src/static_zero_check.py --secs 5
输出: CSV 存到 sysid/data/real/static_zero_<时间>.csv, 并打印倾斜/关节摘要。
"""
import argparse
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu

JN = ["L1-HipPitch", "L2-HipRoll", "L3-HipYaw", "L4-Knee", "L5-AnkPitch", "L6-AnkRoll",
      "R1-HipPitch", "R2-HipRoll", "R3-HipYaw", "R4-Knee", "R5-AnkPitch", "R6-AnkRoll"]
# default 站姿(实机序 L1..L6,R1..R6), 与 deploy.yaml 对应
DEFAULT_REAL = np.array([-0.1, 0, 0, 0.2, -0.1, 0, -0.1, 0, 0, 0.2, -0.1, 0], np.float32)


def gravity_from_quat(q):
    """投影重力(机体系), q = [w, x, y, z] (与部署一致)。"""
    w, x, y, z = q
    return np.array([2 * (-z * x + w * y),
                     -2 * (z * y + w * x),
                     1 - 2 * (w * w + z * z)], dtype=np.float32)


class Collector(Node):
    def __init__(self, secs):
        super().__init__("static_zero_check")
        self.secs = secs
        self.q_rows, self.imu_rows = [], []
        self._q = None
        self._quat = None
        self._w = None
        self.create_subscription(JointState, "/left_joint_states", self._on_joint, 5)
        self.create_subscription(Imu, "/imu", self._on_imu, 5)
        self.t0 = None

    def _on_joint(self, msg):
        if len(msg.position) >= 12:
            self._q = np.array(msg.position[:12], np.float32)

    def _on_imu(self, msg):
        self._quat = np.array([msg.orientation.w, msg.orientation.x,
                               msg.orientation.y, msg.orientation.z], np.float32)
        self._w = np.array([msg.angular_velocity.x, msg.angular_velocity.y,
                            msg.angular_velocity.z], np.float32)

    def spin(self):
        # 等两路数据都到齐
        print("等待 /left_joint_states 与 /imu ...")
        while rclpy.ok() and (self._q is None or self._quat is None):
            rclpy.spin_once(self, timeout_sec=0.1)
        print(f"数据已到, 开始采集 {self.secs}s (保持不动)...")
        self.t0 = time.monotonic()
        while rclpy.ok() and (time.monotonic() - self.t0) < self.secs:
            rclpy.spin_once(self, timeout_sec=0.02)
            t = time.monotonic() - self.t0
            if self._q is not None:
                self.q_rows.append((t, *self._q))
            if self._quat is not None:
                g = gravity_from_quat(self._quat)
                self.imu_rows.append((t, *self._quat, *g, *self._w))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--secs", type=float, default=5.0)
    p.add_argument("--out", type=str, default="sysid/data/real")
    args = p.parse_args()

    rclpy.init()
    c = Collector(args.secs)
    c.spin()
    rclpy.shutdown()

    q = np.array(c.q_rows, np.float32)      # t, q0..11
    im = np.array(c.imu_rows, np.float32)   # t, qw,qx,qy,qz, gx,gy,gz, wx,wy,wz
    if len(q) < 5 or len(im) < 5:
        print("!! 采集太少, 检查话题是否在发布")
        return

    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(args.out, f"static_zero_{stamp}.csv")
    hdr = "t," + ",".join(f"q{i}" for i in range(12))
    np.savetxt(path, q, delimiter=",", header=hdr, comments="")
    path_imu = os.path.join(args.out, f"static_zero_{stamp}_imu.csv")
    np.savetxt(path_imu, im, delimiter=",",
               header="t,qw,qx,qy,qz,gx,gy,gz,wx,wy,wz", comments="")

    # ---------- 即时摘要 ----------
    qm = q[:, 1:].mean(0)
    g = im[:, 5:8].mean(0)   # gx,gy,gz
    gx, gy, gz = g
    lat = np.degrees(np.arctan2(gy, -gz))    # 侧向倾角(gy)
    fwd = np.degrees(np.arctan2(gx, -gz))    # 前后倾角(gx)
    tilt = np.degrees(np.arccos(np.clip(-gz, -1, 1)))
    print("\n" + "=" * 60)
    print(f"采样: 关节 {len(q)} 帧, IMU {len(im)} 帧")
    print(f"投影重力 gx={gx:+.3f} gy={gy:+.3f} gz={gz:+.3f}  (直立: gx=gy=0, gz=-1)")
    print(f"总倾角 {tilt:.1f}°   侧倾(gy) {lat:+.1f}°   前后倾(gx) {fwd:+.1f}°")
    print(f"  侧倾>0 表示往一侧, <0 另一侧; 前后倾同理(具体左右/前后取决于 IMU 安装, 结合关节看)")
    print("-" * 60)
    print(f"{'joint':<14}{'mean q':>9}{'default':>9}{'q-def(mrad)':>13}")
    for i in range(12):
        print(f"{JN[i]:<14}{qm[i]:>9.3f}{DEFAULT_REAL[i]:>9.3f}{(qm[i]-DEFAULT_REAL[i])*1000:>13.0f}")
    print("-" * 60)
    print("左右对比 (mean q, L vs R):")
    for i in range(6):
        print(f"  {JN[i][3:]:<10} L={qm[i]:+.3f}  R={qm[i+6]:+.3f}  L-R={(qm[i]-qm[i+6])*1000:+.0f} mrad")
    print("=" * 60)
    print(f"\nCSV: {path}\n     {path_imu}")


if __name__ == "__main__":
    main()
