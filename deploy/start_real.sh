#!/usr/bin/env bash
# 一键启动实机部署全栈: IMU -> armcontrol -> RL 策略。
# 各开一个 gnome-terminal 窗口(RL 窗口有真终端, 键盘 W/S/A/D/Q/E/P/R 可用)。
# 注: 不在脚本里 build, 编译自行处理; 各节点只 source + 启动。
#
# 关闭: 直接关掉三个窗口, 或在任一窗口 Ctrl-C。

set -e
ROS=/opt/ros/humble/setup.bash
H1="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # deploy 目录(脚本所在)
RL_NODE=rl_real_common   # 要测 wan 就改成 rl_real_wan

command -v gnome-terminal >/dev/null || { echo "错误: 未找到 gnome-terminal"; exit 1; }

# ---- IMU (含 rviz) ----
echo "[IMU] 启动 ..."
gnome-terminal --title="IMU" -- bash -c \
  "source $ROS; source $H1/imu_ws/install/setup.bash; \
   ros2 launch wit_ros2_imu rviz_and_imu.launch.py; \
   echo; echo '[IMU 已退出, 回车关闭]'; read"
sleep 3

# ---- armcontrol (电机驱动) ----
echo "[armcontrol] 启动 ..."
echo "[PD] 从 deploy.yaml 同步 kp/kd -> armcontrol ..."
python3 $H1/sync_pd.py || echo "[PD] 同步失败, 沿用旧 arm yaml"
gnome-terminal --title="armcontrol" -- bash -c \
  "source $ROS; source $H1/control_ws/install/setup.bash; \
   ros2 run armcontrol arm_control_node; \
   echo; echo '[armcontrol 已退出, 回车关闭]'; read"
sleep 3

# ---- RL 策略 (真终端, 键盘控制) ----
echo "[RL] 启动 ($RL_NODE) ..."
gnome-terminal --title="RL policy ($RL_NODE)" -- bash -c \
  "source $ROS; source $H1/rl_real_py/install/setup.bash; \
   ros2 run rl_real_py $RL_NODE; \
   echo; echo '[RL 已退出, 回车关闭]'; read"

echo "全部已启动。流程: 自动缓慢进准备姿态 -> 站立保持 -> 在 RL 窗口按 P 开始行走。"
