#!/usr/bin/env bash
# 一键关闭实机部署全栈 (RL 策略 -> armcontrol -> IMU)。
# ⚠️ 安全: armcontrol 退出可能让电机失力, 关闭前请先扶稳 / 挂好机器人!
#
# 用法: ./stop_real.sh

kill_pat() {  # $1=名称 $2=匹配模式; 先 SIGINT 优雅退出
  local pids
  pids=$(pgrep -f "$2")
  if [ -n "$pids" ]; then
    echo "  关闭 $1 (PID: $pids)"
    kill -INT $pids 2>/dev/null
  else
    echo "  $1 未在运行"
  fi
}

echo "=== 1. 先停 RL 策略 (停止下发 /dog_joint_pos 指令) ==="
kill_pat "RL 策略 common" "rl_real_py rl_real_common"
kill_pat "RL 策略 wan"    "rl_real_py rl_real_wan"
sleep 1

echo "=== 2. 停 armcontrol (电机驱动) ==="
kill_pat "armcontrol" "armcontrol arm_control_node"
sleep 1

echo "=== 3. 停 IMU / rviz ==="
kill_pat "IMU 驱动" "wit_ros2_imu"
kill_pat "rviz"     "rviz2"
sleep 1

# 兜底: 仍存活的强杀
echo "=== 4. 兜底强杀残留 ==="
for pat in "rl_real_py rl_real_common" "rl_real_py rl_real_wan" "armcontrol arm_control_node" "wit_ros2_imu" "rviz2"; do
  pkill -9 -f "$pat" 2>/dev/null && echo "  强杀: $pat"
done

echo "全部已关闭。(各窗口若停在 '回车关闭' 提示, 直接关掉即可)"
