#!/usr/bin/env bash
# 一键启动实机部署全栈: 手柄 -> IMU -> armcontrol -> RL 策略。
# 各开一个 gnome-terminal 窗口；默认配置只启用 rough 走路。
# 注: 不在脚本里 build, 编译自行处理; 各节点只 source + 启动。
#
# 关闭全栈: 先支撑机器人，再执行 stop_real.sh。

set -e
export PATH="/usr/bin:$PATH"
ROS=/opt/ros/humble/setup.bash
H1="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # deploy 目录(脚本所在)
RL_NODE=rl_real_common
CONFIG_FILE="$H1/rl_real_py/configs/common.yaml"
CHECK_ONLY=false
while (($#)); do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || { echo "错误: --config 缺少路径" >&2; exit 2; }
      CONFIG_FILE="$2"; shift 2 ;;
    --check-only) CHECK_ONLY=true; shift ;;
    -h|--help)
      echo "用法: $0 [--config YAML] [--check-only]"
      echo "默认只启用 rough 走路；保持当前姿态，不自动回站立。"
      echo "--check-only: 仅检查配置/策略/PD，不启动驱动、不修改 PD 文件。"
      exit 0 ;;
    *) echo "错误: 未知参数 $1" >&2; exit 2 ;;
  esac
done
CONFIG_FILE="$(readlink -f -- "$CONFIG_FILE")"
RL_ARGS=(--ros-args -p "config_file:=$CONFIG_FILE")

# 必须先确认安装版支持无输出预检，避免旧 main 忽略参数后直接启动控制。
echo "[preflight] 检查安装版、所选模型与配置 ..."
if ! (source "$ROS"; source "$H1/rl_real_py/install/setup.bash";
      python3 -c 'from rl_real_py.rl_real_common import RL_real, main; assert getattr(RL_real, "joint_target_limits", None) is False, "请先重新编译 rl_real_py（无关节限位版）"; main()' \
      "${RL_ARGS[@]}" -p preflight_only:=true); then
  echo "错误: 预检失败，未启动电机驱动；请检查模型/配置，并重新编译 rl_real_py。" >&2
  exit 1
fi
python3 "$H1/sync_pd.py" --config "$CONFIG_FILE" --check-only
# Guard the offline driver mode before executing it: an old binary may ignore that parameter.
(source "$ROS"; source "$H1/control_ws/install/setup.bash";
 python3 -c 'from pathlib import Path; from ament_index_python.packages import get_package_prefix; p = Path(get_package_prefix("armcontrol")) / "lib/armcontrol/arm_control_node"; assert b"Motor configuration PASS; no ports, publishers or motor enable" in p.read_bytes(), "请先重新编译 armcontrol（自研电机配置版）"';
 ros2 run armcontrol arm_control_node --ros-args -p preflight_only:=true \
   -p "motor_config_file:=$H1/control_ws/src/armcontrol/config/motors.yaml")

if $CHECK_ONLY; then exit 0; fi
if pgrep -u "$(id -u)" -f '(^|/)(arm_control_node|rl_real_common)([[:space:]]|$)' >/dev/null; then
  echo "错误: 电机驱动或 RL 策略仍在运行；请先支撑机器人并执行 stop_real.sh，再启动，避免重复控制或混用旧 PD。" >&2
  exit 1
fi
command -v gnome-terminal >/dev/null || { echo "错误: 未找到 gnome-terminal"; exit 1; }
GAMEPAD_EXE=/opt/ros/humble/lib/joy/game_controller_node
[[ -x "$GAMEPAD_EXE" ]] || { echo "错误: 缺少手柄节点，请先执行 sudo apt install ros-humble-joy" >&2; exit 1; }
mapfile -t GAMEPAD_PIDS < <(pgrep -u "$(id -u)" -f "^$GAMEPAD_EXE([[:space:]]|$)" || true)
if ((${#GAMEPAD_PIDS[@]} > 1)); then
  echo "错误: 已有多个手柄节点；请先执行 stop_real.sh，再启动。" >&2
  exit 1
elif ((${#GAMEPAD_PIDS[@]} == 1)); then
  if ! grep -zFxq -e 'joy:=/gamepad' -e '/joy:=/gamepad' "/proc/${GAMEPAD_PIDS[0]}/cmdline"; then
    echo "错误: 已有手柄节点未映射到 /gamepad；请先执行 stop_real.sh，再启动。" >&2
    exit 1
  fi
fi

# 同步失败必须退出，不能用旧 PD 启动另一份策略。
python3 "$H1/sync_pd.py" --config "$CONFIG_FILE"
printf -v RL_ARGS_TEXT '%q ' "${RL_ARGS[@]}"

# ---- 手柄 ----
if ((${#GAMEPAD_PIDS[@]} == 0)); then
  echo "[Gamepad] 启动 ..."
  gnome-terminal --title="Gamepad" -- bash -c \
    "source $ROS; exec ros2 run joy game_controller_node --ros-args -p autorepeat_rate:=20.0 -r joy:=/gamepad"
else
  echo "[Gamepad] 复用已运行的 /gamepad 节点 (PID: ${GAMEPAD_PIDS[0]})"
fi

# ---- IMU ----
echo "[IMU] 启动 ..."
gnome-terminal --title="IMU" -- bash -c \
  "source $ROS; source $H1/imu_ws/install/setup.bash; \
   ros2 launch wit_ros2_imu rviz_and_imu.launch.py; \
   echo; echo '[IMU 已退出, 回车关闭]'; read"
sleep 3

# ---- armcontrol (电机驱动) ----
echo "[armcontrol] 启动 ..."
gnome-terminal --title="armcontrol" -- bash -c \
  "source $ROS; source $H1/control_ws/install/setup.bash; \
   ros2 run armcontrol arm_control_node; \
   echo; echo '[armcontrol 已退出, 回车关闭]'; read"
sleep 3

# ---- RL 策略 (真终端, 键盘控制) ----
echo "[RL] 启动 ($RL_NODE) ..."
gnome-terminal --title="RL policy ($RL_NODE)" -- bash -c \
  "source $ROS; source $H1/rl_real_py/install/setup.bash; \
   ros2 run rl_real_py $RL_NODE $RL_ARGS_TEXT; \
   echo; echo '[RL 已退出, 回车关闭]'; read"

echo "统一入口：默认仅 1/LB+A 启动 rough 走路；2、3 已禁用。"
echo "0/Start 立即停止走路策略并下发默认站姿；手柄 B 中断并保持最后目标，R/Back 解锁。"
echo "RB 切换左摇杆/方向键前后左右平移；切换后先回中/松开，右摇杆始终控制转向。"
echo "4/LB+Start：保持状态下慢回准备站姿；完成后按 1/LB+A 走路。执行动作中无效，须先扶稳/吊起。"
echo "只接管当前姿态；自动异常不锁停。手柄 B 不切断电机电源。"
