#!/usr/bin/env bash
# 一键启动实机部署全栈: IMU -> armcontrol -> RL 策略。
# 各开一个 gnome-terminal 窗口；RL 统一管理走路/下蹲/起身。
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
      echo "统一切换走路/下蹲/起身；保持当前姿态，不自动回站立。"
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
      python3 -c 'from rl_real_py.rl_real_common import RL_real, main; assert hasattr(RL_real, "multi") and not hasattr(RL_real, "_tick_prepare"), "请先重新编译 rl_real_py（统一三任务版）"; main()' \
      "${RL_ARGS[@]}" -p preflight_only:=true); then
  echo "错误: 预检失败，未启动电机驱动；请检查模型/配置，并重新编译 rl_real_py。" >&2
  exit 1
fi
python3 "$H1/sync_pd.py" --config "$CONFIG_FILE" --check-only
if $CHECK_ONLY; then exit 0; fi
command -v gnome-terminal >/dev/null || { echo "错误: 未找到 gnome-terminal"; exit 1; }

# 同步失败必须退出，不能用旧 PD 启动另一份策略。
python3 "$H1/sync_pd.py" --config "$CONFIG_FILE"
printf -v RL_ARGS_TEXT '%q ' "${RL_ARGS[@]}"

# ---- IMU (含 rviz) ----
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

echo "统一入口：1/LB+A 走路，2/LB+X 下蹲，3/LB+Y 起身；任务键表示已人工确认落地。"
echo "0/Start 先停步减速，Enter/再次 Start 确认接地收脚；P/B 中断锁存，R/Back 重新验收。"
echo "4/LB+Start：稳定保持后慢回准备站姿；执行动作中无效，须先扶稳/吊起。"
echo "只接管当前姿态，不自动插值回站立；P/B 不是断电急停。"
