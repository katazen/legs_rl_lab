#!/usr/bin/env bash
# 膝关节 kp 扫描辨识(吊起来): 左右膝 L4(idx3)/R4(idx9), kp∈{100,150,200,250}, kd 固定。
#
# 关键(仿照强化学习的控制路径, 避免电机狂响):
#   - 全身按住 default 站姿(不是全归0!), 只把被测膝移到统一中心 CENTER, 其余关节保持 default。
#   - 只对被测膝叠加固定 正弦/阶跃, 用 /dog_joint_pos 200Hz 下发(与 rl_real 同路径), 看跟踪。
#   - 左右膝用【完全相同】的信号(同中心/同幅值/同频率), 不做差异化。
#
# 每个 kp: set_pd 设两膝 -> 重启 armcontrol -> 对 L4/R4 各跑 正弦+正反阶跃 -> 下一个 kp。
# 数据: sysid/data/real/knee_kp_sweep_<时间>/data/  (excite_record 从 armcontrol 读真实 kp/kd 写文件名)。
#
# 用法(吊稳机器人后, 普通终端):  bash sysid/src/knee_kp_sweep.sh
# 中途要停: Ctrl-C(会尝试收尾停 armcontrol)。
H1="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # deploy 目录(脚本在 deploy/sysid/src/)
source /opt/ros/humble/setup.bash
source "$H1/control_ws/install/setup.bash"

KD=5                       # kd 固定
KPS=(100 150 200 250)
SINE_FREQS=0.5,1,1.5,2     # 走路频带(gait0.6 基频~1.67Hz); 2Hz 为本次新增(第二次辨识)
SINE_AMP=0.5               # 膝走路命令半摆幅~0.5
STEP_HOLD=1.5
CENTER=0.55                # 膝走路工作点(左右统一, 不差异化)
# default 站姿(实机序 L1..L6,R1..R6): 被测膝会被覆盖成 CENTER, 其余关节保持此值
DEFAULT_REAL=(-0.1 0 0 0.2 -0.1 0 -0.1 0 0 0.2 -0.1 0)
T=$(date +%Y%m%d_%H%M%S)
OUT="$H1/sysid/data/real/knee_kp_sweep_$T/data"
mkdir -p "$OUT"
echo "会话目录: $OUT"

start_arm() {
  ros2 run armcontrol arm_control_node > /tmp/arm_kpsweep.log 2>&1 &
  ARM_PID=$!
  for i in $(seq 1 30); do
    grep -q "Ready to Run" /tmp/arm_kpsweep.log && sleep 1 && return 0
    sleep 0.5
  done
  echo "!! armcontrol 15s 内没 Ready, 看 /tmp/arm_kpsweep.log"; return 1
}
stop_arm() {
  [ -n "${ARM_PID:-}" ] && kill "$ARM_PID" 2>/dev/null
  pkill -f arm_control_node 2>/dev/null   # 脚本文件内 pkill -f 不会误杀本脚本(cmdline 是脚本路径)
  sleep 1
}
trap 'echo "中断, 收尾..."; stop_arm; exit 1' INT

excite() {  # $1=被测膝 idx; base=default 站姿, 该膝改到 CENTER, 其余关节保持 default
  local J=$1
  local -a b=("${DEFAULT_REAL[@]}"); b[$J]=$CENTER
  local BASE; BASE=$(IFS=,; echo "${b[*]}")
  echo "   base=$BASE"
  python3 "$H1/sysid/src/excite_record.py" --joint "$J" --mode sine \
     --freqs "$SINE_FREQS" --amp "$SINE_AMP" --cycles 6 --base="$BASE" --limit-frac 0.95 --out "$OUT"
  python3 "$H1/sysid/src/excite_record.py" --joint "$J" --mode step \
     --levels=0.15,0.3 --hold "$STEP_HOLD" --base="$BASE" --limit-frac 0.95 --out "$OUT" --suffix fwd
  python3 "$H1/sysid/src/excite_record.py" --joint "$J" --mode step \
     --levels=-0.15,-0.3 --hold "$STEP_HOLD" --base="$BASE" --limit-frac 0.95 --out "$OUT" --suffix rev
}

for KP in "${KPS[@]}"; do
  echo "========== kp=$KP kd=$KD =========="
  stop_arm
  python3 "$H1/sysid/src/set_pd.py" --joint 3 --kp "$KP" --kd "$KD" --both   # L4(3)+R4(9)
  start_arm || { echo "跳过 kp=$KP"; continue; }
  echo "--- 激励 L4(idx3) center=$CENTER ---"; excite 3
  echo "--- 激励 R4(idx9) center=$CENTER ---"; excite 9
done

echo "========== 全部完成, 恢复 PD 并停 armcontrol =========="
stop_arm
python3 "$H1/sync_pd.py"    # 把 arm yaml 恢复成 deploy.yaml 的 PD
echo "数据在: $OUT"
