#!/usr/bin/env bash
# 通用单关节辨识(吊起来): 左右同名关节 JL(左, idx0-5) 和 JL+6(右)。参数由环境变量传入。
# 全身按住 default 站姿, 只把被测关节移到 CENTER, 其余保持 default; /dog_joint_pos 200Hz(同 RL 路径), 左右同信号。
# 环境变量(带默认): JL CENTER AMP KPS(空格分隔可多值) KD NAME FREQS STEPS(正幅值逗号列)
H1="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # deploy 目录(脚本在 deploy/sysid/src/)
source /opt/ros/humble/setup.bash
source "$H1/control_ws/install/setup.bash"

JL=${JL:-0}
CENTER=${CENTER:--0.15}
AMP=${AMP:-0.4}
KD=${KD:-5}
NAME=${NAME:-joint}
FREQS=${FREQS:-0.5,1,1.5,2}
STEPS=${STEPS:-0.15,0.3}
read -r -a KPS <<< "${KPS:-200}"       # 空格分隔 -> 数组
STEP_HOLD=1.5
JR=$((JL+6))
DEFAULT_REAL=(-0.1 0 0 0.2 -0.1 0 -0.1 0 0 0.2 -0.1 0)
# 负幅值列(把 STEPS 每项取负)
STEPS_REV=$(echo "$STEPS" | tr ',' '\n' | sed 's/^/-/' | paste -sd,)
T=$(date +%Y%m%d_%H%M%S)
OUT="$H1/sysid/data/real/${NAME}_kp_sweep_$T/data"
mkdir -p "$OUT"
echo "会话: $OUT (左idx$JL/右idx$JR 中心$CENTER 幅$AMP kp=${KPS[*]} kd=$KD steps=$STEPS)"

start_arm() {
  ros2 run armcontrol arm_control_node > /tmp/arm_kpsweep.log 2>&1 &
  ARM_PID=$!
  for i in $(seq 1 30); do grep -q "Ready to Run" /tmp/arm_kpsweep.log && sleep 1 && return 0; sleep 0.5; done
  echo "!! armcontrol 15s 没 Ready"; return 1
}
stop_arm() { [ -n "${ARM_PID:-}" ] && kill "$ARM_PID" 2>/dev/null; pkill -f arm_control_node 2>/dev/null; sleep 1; }
trap 'echo 中断; stop_arm; exit 1' INT

excite() {  # $1=被测关节 idx
  local J=$1
  local -a b=("${DEFAULT_REAL[@]}"); b[$J]=$CENTER
  local BASE; BASE=$(IFS=,; echo "${b[*]}")
  echo "   base=$BASE"
  python3 "$H1/sysid/src/excite_record.py" --joint "$J" --mode sine \
     --freqs "$FREQS" --amp "$AMP" --cycles 6 --base="$BASE" --limit-frac 0.95 --out "$OUT"
  python3 "$H1/sysid/src/excite_record.py" --joint "$J" --mode step \
     --levels="$STEPS" --hold "$STEP_HOLD" --base="$BASE" --limit-frac 0.95 --out "$OUT" --suffix fwd
  python3 "$H1/sysid/src/excite_record.py" --joint "$J" --mode step \
     --levels="$STEPS_REV" --hold "$STEP_HOLD" --base="$BASE" --limit-frac 0.95 --out "$OUT" --suffix rev
}

for KP in "${KPS[@]}"; do
  echo "========== $NAME kp=$KP kd=$KD =========="
  stop_arm
  python3 "$H1/sysid/src/set_pd.py" --joint "$JL" --kp "$KP" --kd "$KD" --both
  start_arm || { echo "跳过 kp=$KP"; continue; }
  echo "--- 左 idx$JL ---"; excite "$JL"
  echo "--- 右 idx$JR ---"; excite "$JR"
done
echo "===== $NAME 完成, 恢复 PD ====="
stop_arm
python3 "$H1/sync_pd.py"
echo "数据在: $OUT"
