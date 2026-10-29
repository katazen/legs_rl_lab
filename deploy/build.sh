#!/usr/bin/env bash
set -e
export PATH="/usr/bin:$PATH"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
(cd "$HERE/control_ws" && colcon build --packages-select armcontrol --cmake-args -DARMCOMM=SERIAL -DBUILD_TESTING=OFF)
(cd "$HERE/imu_ws" && colcon build --packages-select wit_ros2_imu)
(cd "$HERE/rl_real_py" && colcon build --packages-select rl_real_py)
