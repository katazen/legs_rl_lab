#!/usr/bin/env bash
# 安装 ROS2 Humble (Ubuntu 22.04 / Jammy)
# 用法:  sudo bash install_ros2_humble.sh
set -euo pipefail

echo "==> [0/5] 环境检查"
. /etc/os-release
if [ "${VERSION_CODENAME:-}" != "jammy" ]; then
  echo "!! 当前系统 codename=${VERSION_CODENAME:-未知},ROS2 Humble 需要 jammy(22.04)。中止。"
  exit 1
fi
if [ "$(id -u)" -ne 0 ]; then
  echo "!! 请用 sudo 运行:  sudo bash $0"
  exit 1
fi

echo "==> [1/5] locale + 前置依赖"
apt-get update
apt-get install -y locales curl gnupg lsb-release software-properties-common
locale-gen en_US en_US.UTF-8
update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
add-apt-repository universe -y

echo "==> [2/5] ROS2 apt key + 源"
install -d -m 0755 /usr/share/keyrings
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu jammy main" \
  > /etc/apt/sources.list.d/ros2.list

echo "==> [3/5] apt update"
apt-get update

echo "==> [4/5] 安装 ros-humble-desktop + 构建工具(较大,耐心等)"
apt-get install -y ros-humble-desktop ros-dev-tools python3-colcon-common-extensions

echo "==> [5/5] 把 source 追加到调用用户的 ~/.bashrc"
TARGET_USER="${SUDO_USER:-$USER}"
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
BASHRC="$TARGET_HOME/.bashrc"
LINE="source /opt/ros/humble/setup.bash"
if [ -f "$BASHRC" ] && ! grep -qxF "$LINE" "$BASHRC"; then
  echo "$LINE" >> "$BASHRC"
  echo "   已追加到 $BASHRC"
else
  echo "   跳过(已存在或无 .bashrc)"
fi

echo ""
echo "==> 完成。验证:"
echo "    source /opt/ros/humble/setup.bash"
echo "    echo \$ROS_DISTRO        # 应输出 humble"
echo "    ros2 doctor              # 检查环境"
