# 双腿机器人最小实机部署

`deploy` 分支只保留实机部署代码和指定走路策略。它是独立的 orphan 分支，不包含训练分支的提交历史；可单独克隆到机器人电脑。

唯一策略：`logs/rsl_rl/nlegs_body_rough/2026-11-03_11-24-26_w1196`，来源 `model_77200.pt`。输入为 **705 维**（47 维 × 15 帧历史），输出为 **12 个关节动作**；策略推理 **50 Hz**，目标指令发布 **200 Hz**。下蹲和起身策略均已禁用。

该策略 L1–L5、R1–R5 的 PD 为 **Kp=300、Kd=5**，L6/R6 为 **Kp=50、Kd=1**。启动脚本按策略的 `params/deploy.yaml` 自动同步 PD。策略和 PD 必须配套，切换前须停止旧控制进程。

## 环境与编译

目标环境：**Ubuntu 22.04、x86_64、ROS 2 Humble、系统 Python 3.10**。需要 ROS 的 `rclcpp`、`rclpy`、`std_msgs`、`sensor_msgs`、`ament_index_cpp`、`ament_index_python`、`launch_ros`、`joy`（`ros-humble-joy`），以及 colcon、C++17 编译工具、`libyaml-cpp-dev` 和 `gnome-terminal`。

先按 [ROS 2 Humble 官方安装说明](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) 配置 UTF-8 locale、Ubuntu Universe 和 ROS 软件源，并完成文档要求的系统更新。然后安装本项目依赖：

```bash
sudo apt update
sudo apt install -y ros-humble-ros-base ros-humble-joy ros-humble-ament-cmake \
  ros-humble-launch-ros build-essential cmake libyaml-cpp-dev \
  python3-colcon-common-extensions python3-pip gnome-terminal git
```

获取代码并编译：

```bash
mkdir -p /home/woan/workspace
cd /home/woan/workspace
git clone --single-branch --branch deploy https://github.com/katazen/legs_rl_lab.git legs_rl_lab_deploy
cd legs_rl_lab_deploy
/usr/bin/python3 -m pip install --user -r requirements.txt
bash deploy/build.sh
/usr/bin/python3 deploy/sync_pd.py
bash deploy/start_real.sh --check-only
```

已同步该目录时，直接进入目录执行依赖安装、编译和预检即可，不要重复克隆。使用系统 Python，避免激活 Conda 后混用 ROS 环境。Python 依赖为 NumPy 1.26.4、ONNXRuntime 1.23.2、PyYAML 5.4–6.x、pyserial 3.5，版本约束见 `requirements.txt`。

`build.sh` 编译电机驱动、IMU、RL 三个 ROS 包，固定使用 USB-CAN 串口后端 `SERIAL`，关闭构建测试。构建产物在机器人本地生成，不从其他电脑复制。

`--check-only` 校验配置、模型输入输出、PD 和电机参数；不打开电机串口、不使能电机、不发布控制目标。通过预检不代表硬件接线、关节方向或实际行走已验证。

## 串口与配置

| 设备 | 串口别名 | 波特率 |
|---|---|---:|
| 左腿 USB-CAN | `/dev/ttyCANL` | 921600 |
| 右腿 USB-CAN | `/dev/ttyCANR` | 921600 |
| IMU | `/dev/tty_imu` | 115200 |

启动前确认这些别名存在、左右对应正确，且当前用户有读写权限。udev 规则应按**机器人电脑的实际 USB 路径和接线**配置，不能套用笔记本上的 USB 端口路径，也不能依赖会变化的 `ttyACM0/1` 编号判断左右。IMU 的设备端波特率须与驱动一致。

- `deploy/rl_real_py/configs/common.yaml`：策略路径、手柄映射和动作过渡；速度范围直接读取策略配置。
- `deploy/control_ws/src/armcontrol/config/motors.yaml`：电机型号、CAN ID、方向和协议量程。
- 策略目录的 `params/deploy.yaml`：观测、动作、PD 和策略周期；启动时由 `sync_pd.py` 同步 PD 到驱动。

## 启动与停止

在**机器人电脑的 NoMachine 桌面终端**中执行；脚本自动启动手柄、IMU、电机驱动和 RL 节点，分别使用 GNOME 终端窗口：

```bash
cd /home/woan/workspace/legs_rl_lab_deploy
bash deploy/start_real.sh
```

**普通启动会使能电机。** 启动前支撑机器人，并确认没有另一套部署进程在运行。RL 收到完整反馈后保持当前姿态，等待人工开始。

| 操作 | 键盘 | 标准 SDL 手柄 |
|---|---|---|
| 开始走路 | `1` | `LB+A` |
| 停止走路并下发默认站姿 | `0` | `Start` |
| 中断并保持最后目标 | — | `B` |
| 解除中断 | `R` | `Back` |
| 保持状态下慢回准备站姿 | `4` | `LB+Start` |
| 切换左摇杆/方向键移动 | — | `RB` |

`W/S`、`A/D`、`Q/E` 调节速度，空格清零。速度范围与训练一致：前后 **[-0.3, 0.5] m/s**、左右 **[-0.3, 0.3] m/s**、转向 **[-1.0, 1.0] rad/s**。

手柄默认用左摇杆前后/左右移动，`RB` 切换为左方向键；右摇杆横轴始终控制转向。切换后需松开方向键、回中左摇杆和右摇杆横轴，移动输入才生效。方向键按下时使用对应方向的速度上限，松开归零；键盘速度独立保留。

`B` 不切断电机电源，`0` 会下发站姿目标；当前代码不会因反馈异常自动锁停。慢回站姿前须支撑机器人。

`start_real.sh` 自动启动手柄节点，已有正确的 `/gamepad` 节点时复用；`stop_real.sh` 一并关闭它。`--check-only` 不启动手柄。仅在单独诊断手柄时使用：

```bash
source /opt/ros/humble/setup.bash
ros2 run joy game_controller_node --ros-args -p autorepeat_rate:=20.0 -r joy:=/gamepad
```

停止前先支撑机器人，再执行：

```bash
cd /home/woan/workspace/legs_rl_lab_deploy
bash deploy/stop_real.sh
```

## 文件范围

仅包含三个 ROS 包的部署源码、启动/停止/编译脚本、配置，以及这一策略的 `policy.onnx`、`deploy.yaml`、`agent.yaml` 和来源记录 `policy_snapshot.json`。`agent.yaml` 是当前加载器所需的小型参数文件；运行使用 ONNXRuntime，无需 PyTorch、Isaac 或 MuJoCo。来源记录包含完整导出时的文件校验值，其中的 `policy.pt` 和训练检查点不随此分支分发。

不包含训练源码、其他训练运行、训练检查点、`policy.pt`、机器人仿真模型、TensorBoard 数据、历史实机日志、分析/辨识工具或测试文件。运行新产生的日志与本地构建产物由 `.gitignore` 排除。
