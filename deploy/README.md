# 实机部署

实机使用两款自研电机：L1–L5、R1–R5 为大电机，L6/R6（ankle roll）为小电机。
旧代码的达妙名称只是历史软件标签，不能用于推断实际硬件、固件或电流—力矩关系。

## 配置与数据流

- `rl_real_py/configs/common.yaml`：选择策略、手柄映射、部署频率和过渡参数。
  当前只启用 rough `2026-08-26_20-37-55`，`crouch` / `rise` 均为 null。
- `control_ws/src/armcontrol/config/motors.yaml`：唯一电机型号、CAN ID、方向及协议编码量程定义。
- 各策略 `params/deploy.yaml`：训练观测、动作、PD 和周期；`sync_pd.py` 按关节名同步 PD。
- RL 节点不读取关节限位；`joint_limits.py` 仅供独立实验工具读取 MJCF 边界。

```text
电机反馈 → direction × (q, dq, effort) → /left_joint_states
q/dq + IMU → 按关节名重排 → 策略
策略动作 → 原始 action 幅度裁剪 → offset + scale × action → 重排 → /dog_joint_pos
direction × 关节位置目标 → MIT(q_des, dq_des=0, effort_ff=0, Kp, Kd)
```

ROS 数组顺序固定为 `[L1..L6,R1..R6]`，`JointState.name` 使用这些名称。
`/joint_states` 和 `/left_joint_states` 都发布完整 12 关节。
策略顺序由模型按关节名映射；重排不取反。IMU 安装旋转独立在 `_on_imu()` 转换。
驱动 `max_vel` 必须为 0，保持训练使用的零目标速度 PD。

## 电机参数与方向来源

| 关节 | 用途 | 型号 | 每侧 Slave/Master ID | 左侧 direction | 右侧 direction | 当前 Kp / Kd |
|---|---|---|---|---:|---:|---|
| 1 | 髋 pitch | large | 1 / 17 | +1 | −1 | 200 / 5 |
| 2 | 髋 roll | large | 2 / 18 | +1 | +1 | 100 / 5 |
| 3 | 髋 yaw | large | 3 / 19 | −1 | −1 | 100 / 5 |
| 4 | 膝 | large | 4 / 20 | +1 | −1 | 250 / 5 |
| 5 | 踝 pitch | large | 6 / 22 | −1 | +1 | 40 / 2 |
| 6 | 踝 roll | small | 7 / 23 | +1 | +1 | 40 / 0.5 |

左右使用独立通信接口，故 CAN ID 可以相同。
大小电机当前都继承原编码范围：位置 ±12.5、速度 ±45、effort ±30。
位置/速度接口使用 rad、rad/s；effort 保留固件反馈标度。
这些量程不是机械行程、实测转速上限或关节输出力矩上限。
PD 表对应当前 walk 模型；换模型时以导出配置与同步结果为准。

方向值继承 Git 提交 `374cc3a`（2026-09-18）中的 `kLegCmdSign` / `kLegFbSign`，本次没有重新测量或改值。
历史反馈注释称 L3、L5、R1、R3、R4 按实测取反；命令注释还提到右腿 3/5/6 的执行方向修正。
但该提交为整份源码导入，没有逐电机标定过程；旧注释描述的修改过程也不能完整推导出最终表。
因此只能确认这些是已有部署约定，不能证明它们与模型物理正方向逐项一致。
型号、CAN ID、左右镜像关系都不能单独推导安装方向。

`direction` 同时用于命令及 q/dq/带符号 effort 反馈，避免两张表各自漂移。
命令与反馈一起取反时，目标跟踪仍能自洽；effort 的相关性检查同样不能证明物理方向正确。
若要标定，需逐关节记录已知姿态下的实际运动方向，与模型小幅正角度变化比较。

## 编译与预检

在机器人使用的主机上执行：

```bash
source /opt/ros/humble/setup.bash
(cd deploy/control_ws && colcon build --packages-select armcontrol)
(cd deploy/imu_ws && colcon build --packages-select wit_ros2_imu)
(cd deploy/rl_real_py && colcon build --packages-select rl_real_py)
./deploy/start_real.sh --check-only
```

默认 x86 使用 USB-CAN 串口 `/dev/ttyCANL` / `/dev/ttyCANR`、921600 baud；
ARM 默认使用 SocketCAN `can2` / `can1`。可用 `-DARMCOMM=SERIAL` 或 `CAN` 显式选择。
两条通信路径使用同一套编码、解码及方向转换。

`--check-only` 检查模型、PD、硬件配置及安装版驱动的离线模式；不打开电机端口、不使能、不发布控制目标。
正常启动先同步 PD 和 motors.yaml 至安装目录，再启动 IMU、驱动和 RL。
驱动正常启动会使能电机；RL 收到完整反馈后保持当前实测姿态，等待操作者开始。

## 操作

| 操作 | 键盘 | 标准 SDL 手柄 |
|---|---|---|
| 开始走路 | 1 | LB+A |
| 停止走路并立即下发默认站姿 | 0 | Start |
| 中断并保持最后目标 | — | B |
| 解锁，不自动恢复运动 | R | Back |
| 保持状态下慢回站姿 | 4 | LB+Start |
| 下蹲 / 起身（启用对应模型后） | 2 / 3 | LB+X / LB+Y |
| 切换左摇杆 / 方向键平移 | — | RB（右肩键，不是 RT 扳机） |

慢回站姿使用 `return_max_speed` 限制目标插值速度，现场须支撑机器人。
B 保持目标，不切断电源；运行中反馈异常、推理异常或电机告警不会自动锁停。
推理异常保留上一目标并重试。0/Start 不等待接地确认；默认站姿目标不代表瞬间到位。
启用下蹲/起身时仍核对 PD、周期、参考文件和起身末帧；参考关节角不再做范围验收。

`start_real.sh` 自动启动手柄节点（需安装 `ros-humble-joy`），已有正确的 `/gamepad` 节点时复用；`stop_real.sh` 一并关闭手柄节点。`--check-only` 不启动手柄。仅在单独诊断手柄时使用以下命令，不与普通 `joy_node` 混用：

```bash
source /opt/ros/humble/setup.bash
ros2 run joy game_controller_node --ros-args -p autorepeat_rate:=20.0 -r joy:=/gamepad
```

W/S、A/D、Q/E 调节速度，空格清零速度。键盘速度与手柄速度叠加后裁剪到模型的训练范围。
手柄默认用左摇杆前后/左右平移；单按 RB 切换为方向键上/下/左/右平移，再按一次切回。
两种模式都由右摇杆左右控制转向。方向键为数字输入，按下即使用对应方向的训练速度上限；同时按相反方向会抵消。
切换时只清空手柄速度，需要左摇杆、方向键和右摇杆转向轴全部回中/松开一次后再操作；键盘速度保持原值。
手柄回中归零，超过 `ctrl_timeout`（默认 0.4 s）未收到消息则手柄速度归零。B、Back、Start 和 LB 组合操作优先于 RB 切换。
`common.yaml` 不再设置 `velocity_command_limits`，速度范围直接读取所选模型 `params/deploy.yaml`。
显式使用带此字段的实验配置时仍会覆盖速度范围，且不得超出训练范围。
关闭前支撑机器人，再执行 `./deploy/stop_real.sh`；退出驱动的实际失力行为不能仅由源码确认。

## 日志与实验工具

RL CSV 的 q、qd、mvel、tau、cmd、pub 都使用关节方向；obs/act 使用策略顺序。
`tau` 是电流相关的驱动 effort 标度，不是独立标定的关节输出力矩，当前只记录、不进入 RL 观测。
驱动诊断 `dual_leg_state_*.csv` 和 `dual_leg_cmd_*.csv` 保留电机坐标；默认关闭诊断。
旧 RL/姿态日志可能有 q/dq 已转换、effort 未转换的混合格式，分析前需确认生成版本和坐标。
`motor_config.detect_effort_frame()` 检查两种格式，激励不足时报告无法判定；不能据相关性完成物理标定。
电机方向更改后分析旧数据还须使用该实验保存的方向配置。

- `tools/`：姿态、插值、重心、上电跳变诊断；有独立入口，不随正常启动自动执行。
- `sysid/src/`：实验采集、分析和 Isaac 回放；由实验流程或 CLI 调用。
- `sysid/doc/`、`sysid/data/`、`pose_logs/`、`analysis/`：历史证据与结果，本次保留。
- `sysid/_scratch/compare_voltage.py` 必须显式指定 `--effort-frame motor` 或 `joint`，避免新日志重复取反。
- `sysid/_scratch/check_stage4_48v.py` 只针对写死的历史电机帧数据目录。

2026-10-13 清理了旧机械臂、夹爪、升降、碰撞/同步发布、空线程池、6/7 自由度分支、
旧方向修正函数、路径平滑器、机械臂 URDF/SRDF、logger 和三个未使用的 vendor 包，
以及生成缓存与旧构建产物。反馈及告警只访问每侧实际存在的 6 台电机。
正常启动、实验采集和可重新启用的任务模块都有入口，保留这些功能。

RL 节点已取消全部关节角度限位：不读 XML，不使用导出的 `JointPositionAction.clip`，
不裁剪首次保持角、策略目标、接管起点、插值目标或最终发布目标。
保留训练的 `policy_action_clip`（原始网络 action 幅度）、观测裁剪和速度指令范围。
因此部署目标在导出的角度 clip 外会与训练/仿真裁剪后的目标不同。
驱动协议编码范围仍由 motors.yaml 定义。

删除了未启用的差分速度/EMA、速度零偏、重复参数检查、旧角度限位辅助函数，
以及依赖旧限位/自动停止 API 的回归目录。实测蹲姿的关节角继续用于入口分类，
未被使用的 `stop_sides` 和机身四元数字段已移除。
保留完整反馈等待、模型输入输出与发布目标的基本维度/有限数检查、任务切换和手动 B 中断。

## 离线检查

```bash
source /opt/ros/humble/setup.bash
source deploy/control_ws/install/local_setup.bash
ctest --test-dir deploy/control_ws/build/armcontrol --output-on-failure
PYTHONPATH="$PWD/deploy/rl_real_py:$PWD/deploy:$PYTHONPATH" \
  /usr/bin/python3 -m pytest deploy/tests deploy/rl_real_py/test_manual_stop_only.py deploy/rl_real_py/test_gamepad_modes.py -q
PYTHONPATH="$PWD/deploy/rl_real_py:$PYTHONPATH" /usr/bin/python3 tests/check_sysid_capture.py
./deploy/start_real.sh --check-only
```

整节点回归只连接自动创建的两个 `/dev/pts/*` 虚拟串口，在独立 ROS domain 验证 12 台电机的命令/反馈方向、告警数量及退出。
这些检查不验证实机标零、物理方向、固件电流标度或实际行走动力学。
