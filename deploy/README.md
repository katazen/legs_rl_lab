# 统一三任务实机部署

## 统一走路 / 下蹲 / 起身

只用 `start_real.sh` 和一个 `rl_real_common` 节点。唯一公共配置是
`rl_real_py/configs/common.yaml`：`tasks` 选择三份模型，其余字段配置硬件与动作过渡，无配置覆盖层。
不会自动选择最新 checkpoint；实机模型由此配置显式选择，不跟随 sim2sim 默认值。
当前启用走路 `nlegs_rough/2026-09-20_19-15-58`、
下蹲 `nlegs_mimic_crouch/2026-09-17_16-42-11`、起身 `nlegs_mimic_stand/2026-09-17_15-40-44`。
下蹲按 2 / LB+X，起身按 3 / LB+Y；换模型仍须通过端点、任务限位与 PD 一致性检查。
需要临时禁用下蹲时可设 `tasks.crouch: null`，禁用按键不会改变当前目标。
当前下蹲训练已改用 `nlegs_crouch/motions/stand_to_crouch_v3.npz`：341 帧、100Hz、3.4s，
末帧与 `crouch_to_stand_v2.npz` 首帧的关节角和机身姿态一致。沿用当前 limit USD/XML、原奖励和朝向跟踪，
随机推扰仍只在 1.48–2.28s 下蹲段施加；旧 v2 文件保留，仅供旧记录核对。
`rl_real_common.py` 中的 `Policy` 共用模型加载、观测、动作映射；`multi_task.py` 只管理切换。
旧单策略入口、`start_mimic.sh`、`start_now.sh`、`--multi`、`--start-from-current` 和自动回站立已退役。

当前模型回放（仓库根目录、`unitree_lab` 环境）：

```bash
python source/legs_rl_lab/legs_rl_lab/tasks/mimic_task/task/nlegs_crouch/sim2sim.py
python scripts/sim2sim.py
```

独立下蹲回放从站姿等待，按 5 开始；统一回放按 2 下蹲、3 起身。
统一回放未指定的模型读取 common 的 `tasks`，不再默认混用旧下蹲/旧起身；可用 `--rise-run` 验证新起身。
Play 使用所选 run 记录的参考文件，不会把旧权重自动配成新版数据；旧模型与当前模型校验不符时仍拒绝运行。
以后更新权重需先用对应任务的 Play 导出，再用 `--run` / `--crouch-run` 离线验证，最后显式更新 `tasks.crouch`。
部署代码自动读取该 run 的参考和 action clip，已有实测单侧任务边界及 common 硬件边界保持不变。
参考端点匹配和仿真通过不能代替实机验证；不混用旧 v2 下蹲权重与 v3 数据。

在实际控制机器人的主机上编译、预检：

```bash
cd /home/woan/workspace/legs_rl_lab/deploy/rl_real_py
source /opt/ros/humble/setup.bash
colcon build --packages-select rl_real_py
cd /home/woan/workspace/legs_rl_lab/deploy
./start_real.sh --check-only
```

`--check-only` 不启动驱动、不发布目标、不改 PD。正常启动前也会预检启用的模型、参考端点、共同 PD/周期和限位；不一致就拒绝启动。
检查通过且现场准备好后，使用 `./start_real.sh`。启动收到完整关节和 IMU 反馈后保持当前实测姿态，按关节角选择就近的站姿或蹲姿入口；**不自动插值回站立**，也不自动运行策略。

RL 窗口在启动、状态切换（包括动作完成）、操作被拒绝及键盘调速/清零后显示 `[操作提示]`：
只用一行显示当前中文状态和可用按键。
正常控制循环和连续摇杆输入不重复刷屏；`--check-only` 不显示控制操作提示。

| 操作 | 键盘 | 手柄 | 允许的起点 |
|---|---|---|---|
| 走路 | 1 | LB+A | `stand_ready` |
| 下蹲 | 2 | LB+X | `stand_ready` |
| 起身 | 3 | LB+Y | `crouch_ready` |
| 手动慢回准备站姿 | 4 | LB+Start | `stand_ready` / `crouch_ready` / `stopped` |
| 立即停止走路策略并下发默认站姿 | 0 | Start | `walking` |
| 中断并锁存，保持最后下发目标 | 无 | B | 任意阶段 |
| 解锁，不自动复位或续播 | R | Back | `stopped` |

W/S、A/D、Q/E 控制速度；空格清零速度，不等于退出走路策略。小键盘使用数字模式。
手柄统一使用 ROS `game_controller_node` 的 SDL 标准映射，发布到专用 `/gamepad`；不再读取原始 `/joy`。
机器人电脑另开终端运行一份手柄节点（`start_real.sh` 不重复启动它）：

```bash
source /opt/ros/humble/setup.bash
ros2 run joy game_controller_node --ros-args -p autorepeat_rate:=20.0 -r joy:=/gamepad
```

左摇杆前后/左右控制前后/横移，右摇杆左右控制转向，扳机不控制速度。
速度指令不做缓升/缓降，摇杆回中后下一控制周期归零；保留死区、失联归零和任务接管过渡。
标准轴为 LEFTX/LEFTY/RIGHTX/RIGHTY/LT/RT=0/1/2/3/4/5；按钮 A/B/X/Y=0/1/2/3、LB=9、Back=4、Start=6。
`gamepad_buttons` 已按此标准配置；不能把普通 `joy_node` 重映射到 `/gamepad`，原始轴序与标准轴序不同。
首次连接或切换有线/无线后，先不运行机器人控制，用 `ros2 topic echo /gamepad` 核对回中、方向及按钮。
摇杆回中仅归零速度命令，当前 rough 仍会踏步；按 Start 会停止策略并立即下发默认站姿目标。
组合键由 A/X/Y/Start 的按下边沿触发，须先按住 LB，再按动作键；长按不会重复启动。

北通手柄访问权限在机器人电脑安装一次（从仓库根目录执行），以后重插/重启仍生效：

```bash
sudo install -m 0644 deploy/99-beitong-joystick.rules /etc/udev/rules.d/99-beitong-joystick.rules
sudo udevadm control --reload-rules
```

随后只重插手柄/接收器，再重启手柄节点；不重插电机 CAN。
规则按 USB 身份匹配并保留旧型号，仅授权 joystick 输入，不依赖 `eventN` 编号，也不放开所有输入设备。
权限与无线配对是两回事：接收器被识别不等于手柄已配对；未知 SDL 映射需先核对，不能猜测轴号。

`4 / LB+Start` 是有支撑的手动复位，不是自主起身策略。无需匹配站姿/蹲姿模板；在保持状态下可立即触发，动作执行中拒绝，不排队。
关节静止不代表已接地、重心稳定或具有承重能力，执行前须先扶稳或吊起。
沿用现有 smoothstep 插值，从最后保持目标开始（不是突然重设为实测角），12 关节同步到走路模型的默认准备站姿。
峰值目标速度由 `multi_task.return_max_speed=0.15rad/s` 限制，时长按最大目标差自动计算；这不限制电机实际速度，也不负责平衡或消除地面摩擦。
插值结束后进入站立就绪；确认机器人站稳后按 1 / LB+A 开始走路，不会自动启动。手柄 B 可中断。

与仿真的必要区别：实机目前没有足底接触和机身线速度反馈，所以不声称能自动验收双脚承重。
任务键表示操作者已确认可以开始；走路时按 0/Start 立即退出策略并下发默认站姿目标，不等待接地确认。实际关节运动仍受电机和负载影响，并非瞬间到位。

切换有 0.2s 平滑接管，不再检查首拍跳变量；各策略的历史、last_action、参考时钟独立，起身会从下蹲首帧开始。
Mimic 参考按有效策略步推进，参考结束后直接进入就绪，继续末帧策略；不检查实测是否到位。
走路停步不做插值；4/LB+Start 的手动慢回站姿仍使用限速插值。
下蹲/起身使用起身导出的任务边界与 common 的交集；启用下蹲时要求两者导出边界相同。
站姿与蹲姿参考端点仅用于入口分类；不新增 XML 限位比较、不放宽 common 目标裁剪。
`common.yaml` 的可选 `crouch_calibration` 用实测 12 关节角和机身 WXYZ 四元数替换蹲姿分类基准，
并按 `stop_sides` 替换 10 个关节各自的一端任务边界；另一端不变，ankle roll 不设实测限位。
标定必须在 common 内且允许站姿。走路保留其他边界，但 L4/R4 上界使用走路模型自身训练 clip 与 common 的交集，不再叠加旧下蹲膝上限。
任务目标与插值目标仍按对应任务边界裁剪；实测角度越界不再自动中断。
这不是关节零位偏移：网络仍接收真实关节/IMU，原训练 clip、NPZ 和模型均不变。
当前标定为 2026-09-17 11:02 实测蹲姿（前倾 29.729°），10 个单侧边界与 v2 起身首帧一致。
网络参考保持生成数据中的几何求解值（前倾约 28.282°），不伪造真实观测。
新起身为 336 帧、100Hz、3.35s；50Hz 策略每步推进两帧，末帧 335 在末尾钳住并继续策略保持。
新训练的起身任务设置 `commands.motion.track_heading: false`：不追绝对朝向，仍跟踪机身倾斜、全部关节角、角速度与相对脚姿，保留落地防滑奖励。
参考不变，训练在末帧额外保持 5s，完整回合 8.35s；实际部署仍连续保持，不因训练时长自动停机。
训练、sim2sim、部署均使用该 run 的观测语义；旧导出缺少此字段时保持原朝向跟踪，不能手改旧模型 YAML 冒充新模型。
当前起身 `2026-09-17_15-40-44` 已使用关闭朝向跟踪的新权重；下蹲仍保留朝向跟踪，各自读取导出配置。
完整组合先在 sim2sim 验证；实机仍需现场保护，裁剪目标不能消除惯性造成的实测超调。
删除整个 `crouch_calibration` 段会恢复训练端点与较宽的训练任务边界，不建议在本次限位块测试中删除。
预检通过不代表实际反馈或起身动力学通过。策略接管时把插值起点裁到当前任务目标边界，不限制这个修正的幅度。
运行中的关节/IMU 反馈越界或过期、电机告警、推理/调度延迟、参考结束未到位均不再自动锁停；手柄 B 是唯一的中断锁存入口。
格式错误的手柄反馈被忽略；策略推理异常时保留上一个目标并重试。首次接管仍须收到有限数值的完整关节与 IMU 反馈。

日志按当前模型分别写入其 `sim2real` 目录，记录 `task`、`state`、参考帧及实际待发布目标 `pub*`。
手柄 B 只保持最后目标，不切断电机，也无法在手柄消息中断时触发。首次运行应核对左右腿、零位和按钮映射。

离线验证命令见本文末尾；新增检查仅验证自动异常不锁停与手柄 B 中断，不连接机器人。

## 文件分工

- `rl_real_common.py`：三任务共用的模型推理、ROS 输入/输出、键盘/手柄输入和日志。
- `multi_task.py`：唯一切换状态机；不是另一个 ROS 节点或发布者。
- `motion_reference.py`：下蹲/起身参考文件、坐标和训练参数校验。
- `deployment_config.py`：读取完整配置、解析仓库路径、核对共同 PD。
- `sync_pd.py`：驱动启动前同步 PD；保留 `max_vel=0`，不添加训练外速度前馈。
- `tools/`：`pose_move.py` / `pose_transition.py` 实验插值，`latch_check.py` 上电跳变诊断，`com_check.py` 重心分析，`analyze_pose_log.py` 日志分析。不会被正常部署自动调用。实验控制工具不能与 RL 同时发布。
- 实验数据仍在 `deploy/pose_logs/`，未迁移、未删除；已有 `sysid/` 辨识工作区保持独立。

每个 run 的 `params/deploy.yaml` 是训练导出元数据，不属于需要合并的公共配置。
ROS 工作区的 `setup.py`、`package.xml` 和驱动配置也保留；`build/`、`install/`、`log/` 是生成产物。

## 模型同步与目标边界

Git 同步部署源码和本页的最小离线检查；`logs/` 下的参数、导出模型和检查点需另行复制，其余本地测试与 `tools/` 实验脚本不会随 Git 同步。

同步到机器人主机后重新编译 `rl_real_py`，然后运行 `./start_real.sh --check-only`。
模型需包含启用 run 的 `params/`、`exported/`，以及训练使用的参考 NPZ 和 XML；禁用下蹲时不依赖旧下蹲文件。
XML 仅用于参考模型 SHA256 校验，不读取或对比其关节限位。
参考路径可按仓库的 `source/` 相对位置重定位，不会自动替换动作。
旧导出器的 `real_deployment_supported: false` 不必手改；只有通过完整格式校验的参考动作类型才受支持。

动作仍为训练的 `offset + scale * action`，参考 q/dq 仅作为网络输入；
保留策略动作限幅、训练角度 clip、任务边界与 common 硬件边界。
通过离线测试不等于机械限位块的强度或承载验收。
聚合关节消息持续到达也不能证明每台电机反馈都新鲜，现场仍需核实零位、方向、通信和供电。
贴靠限位后的目标保持不等于可断电，实际接地、平衡和限位块承重需独立确认。

## 离线测试

从仓库根目录运行；不会初始化控制节点或连接电机：

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/deploy/rl_real_py:$PYTHONPATH" /usr/bin/python3 -m pytest deploy/rl_real_py/test_manual_stop_only.py -q
```

跨运行时对齐先在训练环境生成当前下蹲、起身回放（输出到自己的临时目录）：

```bash
python tests/check_crouch_mimic_sim2sim.py --task crouch --parity-output /tmp/crouch.npz
python tests/check_crouch_mimic_sim2sim.py --task stand --parity-output /tmp/rise.npz
```

再给上述 pytest 命令设置 `MULTI_PARITY_DIR=/tmp`，验证当前两份模型的观测、ONNX 输出和训练裁剪目标逐帧一致。
部署随后仍会按实测任务边界裁剪，不能把模型输出对齐等同于实机动力学对齐。
未提供回放文件的动作跳过对齐检查，其余测试照常执行；旧三任务/事故回归使用固定旧模型与测试专用 XML，
不会修改当前训练资产或恢复旧下蹲部署。
