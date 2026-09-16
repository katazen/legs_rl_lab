# 统一三任务实机部署

## 统一走路 / 下蹲 / 起身

只用 `start_real.sh` 和一个 `rl_real_common` 节点。唯一公共配置是
`rl_real_py/configs/common.yaml`：`tasks` 选择三份模型，其余字段配置硬件与切换阈值，无配置覆盖层。
默认模型与 `scripts/sim2sim.py` 一致；不会自动选择最新 checkpoint。
`rl_real_common.py` 中的 `Policy` 共用模型加载、观测、动作映射；`multi_task.py` 只管理切换。
旧单策略入口、`start_mimic.sh`、`start_now.sh`、`--multi`、`--start-from-current` 和自动回站立已退役。

在实际控制机器人的主机上编译、预检：

```bash
cd /home/woan/workspace/legs_rl_lab/deploy/rl_real_py
source /opt/ros/humble/setup.bash
colcon build --packages-select rl_real_py
cd /home/woan/workspace/legs_rl_lab/deploy
./start_real.sh --check-only
```

`--check-only` 不启动驱动、不发布目标、不改 PD。正常启动前也会预检三份模型、参考端点、共同 PD/周期和限位；不一致就拒绝启动。
检查通过且现场准备好后，使用 `./start_real.sh`。启动先保持当前实测姿态，等待使能后的反馈恢复，再识别站姿或蹲姿；**不自动插值回站立**，也不自动运行策略。

RL 窗口在启动、状态切换（包括动作完成）、操作被拒绝及键盘调速/清零后显示 `[操作提示]`：
当前中文状态、状态代号，以及当前可用的键盘/手柄操作和目标状态。
正常控制循环和连续摇杆输入不重复刷屏；`--check-only` 不显示控制操作提示。

| 操作 | 键盘 | 手柄 | 允许的起点 |
|---|---|---|---|
| 走路 | 1 | LB+A | `stand_ready` |
| 下蹲 | 2 | LB+X | `stand_ready` |
| 起身 | 3 | LB+Y | `crouch_ready` |
| 正常停步，速度缓降归零 | 0 | Start | `walking` |
| 人工确认接地，平滑收脚 | Enter | 再次按 Start | `stopping` |
| 中断并锁存，保持最后下发目标 | P | B | 任意阶段 |
| 重新验收，不自动复位或续播 | R | Back | `stopped` |

W/S、A/D、Q/E 控制速度；空格清零速度，不等于退出走路策略。小键盘使用数字模式。
手柄沿用已有 `/joy` 节点；按钮索引集中在 `gamepad_buttons`，默认 A/B/X/Y=0/1/2/3、LB=4、Back=6、Start=7，首次使用须根据实际 `/joy` 核对。
组合键由 A/X/Y 的按下边沿触发，须先按住 LB，再按动作键；长按不会重复启动。

与仿真的必要区别：实机目前没有足底接触和机身线速度反馈，所以不声称能自动验收双脚承重。
任务键表示操作者已确认落地站稳；正常停步必须先 0，再观察接地后 Enter。Enter 不是提前预约；姿态/速度不满足时拒绝，需要再次确认。
关节角、关节速度、倾角和角速度仍按实测连续验收 0.3s；6s 内未完成停步确认则锁存保持。
键盘不用重复 0 确认，避免按键自动连发触发收脚。

切换有 0.2s 平滑接管和首拍跳变量检查，各策略的历史、last_action、参考时钟独立；起身会从下蹲首帧开始。
Mimic 参考按有效策略步推进，结束后继续末帧策略，实测姿态到位才进入就绪；4s 未到位则锁存。
停步确认后用 0.5s 短程插值收回站姿，再连续验收；不是从任意蹲姿插值站起。
未配置标定时，所有策略共用下蹲/起身导出的任务边界与 common 的交集；不新增 XML 限位比较、不放宽 common。
`common.yaml` 的可选 `crouch_calibration` 用实测 12 关节角和机身 WXYZ 四元数替换蹲姿验收基准，
并按 `stop_sides` 替换 10 个关节各自的一端任务边界；另一端不变，ankle roll 不设实测限位。
标定必须在 common 内且允许站姿；反馈、保持目标检查、接管起点、所有策略最终目标和收脚共用该任务边界。
这不是关节零位偏移：网络仍接收真实关节/IMU，原训练 clip、NPZ 和模型均不变。
当前标定为 2026-09-16 16:00 实测蹲姿（前倾 32.098°）。首次仅验证起身→站立→走路；
旧策略适应新几何、贴合新限位和动态平衡均未实机验证，裁剪目标也不能消除惯性造成的实测超调。
重新训练后核对标定；删除整个 `crouch_calibration` 段恢复训练端点。预检通过不代表实际反馈或起身动力学通过。
统一模式的就绪与运行反馈都保留 0.03rad 任务测量容差，避免接触偏差让“蹲完起身”卡住；common 反馈边界仍无容差。
若刚启动时捕获的保持目标仅在任务边界外这点容差内，进入策略的插值起点先裁到边界（最多 0.03rad 修正）；更大偏差拒绝接管，策略目标始终不放宽。
每个模型仍先执行自身训练 action clip；硬件越界、任务反馈越界、NaN/Inf、反馈过期、推理/调度超时和收到的电机故障均锁存，不自动续播。

日志按当前模型分别写入其 `sim2real` 目录，记录 `task`、`state`、参考帧及实际待发布目标 `pub*`。
P/B 不是硬件急停；保持目标也不保证失衡或掉电时不倒。代码验收不等于实机验收，首次运行应保护机器人，并核对左右腿/零位/按钮映射。

离线回归命令见本文末尾；`test_multi_task.py` 用真实 ONNX 与合成反馈验证状态机，不连接机器人。

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

## 模型同步与保护边界

同步到机器人主机后重新编译 `rl_real_py`，然后运行 `./start_real.sh --check-only`。
模型需包含各 run 的 `params/`、`exported/`，以及训练使用的参考 NPZ 和 XML。
XML 仅用于参考模型 SHA256 校验，不读取或对比其关节限位。
参考路径可按仓库的 `source/` 相对位置重定位，不会自动替换动作。
旧导出器的 `real_deployment_supported: false` 不必手改；只有通过完整格式校验的参考动作类型才受支持。

动作仍为训练的 `offset + scale * action`，参考 q/dq 仅作为网络输入；
保留策略动作限幅、训练角度 clip、任务边界与 common 硬件边界。
历史越限日志保留在离线回归中；通过测试不等于机械限位块的强度/承载验收。
聚合关节消息持续到达也不能证明每台电机反馈都新鲜，现场仍需核实零位、方向、通信和供电。
贴靠限位后的目标保持不等于可断电，实际接地、平衡和限位块承重需独立确认。

## 离线测试

从仓库根目录运行；不会初始化控制节点或连接电机：

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/deploy/rl_real_py:$PYTHONPATH" /usr/bin/python3 -m pytest deploy/rl_real_py/test -q
/usr/bin/python3 deploy/tools/pose_transition.py --selftest
```

跨运行时对齐先在训练环境生成两个回放文件（输出到自己的临时目录）：

```bash
python tests/check_crouch_mimic_sim2sim.py --task crouch --parity-output /tmp/crouch.npz
python tests/check_crouch_mimic_sim2sim.py --task stand --parity-output /tmp/rise.npz
```

再给上述 pytest 命令设置 `MULTI_PARITY_DIR=/tmp`，验证两个动作的观测、ONNX 输出和目标逐帧一致。
未提供回放文件时仅跳过这两项，其余限位、状态切换和历史事故回归照常执行。
