# 统一三任务实机部署

## 统一走路 / 下蹲 / 起身

只用 `start_real.sh` 和一个 `rl_real_common` 节点。唯一公共配置是
`rl_real_py/configs/common.yaml`：`tasks` 选择三份模型，其余字段配置硬件与切换阈值，无配置覆盖层。
不会自动选择最新 checkpoint；实机模型由此配置显式选择，不跟随 sim2sim 默认值。
当前启用走路 `nlegs_flat_static/2026-09-16_11-41-05`、
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
检查通过且现场准备好后，使用 `./start_real.sh`。启动先保持当前实测姿态，等待使能后的反馈恢复，再识别站姿或蹲姿；**不自动插值回站立**，也不自动运行策略。

RL 窗口在启动、状态切换（包括动作完成）、操作被拒绝及键盘调速/清零后显示 `[操作提示]`：
只用一行显示当前中文状态和可用按键；安全说明仅启动时显示一次。
拒绝操作时列出该项检查中所有异常关节的名称、当前角度、允许范围（rad），区分 common 硬件限位、任务限位和姿态验收范围；任务测量容差单独注明，不放宽目标角限位。
正常控制循环和连续摇杆输入不重复刷屏；`--check-only` 不显示控制操作提示。

| 操作 | 键盘 | 手柄 | 允许的起点 |
|---|---|---|---|
| 走路 | 1 | LB+A | `stand_ready` |
| 下蹲 | 2 | LB+X | `stand_ready` |
| 起身 | 3 | LB+Y | `crouch_ready` |
| 手动慢回准备站姿 | 4 | LB+Start | 稳定的 `checking` / `stand_ready` / `crouch_ready` / `stopped` |
| 正常停步，速度缓降归零 | 0 | Start | `walking` |
| 人工确认接地，平滑收脚 | Enter | 再次按 Start | `stopping` |
| 中断并锁存，保持最后下发目标 | P | B | 任意阶段 |
| 重新验收，不自动复位或续播 | R | Back | `stopped` |

W/S、A/D、Q/E 控制速度；空格清零速度，不等于退出走路策略。小键盘使用数字模式。
手柄沿用已有 `/joy` 节点；按钮索引集中在 `gamepad_buttons`，默认 A/B/X/Y=0/1/2/3、LB=4、Back=6、Start=7，首次使用须根据实际 `/joy` 核对。
组合键由 A/X/Y/Start 的按下边沿触发，须先按住 LB，再按动作键；长按不会重复启动。

`4 / LB+Start` 是有支撑的手动复位，不是自主起身策略。无需匹配站姿/蹲姿模板，但须先扶稳或吊起，
反馈新鲜、无电机故障、在硬件/任务允许范围内、倾角不超过 `max_tilt`，关节和机身角速度连续低于
现有稳定阈值 `stable_time=0.30s`；走路、起身、下蹲、停步和插值中均拒绝，不排队，不自动执行。
中断后可等稳定再按 4；未解除的故障不能绕过。关节静止不代表已接地、重心稳定或具有承重能力。
沿用现有 smoothstep 插值，从最后保持目标开始（不是突然重设为实测角），12 关节同步到默认准备站姿：
每腿 `[-0.1, 0, 0, 0.2, -0.1, 0]`。峰值目标速度由 `multi_task.return_max_speed=0.15rad/s` 限制，
时长按最大目标差自动计算，当前标定蹲姿约 8.9s；这不限制电机实际速度，也不负责平衡或消除地面摩擦。
全程保留反馈、限位和调度超时检查，P/B 可中断；结束后实测站姿稳定才就绪，超时锁存，不自动走路。

与仿真的必要区别：实机目前没有足底接触和机身线速度反馈，所以不声称能自动验收双脚承重。
任务键表示操作者已确认落地站稳；正常停步必须先 0，再观察接地后 Enter。Enter 不是提前预约；姿态/速度不满足时拒绝，需要再次确认。
关节角、关节速度、倾角和角速度仍按实测连续验收 0.3s；6s 内未完成停步确认则锁存保持。
键盘不用重复 0 确认，避免按键自动连发触发收脚。

切换有 0.2s 平滑接管和首拍跳变量检查，各策略的历史、last_action、参考时钟独立；起身会从下蹲首帧开始。
Mimic 参考按有效策略步推进，结束后继续末帧策略，实测姿态到位才进入就绪；4s 未到位则锁存。
停步确认后用 0.5s 短程插值收回站姿，再连续验收；不是从任意蹲姿插值站起。
下蹲/起身使用起身导出的任务边界与 common 的交集；启用下蹲时要求两者导出边界相同。
站姿验收来自起身末帧，蹲姿验收来自起身首帧，禁用下蹲时仍可验收；不新增 XML 限位比较、不放宽 common。
`common.yaml` 的可选 `crouch_calibration` 用实测 12 关节角和机身 WXYZ 四元数替换蹲姿验收基准，
并按 `stop_sides` 替换 10 个关节各自的一端任务边界；另一端不变，ankle roll 不设实测限位。
标定必须在 common 内且允许站姿。走路保留其他边界，但 L4/R4 上界使用走路模型自身训练 clip 与 common 的交集，不再叠加旧下蹲膝上限。
反馈、保持目标检查、接管起点和收脚使用当前任务边界；停机保持、手动慢回保留上一个任务边界，切换任务前按目标任务边界验收。每个策略的最终目标按其自身任务边界裁剪。
这不是关节零位偏移：网络仍接收真实关节/IMU，原训练 clip、NPZ 和模型均不变。
当前标定为 2026-09-17 11:02 实测蹲姿（前倾 29.729°），10 个单侧边界与 v2 起身首帧一致。
ankle roll 和机身倾斜仍用实测值验收；网络参考保持生成数据中的几何求解值（前倾约 28.282°），不伪造真实观测。
新起身为 336 帧、100Hz、3.35s；50Hz 策略每步推进两帧，末帧 335 在末尾钳住并继续策略保持。
新训练的起身任务设置 `commands.motion.track_heading: false`：不追绝对朝向，仍跟踪机身倾斜、全部关节角、角速度与相对脚姿，保留落地防滑奖励。
参考不变，训练在末帧额外保持 5s，完整回合 8.35s；实际部署仍连续保持，不因训练时长自动停机。
训练、sim2sim、部署均使用该 run 的观测语义；旧导出缺少此字段时保持原朝向跟踪，不能手改旧模型 YAML 冒充新模型。
当前起身 `2026-09-17_15-40-44` 已使用关闭朝向跟踪的新权重；下蹲仍保留朝向跟踪，各自读取导出配置。
完整组合先在 sim2sim 验证；实机仍需现场保护，裁剪目标不能消除惯性造成的实测超调。
删除整个 `crouch_calibration` 段会恢复训练端点与较宽的训练任务边界，不建议在本次限位块测试中删除。
预检通过不代表实际反馈或起身动力学通过。
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

Git 仅同步部署源码；`logs/` 下的参数、导出模型和检查点需另行复制，测试与 `tools/` 实验脚本仅保留在本地。

同步到机器人主机后重新编译 `rl_real_py`，然后运行 `./start_real.sh --check-only`。
模型需包含启用 run 的 `params/`、`exported/`，以及训练使用的参考 NPZ 和 XML；禁用下蹲时不依赖旧下蹲文件。
XML 仅用于参考模型 SHA256 校验，不读取或对比其关节限位。
参考路径可按仓库的 `source/` 相对位置重定位，不会自动替换动作。
旧导出器的 `real_deployment_supported: false` 不必手改；只有通过完整格式校验的参考动作类型才受支持。

动作仍为训练的 `offset + scale * action`，参考 q/dq 仅作为网络输入；
保留策略动作限幅、训练角度 clip、任务边界与 common 硬件边界。
历史越限日志保留在离线回归中；通过测试不等于机械限位块的强度/承载验收。
聚合关节消息持续到达也不能证明每台电机反馈都新鲜，现场仍需核实零位、方向、通信和供电。
贴靠限位后的目标保持不等于可断电，实际接地、平衡和限位块承重需独立确认。

## 离线测试

以下命令需要本地保留的测试和实验脚本，这些文件不随 Git 同步。

从仓库根目录运行；不会初始化控制节点或连接电机：

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/deploy/rl_real_py:$PYTHONPATH" /usr/bin/python3 -m pytest deploy/rl_real_py/test -q
/usr/bin/python3 deploy/tools/pose_transition.py --selftest
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
