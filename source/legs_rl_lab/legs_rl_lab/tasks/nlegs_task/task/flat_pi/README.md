# nlegs_v2_pi

独立的 Isaac Lab `DirectRLEnv` 任务。步态、观测、奖励和 PPO 对照
[Mini_Pi_RL_Baseline 的 pai_config.py](https://github.com/HighTorque-Robotics/Mini_Pi_RL_Baseline/blob/e8b787afa1e30da4f28f8cd7fefe542a3912ac27/humanoid/envs/pai/pai_config.py)
及同提交的 `pai_env.py`、`base/legged_robot.py`，固定提交为
`e8b787afa1e30da4f28f8cd7fefe542a3912ac27`（`release_v1.0.0`）。
机器人使用 `assets/nlegs_v2/nlegs_v2.py` 的 `W1196_CFG`；不继承 flat/rough 任务配置。

## 配置与机器人适配

| 项目 | 新任务 |
|---|---|
| 物理步长 / 控制频率 | 1 ms / 50 Hz，decimation=20 |
| 平地 / 初始环境数 / 回合 | 摩擦 0.6 / 4096 / 12 s |
| 步态周期 / 抬脚累计高度目标 | 0.4 s / 0.02 m |
| 动作 | 默认姿态 + 0.25 × action；随机混合和噪声后裁剪到 ±18 |
| actor / critic | 47 × 15 = 705 / 73 × 3 = 219，整帧按时间由旧到新排列 |
| 网络 | actor 512/256/128；critic 768/256/128；ELU；无观测归一化、无镜像增强 |
| PPO | lr=1e-5、epochs=2、minibatches=4、gamma=.994、lambda=.9、entropy=.001 |
| 采样 / 训练 | 24 steps、10001 iterations、每 100 iterations 保存、seed=5 |
| v2 默认姿态 | 左右 J1/J4/J5 = −0.1 / 0.2 / −0.1 rad，其余零 |
| v2 执行器 | 保留资产中的 PD、DC motor、力矩/速度限制、摩擦和惯量 |
| 执行器延迟 | 任务副本设为 5～25 个物理步，即 5～25 ms；共享资产不改 |
| base 高度目标 / 终止高度 | 0.51722 / 约 0.44937 m |
| 足距离范围 / 膝距离范围 | 约 0.23577～0.31436 / 0.23577～0.62872 m |
| 足力上限 / 承重接触阈值 | 约 156.074 / 7.804 N |
| 脚原点 / 抬脚测量连杆 | Link_L6/R6，脚底偏移 .015 m / Link_L5/R5，偏移 .065 m |

距离阈值按默认站姿脚距比 `.251803/.1602` 换算；力阈值按 URDF 总质量比
`10.8311180249/6.93973513` 换算。抬脚目标、关节参考幅度、奖励权重不缩放。
保留 v2 的初始 base z=.555 m 和关节限位；随机复位角度为默认姿态 ±.1 rad，裁至物理限位。
Pi 的 contact/rest offset=.01/0 m 在环境初始化时直接写入 PhysX 碰撞形状；
参数为 `robot_contact_offset` / `robot_rest_offset`，保持 USD 视觉与碰撞实例复用。
不使用会跳过实例化节点的 `spawn.collision_props`；v2 自身 solver iterations=8/4 保留。
不同版本 PhysX 的线程、内存和接触收集接口不保证逐项等价，也不保证两种模拟器轨迹相同。

训练域随机化保留 Pi 的启动时摩擦 [.1,2]（256 桶）、base 加质量 [−1,1] kg、
每 4 s 设置随机世界 xy 速度 ±.2 m/s 和角速度 ±.4 rad/s。
动作每控制步按一个环境一个随机系数混合新旧动作，并叠加 2% 乘性高斯噪声。
这些训练扰动与 v2 的物理步延迟同时存在；sim2sim 使用名义摩擦 .6，关闭训练噪声，保留执行器延迟。
`PiPlayEnvCfg` 仅减少环境数，仍保留 Pi 原始 play 的噪声与扰动。

速度指令每 8 s 重采样：vx=[−.3,.6]，vy=[−.3,.3]，xy 模长不超过 .2 时清零。
训练启用 heading 指令：`yaw_rate=clip(.5*wrap(heading-yaw), -1, 1)`。
这保留了原代码实际的 ±1 rad/s，而不是仅根据配置里的 yaw 范围误改成 ±.3。
sim2sim 的 `--command` 第三个值直接是 yaw rate，手动范围仍为配置的 ±.3 rad/s。

## 观测与奖励语义

所有关节输入和动作使用 v2 SDK 顺序：`R1..R6,L1..L6`。原 Pi 的策略为左腿在前，
这里按关节、连杆名称显式适配；步态掩码顺序为左脚、右脚，sin≥0 时左支撑、右摆动。

单帧 actor：

```
[sin, cos, vx*2, vy*2, wz,
 q-default (12), dq*.05 (12), 上一次实际动作 (12),
 base机体系角速度 (3), base世界Euler角RPY (3)]
```

actor 对 q/dq/角速度/Euler 加 Pi 高斯噪声，整体 noise_level=.6。critic 无噪声，
包含命令、关节状态、动作、参考姿态误差、base 速度、Euler、推力记录、支撑/接触状态。
Pi 源码中的 `env_frictions`、`body_mass` 分配后始终为零；为保持其观测定义，这两个
critic 占位仍保留为零，实际随机化结果另存 `sampled_friction`、`sampled_base_mass`。
观测最终裁剪 ±18。复位历史零填充；读取观测不会重复推进历史。

25 项奖励的公式及权重在 [env_cfg.py](env_cfg.py) 与 [rewards.py](rewards.py) 中独立列出。
每项先乘控制 dt=.02，非终止项总和裁至非负，再加入终止惩罚。保留原代码实际定义：

- `foot_slip`：承重脚世界角速度 **x/y** 的范数再开方；不直接惩罚脚 yaw 或脚底平移。
- `default_hip_roll_joint_pos`：髋 roll 与踝 roll 的零位偏差；`default_thigh_joint_pos`：髋 yaw 零位偏差。
- `default_ankle_roll_pos`：两只脚的世界 roll/pitch 姿态，而非踝 roll 关节位置。
- `tracking_*` 的核实际是 `exp(-error² / sigma³)`，保留原数学，不按其注释改写。
- `feet_clearance`：累计踝 pitch 连杆 z 的变化；planned swing 时累计上升量落在严格
  `abs(height-.02)<.01` 范围内得分。计算奖励之后，实际触地清累计量；不是脚底绝对离地高度。
- `feet_air_time`：实际接触、计划支撑、上一拍接触三者取或来结束腾空计时。
- `base_acc` 使用六维世界速度的一拍变化，不除 dt。

修复状态与时序：每个复位环境独立清理累计高度、接触/动作/观测历史，并用复位后 FK
初始化上一脚高度；参考姿态在当前拍奖励之前更新；推扰后同拍刷新速度；回合在恰好 12 s 超时。
RSL-RL 随机化初始回合计数后，首次动作前同步刷新相位和历史。
复位刷新 FK 不推进其它环境的执行器延迟队列。

## 运行

在仓库根目录、已安装 Isaac Lab/RSL-RL 的环境中运行。本机环境初始化示例：

```bash
source /home/woan/miniconda3/etc/profile.d/conda.sh
conda activate unitree_lab
source /home/woan/isaacsim/setup_conda_env.sh
export PYTHONPATH="$PWD/source/legs_rl_lab:${PYTHONPATH:-}"

python scripts/rsl_rl/train.py --task nlegs_v2_pi --headless
python scripts/rsl_rl/play.py --task nlegs_v2_pi --num_envs 1 \
  --checkpoint /absolute/path/to/run/model_10000.pt

python source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/flat_pi/sim2sim.py \
  --run '<训练目录名>' --headless --duration 10 --command 0.3 0 0
```

可用 `--num_envs` 调整训练并行数。日志在 `logs/rsl_rl/nlegs_v2_pi/`；play 导出
`exported/policy.pt` 与 `policy.onnx`；训练同时保存独立 `params/deploy.yaml` 和任务源码快照。
直接运行 sim2sim 时默认使用脚本顶部的 `RUN`；命令行 `--run` 可覆盖它。
sim2sim 也接受 `--deploy /path/deploy.yaml --policy /path/policy.pt`，同时指定这两个路径时无需设置 `RUN`。
本机 `--video` 录制若遇到 `CXXABI_1.3.15` 缺失，可在启动前设置
`export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"`；正式 play 录制检查使用此环境设置通过。

```bash
python tests/test_pi_rewards.py
python tests/test_pi_deploy.py
python tests/check_pi.py --headless --device cpu
```

奖励测试在本地存在固定 Pi 参考源码时额外逐项比较原公式；缺参考源码时仍运行独立数值与状态检查。
集成检查覆盖四环境短程步进、部分复位、相位/观测/延迟一致性、随机化分支、一次 PPO、
JIT/ONNX 数值对齐、训练和 MuJoCo 的 47 维输入对齐以及 .4 s MuJoCo 回放。

这验证的是实现和链路，不代表已完成步态训练或改善了扭脚。原 flat/rough 的 705 维模型
与本任务的观测含义不兼容，必须重新训练。导出格式为 3，明确标记
`real_deployment_supported: false`；现有实机加载器会拒绝此任务，尚未适配新布局。
