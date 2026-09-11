# nlegs_flat_step_to_crouch

独立的“站立→踏步调整落脚点→限位下蹲→停稳”任务，不修改 flat、flat_crouch 或实机控制。
这是下蹲策略，不是 flat_crouch 的“下蹲起步后继续走路”策略。

## 时间与目标

| reset 后时间 | 要求 |
|---|---|
| 0–1 s | 从 flat 原有站立位置/速度随机状态开始，稳定身体 |
| 1–9 s | 参考进度由 0 平滑增至 1，通过交替踏步逐步下蹲 |
| 最后 20% 下降时间 | 渐弱踏步奖励与抬脚高度，准备双脚落地 |
| 9–12 s | 进度固定为 1，关闭踏步奖励，低速保持 |

进度使用 smoothstep，身体高度/倾角和十个关节的**奖励参考**随进度变化；中途关节容差较宽，
末端收紧。没有程序插值驱动、强行写关节角、固定脚或固定根节点。动作仍是原来的
`default_joint_pos + 0.25 * action`，同样使用执行器延迟和力矩限制。步态周期仍为 0.6 s。

计时从每次 reset 的 `common_step_counter` 快照开始；不依赖 PPO 随机初始化的
`episode_length_buf`。超时也使用这个计时，因此第一批环境仍然从站立要求开始。

终点十个角直接读取 `nlegs_limit.xml`：左右 hip pitch/ankle pitch 下限、knee 上限，
左 hip roll/yaw 上限、右 hip roll/yaw 下限。双踝 roll 不设关节目标，由贴地/保持要求约束。
参考身体 roll/pitch 为约 0.6064°/23.0374°，落地后 base 高度约 0.496892 m。
这些参考来自现有 `PoseSolver.solve_limit()`；修改模型后须重新计算并通过检查脚本。
启动时检查实际 USD 与 XML 硬限位一致，不再使用 crouch_pose_bank.json。

## 奖励、随机化与失败

- 删除速度追踪、直立姿态、固定站高/脚间距、髋关节回默认位置等冲突目标。
- 保留动作平滑、关节运动、能耗及接触冲击惩罚；仅对十个目标关节的目标侧放宽软限位惩罚，硬限位不变。
- 初始位置/速度随机化、质量、零偏、增益和执行器延迟随机化保留；关闭推扰和课程。
- 新任务脚底摩擦范围 0.6–1.3，动态摩擦不大于静态摩擦；这不是实机摩擦标定结果。
- 独立的左右脚接触传感器获得与地面的平均接触位置、支撑力，计算该接触中心的水平速度；
  承重绕竖轴扭脚另加惩罚，避免只看脚中心平移。
- 准备 1 s 后，只累加单脚法向力 >10 N 时的接触中心滑移估计。任一脚累计超过 2 cm 就失败重置。
  这是 50 Hz、平均接触点的估计，**不是所有接触点的精确滑移积分**；需用更高频回放和实机数据复核。
- 双脚腾空受罚；身体塌陷、倾斜超过约 60°也会失败。

GPU 接触过滤不支持原来的无限 Plane 碰撞体，因此仅本任务使用本地生成、固定不动的运动学平板；
顶面 z=0，尺寸覆盖整个环境阵列，摩擦配置照常生效。没有在线地面 USD、MDL 或 HDR 依赖。

## 成功指标

`Metrics/crouch_progress/success` 要求以下条件连续满足至少 1 s，并且每只脚至少完成两次有效抬落脚、
任一脚累计承重滑移估计不超过 2 cm：

- 已进入最终保持阶段，十个关节最大误差 ≤0.02 rad（约 1.15°）。
- 双脚各承重 >10 N、倾角 <7°，身体高度误差 <2.5 cm、重力方向误差约 <5°。
- 机身线速度 <0.05 m/s、角速度 <0.1 rad/s，各关节速度 <0.15 rad/s，接触中心滑速 <0.02 m/s。

有效抬脚须整个脚底离地 >2 cm、接触力 <1 N、另一脚承重 >10 N，持续至少 40 ms，然后重新承重；双脚一起跳不计数。
成功只记录指标，不提前结束；策略还须保持到回合结束。日志也记录最大关节误差、左右抬落脚次数、
累计滑移和连续稳定时间。**0.02 rad 是第一版到位容差，不等于十个机械限位块已经实际受力；
也不验证断电保持。最终停放验收还须独立检查真实限位接触，以及撤除驱动力后的稳定性。**

## 使用

```bash
conda activate unitree_lab
cd /home/woan/workspace/legs_rl_lab
./legs_rl_lab.sh -t --task=nlegs_flat_step_to_crouch
./legs_rl_lab.sh -p --task=nlegs_flat_step_to_crouch --num_envs=16
python scripts/check_step_to_crouch.py --headless --device cuda:0
```

训练入口支持 Hydra 参数覆盖，例如改变下降时间：

```bash
./legs_rl_lab.sh -t --task=nlegs_flat_step_to_crouch env.commands.crouch_progress.lower_s=10.0
```

使用非默认时间训练后，play 配置也应使用相同值（当前 play 入口不支持 Hydra 覆盖）。
时间/容差在 `step_to_crouch_mdp.py:CrouchProgressCfg`，奖励权重在 `step_to_crouch_env_cfg.py:RewardsCfg`。
非默认时间覆盖后，超时和日志所用的回合长度均按三个阶段的时间更新。

新策略观测为 actor 450、critic 600，含进度与渐停步态相位；关闭镜像增强。
`deploy.yaml` 会记录新命令参数，但**不能直接用旧 flat sim2sim/start_real.sh 部署，也不能直接恢复旧 flat 检查点**。
本次实现训练/Play 任务和有界检查；短测试不是训练完成的策略。
