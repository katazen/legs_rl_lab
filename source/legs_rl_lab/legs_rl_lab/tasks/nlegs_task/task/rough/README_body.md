# nlegs_body_rough

当前任务使用 `assets/nlegs_v2/nlegs_v2.py` 的 `W1196_CFG`，训练资产为 `assets/nlegs_v2/w1196/w1196.usd`。环境与 PPO 参数仍在 `body_env_cfg.py` 独立定义，不继承 `RoughEnvCfg`。任务名和输出根目录保持 `nlegs_body_rough`，新 run 默认后缀为 `w1196`，默认不恢复旧 checkpoint。

本次修改范围是训练配置及验证代码。XML、USD、关节机械边界和碰撞几何由用户维护；配置切换不能为缺少碰撞体的 USD 自动补出地面接触。训练前应在用户资产中配置脚部碰撞，并重新导出 USD。当前源 `assets/nlegs_v2/w1196.xml` 的左右第 5 关节限位为 `[-0.8, 0.8] rad`；USD 也应包含对应限位，其他关节同样需与源模型核对。

| 配置 | 当前值与依据 |
|---|---|
| 默认站姿 | hip/knee/ankle pitch 为 `-0.1 / 0.2 / -0.1 rad`；其余为 0；出生高度 `0.555 m` |
| base 高度 | 奖励目标与 critic 高度图 offset 均为 `0.51722 m`，来自 W1196 名义站姿及脚底几何 |
| 左右脚距 | `0.251803 m`，按两脚 body 原点的 y 间距；不等同于脚内侧净间隙 |
| 抬脚 | 保留 `0.12 m` 脚底离地目标，foot body 原点到脚底补偿改为 `0.015 m` |
| 倒地终止 | base 高出最低足部 body 原点不足 `0.14 m` 时终止，仍使用地形无关的相对高度 |
| 名义质量 | USD 约 `10.831 kg`；实机称重 `11.7 kg`，差额分布未识别，不直接改写各连杆惯量 |
| 质量随机化 | base 加 `[-0.9, 0.9] kg`，每次从名义质量计算；这是训练扰动，不是对线束质量位置的识别 |
| 足部实体顺序 | 所有足部奖励、接触传感器选择与终止条件均显式使用 `Link_R6, Link_L6` |
| 步态 | 周期 `0.6 s`，支撑比例 `0.55`，左右相位差 `0.5` |
| 地形 | 平地/随机崎岖/正金字塔坡/倒金字塔坡为 `20/20/30/30%`，无台阶和方块 |
| 地形难度 | 无地形课程，完整难度 `[0, 1]`；起伏 `2～6 cm`，坡度参数 `0～0.3` |
| 观测 | 15 帧；actor `705` 维，critic `3735` 维，含 187 点高度图；策略 `50 Hz` |
| PPO | 4 个 mini-batch；网络 `512/256/128`；左右镜像增强，动作裁剪 `±5`，动作 scale=`0.25` |

新旧模型的 base 原点位置不同。高度目标下降主要用于匹配新原点，髋到脚的物理长度接近，因此没有按 base 高度比例缩小地形难度、抬脚目标、速度命令或步态周期。默认站姿、动作尺度、奖励权重和 PPO 结构保留，训练后的实际步态仍需评价。

承重脚防扭奖励 `feet_contact_yaw` 的权重为 `-0.1`，根据当前竖直接触力在 `5～20 N` 连续启用，惩罚足部绕合成接触法向的角速度平方。`base_yaw` 权重为 `-1.0`，只在偏航指令绝对值小于 `0.1 rad/s` 时惩罚机身 z 角速度。`joint_deviation_hip_yaw` 以 `-0.1` 约束第 3 关节，第 2、6 关节仍为 `-0.5`。保留 xy/yaw 速度跟踪、脚掌水平、前后脚距及动作平滑奖励，不添加额外横移惩罚。

命令使用 `mdp.UniformLevelVelocityCommandCfg` 连续采样，保留 10% standing 环境，不额外把 yaw 指令置零。仅左右第 5 关节采样编码器零偏：策略位置观测为 `q_phys + b`，`ZeroBiasJointPositionAction` 将物理目标换为 `q_cmd - b`；`b` 在 reset 时采样于 `±0.05 rad`，整幕不变，critic 保持物理真值。保留材料摩擦、关节初态及速度扰动，关闭额外连杆质量、base CoM 和 PD 增益随机化。

| 执行器 | 类型 | Kp / Kd | 显式力矩 / 速度参数 | 延迟（5 ms/步） |
|---|---|---|---|---|
| 1～4 关节 | DelayedDCMotor（legs） | Kp 均 `300`；Kd 均 `5` | effort/saturation/solver `26 Nm`，模型/solver 速度 `6 rad/s` | `1～5` 步 |
| 5 踝 pitch | DelayedDCMotor（同属 legs） | `300 / 5` | 同上 | `1～5` 步 |
| 6 踝 roll | DelayedPDActuator | `50 / 1` | 模型/solver 力矩 `5.8 Nm`，速度 `29 rad/s` | `1～5` 步 |

增益、armature、摩擦和延迟沿用当前策略设置；显式给出执行器限幅，避免 USD 默认驱动值改变训练控制链。这些沿用参数尚未完成 W1196 实机系统辨识。脚部胶囊碰撞是训练所需的接触近似；若资产仅有足部碰撞，就不能通过此训练判断机身、腿部撞地或自碰撞行为。

在仓库根目录配置运行环境：

```bash
conda activate unitree_lab
source /home/woan/isaacsim/setup_conda_env.sh
export PYTHONPATH="$PWD/source/legs_rl_lab:$PYTHONPATH"
```

先检查用户资产；缺少脚部碰撞或源 XML/USD 限位不一致时，该检查会报出前置条件，且不会修改资产：

```bash
python tests/check_rough_body.py --headless --device cpu --asset-preflight-only
```

用户完成资产转换后，再顺序运行平地接触检查与完整训练链检查：

```bash
python tests/check_rough_body.py --headless --device cpu --plane-contact-only
python tests/check_rough_body.py --headless --device cpu
python tests/test_observation_symmetry.py
python tests/test_body_foot_twist_rewards.py
python tests/test_body_yaw_reward.py
```

平地检查关闭随机化，核对名义脚距、高度及 1 秒零动作 PD 的接触支撑。完整检查保留与 8.26 配置的非迁移项对照，运行 4 环境/60 步、一次 PPO 更新、JIT/ONNX 数值对照，以及 deploy 控制参数回读。MuJoCo 回放需要用户维护的匹配 XML；本检查不会用旧机器人 XML 宣称 W1196 回放通过。

资产和上述检查通过后启动新训练：

```bash
python scripts/rsl_rl/train.py --task nlegs_body_rough --headless --num_envs 2048
python scripts/rsl_rl/play.py --task nlegs_body_rough --load_run '<新训练目录名_w1196>' --num_envs 32 --real-time
```

输出位于 `logs/rsl_rl/nlegs_body_rough/<时间_w1196>/`。`params/env.yaml` 保存实际 USD 路径和环境配置，`params/deploy.yaml` 保存观测、站姿、PD、动作尺度及相位周期。15 帧旧 checkpoint 即使维度仍为 705，也是在另一种质量、几何和碰撞条件下训练的，不能据此认为适配 W1196；默认开始新 run，不重写既有 checkpoint。10 帧、470 维模型也不能直接载入当前配置。短训练及导出检查只证明执行链可运行，不代表步态质量或实机可用。
