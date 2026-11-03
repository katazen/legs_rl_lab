# nlegs_body_rough_static

这是当前 W1196 步行任务的独立零速站立版本。`body_static_env_cfg.py` 完整复制了 2026-11-03 的 `body_env_cfg.py`（与 `2026-11-03_11-24-26_w1196` 保存配置一致），在本文件内独立定义场景、MDP 和 PPO，不继承 `RoughBodyEnvCfg` 或旧的 `nlegs_rough_static`。原步行任务保持原样。

这是可训练的新任务配置，**不是已经训练完成的静止模型**。旧 `model_77200` 的 JIT/ONNX 仍是踏步策略，不能通过修改 `deploy.yaml` 变成站立策略。新模型需要训练、导出并验证。

## 停步如何实现

零命令包含前后、横向和转向三维命令同时为零，阈值为 `1e-6`，不吞掉小速度或纯转向。初始就是零命令时直接采用站立目标；从运动收杆到零时，运动权重在 `1.2 s` 内渐降，站立权重同步渐升：

- gait 的两维 sin/cos 观测乘运动权重；零速稳定后为 `[0, 0]`。
- 步态、抬脚与运动高度奖励乘运动权重。
- 默认关节站姿、低关节速度、双脚接触奖励乘 `1 - 运动权重`。
- 姿态稳定、零速度跟踪、动作平滑、防滑等原有奖励继续有效。

策略始终根据当前状态输出目标角，不锁死收到零命令那一刻的动作，也不直接把动作插值成默认站姿。训练期间允许继续落脚、调整重心，再收敛到站立。`1.2 s` 是目标渐变时间，不是强制停止时间或固定两步：实际需要几步由策略和机器人状态决定。收到新的非零命令时恢复运动目标。

“初始站姿”指资产默认姿态（髋/膝/踝 pitch 为 `-0.1 / 0.2 / -0.1 rad`，其他为零），不是 reset 随机化后的关节角。它是软目标，不能保证每个编码器数值完全不变；坡地、外扰和模型误差下仍允许平衡修正，必要时可以补步。没有新增零速硬锁或修改实机停机按钮。

## 与原任务的差异

| 项目 | 新任务 |
|---|---|
| 任务及日志目录 | `nlegs_body_rough_static` |
| 命令重采样 | `3～6 s`，原为固定 `10 s` |
| 零速命令比例 | `0.35`，原为 `0.1` |
| 停步渐变 | `stop_transition_time=1.2 s` |
| 默认站姿 L1 惩罚 | `-1.0 × 站立权重` |
| 关节速度 L2 惩罚 | `-0.02 × 站立权重` |
| 双脚竖直接触奖励 | 每脚大于 `1 N` 时，`+0.5 × 站立权重` |
| 原 base_height 奖励 | 仅乘运动权重，避免运动高度与默认站姿冲突 |

W1196 资产、执行器、PD、延迟、摩擦、零偏随机化、地形、速度范围、动作尺度与裁剪、PPO 网络和其余奖励均保持当前基线。actor 仍为 `705` 维，critic `3735` 维，15 帧历史、50 Hz、12 个动作；观测顺序与左右镜像规则不变。

## 训练与导出

在仓库根目录使用本机已配置的 Isaac Lab 环境：

```bash
conda activate unitree_lab
source /home/woan/isaacsim/setup_conda_env.sh
export PYTHONPATH="$PWD/source/legs_rl_lab:$PYTHONPATH"
python scripts/rsl_rl/train.py --task nlegs_body_rough_static --headless --num_envs 2048
```

默认从头训练，输出为 `logs/rsl_rl/nlegs_body_rough_static/<时间_w1196>/`。如要从当前策略微调，需要完整的 `model_77200.pt` 等训练 checkpoint；本地该 run 目前只有导出模型，不包含完整 checkpoint，且现有 `train.py` 的跨任务恢复入口需另行处理。不要把 `exported/policy.pt` 当 PPO checkpoint。

训练完成后，用新任务及其 run 导出：

```bash
python scripts/rsl_rl/play.py --task nlegs_body_rough_static \
  --load_run '<新训练目录名_w1196>' --num_envs 32 --real-time
```

`params/deploy.yaml` 应包含 gait 观测的 `gate_by_cmd: true`、`command_threshold: 1e-6`、`stop_transition_time: 1.2`。部署与 MuJoCo 必须使用支持同样渐变语义的代码及该 run 的配套导出文件。

## MuJoCo 验证

使用有 MuJoCo、PyTorch 的 Python 环境。入口默认 W1196 场景，复用原 rough 回放器的平地起点及远处上坡/台阶，必须明确指定新 run：

```bash
python source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/rough/sim2sim_body_static.py \
  --run '<新训练目录名_w1196>'
python source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/rough/sim2sim_body_static.py \
  --run '<新训练目录名_w1196>' --headless --duration 20 --save-data
```

聚焦窗口后，主键盘或小键盘 `8/2` 前后、`4/6` 横向、`7/9` 转向、`0` 清零速度。清零后策略保持运行，禁止用旧模型的结果宣称新站立策略已经有效。headless 默认全程零命令，只能检查初始站立，不能替代运动到静止的过渡测试。

训练后至少比较：初始零速的脚部运动与站姿误差；前后、左右和纯转向后的收步；不同步态相位清零；停稳后重新启动；坡地和受扰动后恢复。记录是否持续踏步、收步时间、脚滑、身体倾角、关节目标是否突跳。软件和导出检查仅证明执行链一致，不保证停步质量或实机稳定性。

## 代码验证

在上述 `unitree_lab` 环境中运行：

```bash
python tests/test_gait_stop_transition.py
python tests/test_standing_transition_runtime.py
python tests/check_rough_body_static.py --headless --device cpu
```

前两项核对初始零命令、收杆、重新运动、局部 reset、重复读取与三端时序；最后一项使用 4 个环境检查独立配置、705/3735 维观测、奖励渐变、短仿真、一次 PPO 更新和 JIT/ONNX 导出。当前任务没有启用 Isaac RecorderManager 记录项；若启用其命令重采样前的额外观测，需要重新核对过渡时序。常规视频录制不使用这条路径。

2026-11-03 已通过上述检查、41 项原实机手柄/停止离线回归，以及烟雾测试导出模型的 MuJoCo 加载与 1.4 秒运行。烟雾模型只有一次更新，不是可部署的站立策略。
