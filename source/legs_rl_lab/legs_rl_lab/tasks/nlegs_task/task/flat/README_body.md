# nlegs_body_flat

W1196 平地行走与零速踏步任务。24 项共同奖励的权重取自 `nlegs_flat/2026-08-26_13-57-34/params/env.yaml`；其余奖励按用户选择保留当前 rough。scene、observations、actions、commands、rewards、terminations、events、curriculum、gait 和 PPO 配置均在 `body_env_cfg.py` 内直接定义，不继承旧 flat、rough 或 static 任务配置；只复用机器人资产与基础奖励函数。任务名仍为 `nlegs_body_flat`。

## 零速踏步

- actor/critic 的相位均明确设置 `gate_by_cmd=False`，`[sin, cos]` 始终随时间变化，零速也不归零。
- 步态接触和抬脚奖励明确设置 `moving_only=False`，零速也启用。
- 删除 `standing_joint_pos`、`standing_joint_vel`、`standing_feet_contact` 三项站立专用奖励。
- 删除 1.2 秒停步过渡和环境初始化时批量改写奖励/观测参数的逻辑。每项参数直接写在对应定义里。
- 使用与 rough 相同的 `UniformLevelVelocityCommandCfg`，不再进行低速度命令清零或静止模式切换。命令每 `10 s` 重采样，`10%` 的零命令样本用于学习原地踏步；其余样本仍学习前后、横移和转向。
- `0.1` 仍是无转向偏航惩罚和脚间距奖励的命令门槛；这些门槛不会关闭相位或让机器人静止。
- 前后对齐奖励是 `feet_x_distance`：仅 `abs(cmd_vx) < 0.1` 且两只脚当前都接触地面时，按权重 `-2` 惩罚机体系前后位置差 `abs(x_R-x_L)`。接触由当前接触时间判断，单脚支撑/腾空时为零，不使用相位代替接触。位置沿用完整机体系（含身体倾斜），不是世界系坐标。横向宽度奖励 `feet_y_distance` 保持原规则与 W1196 宽度目标。

零速要求平均平移和偏航速度接近零，同时保持踏步；它不要求关节完全不动，也不保证未经训练的策略已能踏稳。

## 保留的配置

共同奖励按指定旧 flat 的权重逐项核对，保留 W1196 几何、执行器与零速踏步语义：

- 纯平面，无高度扫描；actor `705D`、critic `930D`，15 帧历史。
- W1196 资产和当前 rough 的执行器：1～5 号关节 Kp/Kd `300/5`，踝 roll `50/1`，延迟均为 `1–5` 个物理步。
- 物理步长 `0.005 s`、策略步长 `0.02 s`，动作裁剪 `±5`、缩放 `0.25`。
- 步态周期 `0.6 s`、支撑比例 `0.55`、左右相位差 `0.5`；抬脚奖励目标 `0.06 m`。
- 平地稳定性权重：竖直速度 `-2`、roll/pitch 角速度 `-0.05`、机身倾斜 `-4`、高度 `-5`、足部水平 `-1`。共有的 24 项权重均与指定旧 flat 一致。
- 用户选择保留 rough 的额外项：身体无转向偏航 `-1`、承重足防扭 `-0.1`、髋 yaw 单独回中 `-0.1`；2/6 号关节回中 `-0.5`。不加入旧 flat 的 `lateral_move=-5`，髋 yaw 不重复计罚。
- 仅参考旧策略权重，不复制旧机器人目标：base 高度 `0.51722 m`、脚宽 `0.251803 m`、抬脚 `0.06 m`、接触力阈值 `300 N` 继续使用当前 W1196 rough 的数值。
- 与当前 rough 相同的摩擦、质量、初态、关节零偏和推扰事件，以及 PPO、镜像增强、动作模型和终止条件。
- 命令范围：vx `[-0.3, 0.5]`、vy `[-0.3, 0.3] m/s`、yaw `[-1, 1] rad/s`。

## 训练与回放

在仓库根目录、已配置的 Isaac Lab 环境中从头训练：

```bash
python scripts/rsl_rl/train.py --task nlegs_body_flat --headless --num_envs 4096
```

阿里云当前同步目录为 `/mnt/workspace/legs_rl_lab_20261022`。该服务器的 Python 默认可编辑包指向另一份工程，启动这份任务时显式设置项目路径（仅影响当前终端）：

```bash
cd /mnt/workspace/legs_rl_lab_20261022
source /root/miniconda3/etc/profile.d/conda.sh
conda activate unitree_lab
source /mnt/workspace/isaacsim/setup_conda_env.sh
export PYTHONPATH="$PWD/source/legs_rl_lab:${PYTHONPATH}"
OMNI_KIT_ALLOW_ROOT=1 python scripts/rsl_rl/train.py --task nlegs_body_flat --headless --num_envs 4096
```

每次创建独立的日期目录，旧静止版日志和权重保留。不要把旧模型的相位门控直接改为 false 当成已训练的踏步版本；这会改变它的输入语义。训练后须分别检验原地踏步、行走、转向与抗扰动效果。

导出及回放使用对应训练目录保存的配置：

```bash
python scripts/rsl_rl/play.py --task nlegs_body_flat --load_run '<训练目录名_w1196>' --num_envs 32 --real-time
python source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/flat/sim2sim_body.py --run '<训练目录名_w1196>'
```

MuJoCo 和实机运行代码读取该 run 的 `params/deploy.yaml`；新踏步版本会导出 `gate_by_cmd: false`，旧静止版导出保持原来的语义。

实现检查入口：`python tests/check_flat_body.py --headless --device cpu`。检查配置独立性、零速周期相位、零速踏步/抬脚奖励、短程运行、一次 PPO 更新及导出。它证明实现可运行，不代表正式训练的步态质量。
