# rough_nogait

复用当前 `nlegs_rough` 的机器人、复杂地形、速度范围、课程、随机化与控制频率（50 Hz）。
actor 和 critic 均移除 `gait_phase`，不使用步态时钟或相位接触奖励；actor 仍盲走，critic 保留高度图。

- `feet_air_time`：复用 Isaac Lab 的双足单脚支撑奖励，权重 0.5，时间上限 0.3 s；平移速度命令大于 0.1 m/s 时启用。
- `feet_clearance`：权重 1.0，以接触状态识别摆动脚，奖励脚底相对脚下地形接近 0.12 m 的高度，宽度参数 0.05 m。运动命令范数至少 0.1 时启用（包括原地转向）。双脚支撑、双脚腾空或扫描全部未命中时不奖励。
- PPO 保留 rough 的网络和 16 个 mini-batch；关闭依赖相位观测布局的镜像增强。观测维度为 actor 450、critic 2470，均含 10 帧历史，需重新训练。

在已激活的 Isaac Lab / RSL-RL 5.x 环境中运行（本机使用 `mimic` 环境）：

```bash
# 本机环境准备，在仓库根目录执行
conda activate mimic
source ~/isaacsim/setup_conda_env.sh
export PYTHONPATH="$PWD/source/legs_rl_lab:$PYTHONPATH"

./legs_rl_lab.sh -t --task rough_nogait --num_envs 4096
./legs_rl_lab.sh -p --task rough_nogait --load_run '<训练目录名>' --num_envs 32 --real-time
```

输出目录为 `logs/rsl_rl/rough_nogait/`。MuJoCo 专用入口为同目录的 `sim2sim_nogait.py`，
默认 run 为 `2026-10-06_18-20-58`；地形、执行器和动作裁剪复用 rough，拒绝带相位的策略。
先用 play 导出 `exported/policy.pt`，再在带 MuJoCo 的环境中回放（本机使用 `env_isaaclab`）：

```bash
conda activate env_isaaclab
python source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/rough/sim2sim_nogait.py
# --run 接受训练目录名或绝对路径；可视化窗口下 8/2、4/6、7/9 调速
python source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/rough/sim2sim_nogait.py --run '<训练目录名>' --headless --duration 10 --cmd 0.2 0 0 --save-data
python source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/rough/sim2sim_nogait.py --check-terrain
```

配置、奖励边界及短运行检查：

```bash
python tests/check_rough_nogait.py --headless --device cuda:0
# 在 MuJoCo 环境中检查专用回放入口，需先导出默认 run 的策略
python tests/check_rough_nogait_sim2sim.py
```

奖励参数为初始训练配置；实际步态和复杂地形通过能力需要训练后验证。
