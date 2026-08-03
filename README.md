# legs_rl_lab

> 基于 [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) 的双足腿式机器人强化学习工程：用 PPO / AMP（rsl_rl）在 GPU 并行仿真中训练腿式本体的速度跟踪 locomotion 策略，内置参数化步态时钟、左右对称数据增强、MuJoCo sim2sim 验证，以及一套完整的 **ROS 2 实机部署栈**（IMU → 电机驱动 → RL 策略，键盘/手柄控速）。

![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.5%20%7C%205.x-76b900)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue)
![RL](https://img.shields.io/badge/RL-rsl__rl%20PPO%20%2F%20AMP-orange)
![Deploy](https://img.shields.io/badge/Deploy-ROS%202%20Humble-22314e)
[![Stars](https://img.shields.io/github/stars/katazen/legs_rl_lab?style=social)](https://github.com/katazen/legs_rl_lab)

**适合谁用**：正在用 Isaac Lab / Isaac Sim 做腿足运动控制（locomotion）RL、并希望一路做到真机部署的研究者与工程师。训练需要一台带 NVIDIA GPU 的机器并装好 Isaac Lab；部署需要一台装了 ROS 2 Humble 的机器与目标本体（12-DOF 双足）。

---

## ✨ 项目亮点

- **两种训练范式**：常规 PPO 速度跟踪（`legs` / `nlegs`），以及 **AMP（对抗式动作先验）** 走路任务 `nlegs_amp`——专家动作由 TienKung-Lab 的 walk 轨迹重定向到窄本体，判别器提供风格奖励。
- **参数化步态时钟**：步态周期 / 支撑相占比 / 左右相位偏移收敛到 `GaitCfg`，随 `env.yaml` 落盘，训练与部署逐位可复现（相位仅由 episode 时间与周期决定，不靠计数器堆积）。
- **面向真机的奖励设计**：速度跟踪 + 姿态 + 足部间距 / 打滑 / 接触力 / 触地相位匹配等成套奖励，AMP 任务另移植了 TienKung 的相位步态（frc/spd/support）与接触力子步平均。
- **左右对称增强**：内置矢状面镜像数据增强（rsl_rl symmetry），提升步态对称性与样本效率。
- **端到端 sim2sim → sim2real**：MuJoCo 独立回放脚本做部署前验证；ROS 2 部署栈把导出的 `policy` 直接跑上真机，训练/仿真/实机三方相位对齐。
- **单一真源部署**：部署只需在一个 yaml 里填 `run` 目录名，模型的默认站姿 / 观测顺序与 scale / history / action_scale / 步态周期 / PD 增益全部从训练 run 的 `params/deploy.yaml` 自动读取。

---

## 📦 环境与安装

**前置依赖**（本项目不含 Isaac Lab / Isaac Sim 本体）：

1. 按官方[安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)装好 Isaac Lab（推荐 conda）。
2. Python 3.10 / 3.11；训练用到 `rsl_rl`，sim2sim 用到 `mujoco`、`pynput`。

**安装本扩展**（用装有 Isaac Lab 的解释器，以可编辑模式安装）：

```bash
git clone git@github.com:katazen/legs_rl_lab.git
cd legs_rl_lab

# 若 Isaac Lab 不在 conda/venv 里，把 python 换成 'PATH_TO/isaaclab.sh -p'
python -m pip install -e source/legs_rl_lab
```

> ⚠️ 机器人 USD 不入库（见 `.gitignore`，`nlegs` 的 MJCF/USD 资产有例外已入库）。`legs` 需在本地先由 MJCF 生成 USD（`assets/legs_URDF/`）；`nlegs` 资产在 `assets/legs_narrow/`。资产配置里的 `usd_path` 为绝对路径，克隆到别的机器后请改成本机实际路径。

---

## 🚀 快速开始（训练 / 回放）

> 下面命令里的 `python` 均指“装有 Isaac Lab 的解释器”。若不在 conda/venv，请替换为 `FULL_PATH_TO/isaaclab.sh -p`。

```bash
# 训练：窄本体速度跟踪，4096 环境，无头模式
python scripts/rsl_rl/train.py --task nlegs --headless --num_envs 4096

# 训练：AMP 走路（判别器 + 专家数据自动加载）
python scripts/rsl_rl/train.py --task nlegs_amp --headless --num_envs 4096

# 回放 / 评估已训练策略（少量环境、可实时观看）
python scripts/rsl_rl/play.py --task nlegs --num_envs 32 --real-time

# 冒烟测试：确认环境能正常起（零动作 / 随机动作）
python scripts/zero_agent.py --task nlegs --num_envs 16
python scripts/random_agent.py --task nlegs --num_envs 16
```

常用训练参数：`--task {legs,legs_dr,nlegs,nlegs_amp,g1qie}`、`--num_envs`、`--max_iterations`、`--seed`、`--headless`、`--video`（录制训练视频）。

训练产物默认写到 `logs/rsl_rl/<experiment_name>/<时间戳>/`：
- `params/env.yaml` —— 完整环境配置（含 `gait` 步态参数）。
- `params/deploy.yaml` —— 部署单一真源（默认站姿 / 观测规格 / action_scale / 步态周期 / PD 增益）。
- `exported/policy.pt`（及 `policy.onnx`，若导出）—— 推理模型。

### AMP 走路（`nlegs_amp`）

AMP 相关约定见项目记忆与代码，核心：
- **数据**：`scripts/amp/convert_tienkung_motion.py` 把 TienKung `walk.txt` 转成 30 维专家轨迹
  （关节位置 12 + 关节速度 12 + 左右脚 base 系 xyz 6），存到 `tasks/amp_task/datasets/motion_amp_expert/nlegs_walk.txt`；
  `scripts/amp/replay_amp_motion.py` 可在 MuJoCo 里回放校验。
- **算法**：`source/legs_rl_lab/legs_rl_lab/amp/` 里的 `AMPPPO` 最小化继承 rsl_rl 5.0.1 的 PPO，判别器用独立 Adam；
  AMP 观测走名为 `"amp"` 的观测组，靠 `obs["amp"]` 取用，不改 `train.py`。
- **任务**：`tasks/amp_task/task/nlegs/nlegs_env_cfg.py` 自包含（scene / 三组观测 policy+critic+amp / 奖励移植自 TienKung walk_cfg）。

### sim2sim（MuJoCo 部署前验证）

```bash
python source/legs_rl_lab/legs_rl_lab/tasks/legs_task/task/nlegs/sim2sim.py
```

> sim2sim 需要窄本体的 scene xml 与 nlegs 导出的 `policy`，脚本顶部 `SimToSimCfg` 里配置模型/网络路径。相位按 `elapsed_time / period` 推进，与实机部署节点逐位对齐。

---

## 🤖 实机部署（ROS 2 Humble）

部署代码全部在 `deploy/`，面向 **12-DOF 双足**（2 腿 × 6 关节：1 髋pitch · 2 髋roll · 3 髋yaw · 4 膝 · 5 踝pitch · 6 踝roll）。整栈分三个 ROS 2 节点：

| 工作区 | 节点 | 职责 |
|--------|------|------|
| `deploy/imu_ws`     | `wit_ros2_imu`      | 维特 IMU 驱动（含 rviz 可视化），发布姿态/角速度 |
| `deploy/control_ws` | `armcontrol`        | 电机驱动，订阅关节位置指令 `/dog_joint_pos`，PD 由训练 run 同步 |
| `deploy/rl_real_py` | `rl_real_common`    | RL 策略节点：读观测 → 推理 → 下发目标位置；键盘/手柄控速 |

### 单一真源：只改一个 `run`

部署时**通常只改** `deploy/rl_real_py/configs/common.yaml` 里的 `run`（训练时间戳目录名）与 `logs_root`：

```yaml
run: 2026-07-30_18-18-58
logs_root: logs/rsl_rl/nlegs
```

模型侧参数（默认站姿、观测顺序+scale、history、action_scale、步态周期、step_dt、PD 增益）全部从 `<logs_root>/<run>/params/deploy.yaml` 自动读取；策略从 `<run>/exported/policy.onnx`（优先）或 `policy.pt` 加载；实机数据自动存到 `<run>/sim2real/<时间>.csv`。`common.yaml` 里其余项是**硬件相关**部署参数（下发率、EMA 平滑、关节顺序映射、安全限位、看门狗、指令零偏、键盘/手柄配置）。

### 编译

各工作区自行 `colcon build`（脚本不代编译）。ROS 2 Humble 可用 `deploy/install_ros2_humble.sh` 参考安装。

```bash
source /opt/ros/humble/setup.bash
cd deploy/imu_ws     && colcon build && cd -
cd deploy/control_ws && colcon build && cd -
cd deploy/rl_real_py && colcon build && cd -
```

### 一键启停

```bash
# 启动：IMU → armcontrol（先由 sync_pd.py 从 deploy.yaml 同步 PD）→ RL 策略
# 各开一个 gnome-terminal 窗口；RL 窗口是真终端，键盘可用
cd deploy && ./start_real.sh
```

`sync_pd.py` 在启动 `armcontrol` 前，把训练 run 的 `deploy.yaml` 里 `stiffness/damping` 换算顺序后写入 armcontrol 参数 yaml —— **PD 增益与训练严格一致，避免手改漂移**。

流程：上电后缓慢进准备姿态（`prepare_time` 秒 smoothstep）→ 站立保持 → 在 RL 窗口按 `P` 开始行走。

```bash
# 关闭：RL → armcontrol → IMU 依次优雅停，兜底强杀
cd deploy && ./stop_real.sh
```

> ⚠️ **安全**：`armcontrol` 退出可能让电机失力，关闭前请先扶稳 / 挂好机器人。

### 遥控

| 输入 | 说明 |
|------|------|
| **键盘**（RL 窗口）| `W/S` vx±，`A/D` vy±，`Q/E` yaw±（累加式，每按一下加/减 `step`）；`空格` 清零；`P` 行走/暂停；`R` 复位 |
| **手柄** | 满杆对应 `cmd_clip=[vx,vy,wz]`；`A`=行走 `B`=停 `X`=复位；`deadzone` 死区；松开 `ctrl_timeout` 秒归零 |

安全机制：数据新鲜度看门狗（关节/IMU 超 `state_timeout` 没更新则冻结指令，不拿过期观测推理）、发布安全限位、下发目标 EMA 平滑（`target_ema_alpha`，抑制推理 50Hz 与发布 200Hz 之间的阶梯抖动）。若零指令下持续漂移，可用 `cmd_bias` 做指令零偏修正。

### 系统辨识（`deploy/sysid`）

电机/关节的激励-辨识与 sim2real 对比工具：`excite_record.py`（激励采数）、`fit_actuator.py`（拟合电机模型）、`analyze_stepsine.py` / `bode.py`（阶跃/正弦/频响分析）、`static_zero_check.py`（静态零位/倾斜检查）、`goto_zero.py`（平滑回零）、以及 `src/plot_*` 一组踝关节 sim/real 对比绘图脚本。背景与结论见 `deploy/SYSID_HANDOFF.md`。

---

## 📁 项目结构

```
legs_rl_lab/
├── scripts/
│   ├── rsl_rl/               # train.py / play.py / cli_args.py
│   ├── amp/                  # AMP 数据转换与 MuJoCo 回放
│   └── list_envs.py  zero_agent.py  random_agent.py
├── deploy/                   # ROS 2 实机部署栈
│   ├── imu_ws/               # IMU 驱动工作区
│   ├── control_ws/           # armcontrol 电机驱动工作区
│   ├── rl_real_py/           # RL 策略节点 + configs/common.yaml
│   ├── sysid/                # 系统辨识与 sim/real 对比
│   ├── sync_pd.py            # 从 deploy.yaml 同步 PD 到 armcontrol
│   └── start_real.sh  stop_real.sh
└── source/legs_rl_lab/legs_rl_lab/
    ├── amp/                  # in-repo AMPPPO（继承 rsl_rl 5.0.1）+ 判别器
    ├── assets/
    │   ├── legs_URDF/        # legs（A1 双腿）MJCF/STL/资产配置
    │   └── legs_narrow/      # nlegs（窄本体）MJCF/URDF/资产配置(nlegs.py)
    └── tasks/
        ├── legs_task/        # 速度跟踪任务集（legs / legs_dr / nlegs …）
        │   ├── mdp/          # rewards / observations / gait / symmetry ...
        │   ├── agents/       # rsl_rl PPO 配置
        │   └── task/{legs,nlegs}/
        ├── amp_task/         # AMP 走路任务（自包含 nlegs_amp）
        │   ├── mdp/  agents/  datasets/  task/nlegs/
        └── g1_task/          # 任务 "g1qie"
```

---

## 🧩 任务一览

| Task id       | 机器人 | 说明 |
|---------------|--------|------|
| `legs`        | A1 双腿原型 | 速度跟踪 locomotion，脚间距 0.36 m |
| `legs_dr`     | A1 双腿原型 | `legs` + 加强域随机化（质量/COM/PD/关节参数） |
| `legs_static` | A1 双腿原型 | 站立保持变体 |
| `nlegs`       | A1 窄本体 | 与 `legs` 共用全部 mdp/agents 与 env 配置，仅机器人资产（脚间距 ~0.22 m）和 `feet_y_distance` 目标间距不同（env_cfg 子类化继承 legs） |
| `nlegs_static`| A1 窄本体 | 窄本体站立保持变体 |
| `nlegs_amp`   | A1 窄本体 | **AMP 走路**：自包含 env_cfg（policy/critic/amp 三组观测），奖励移植 TienKung walk_cfg |
| `g1qie`       | G1 | G1 相关任务 |

> 实机部署当前以 **nlegs** 系为主（`common.yaml` 的 `logs_root: logs/rsl_rl/nlegs`）。

---

## 🛠️ 开发

代码风格用 ruff（配置见根目录 `pyproject.toml`，line-length 120，目标 py310）。可选装 pre-commit 自动格式化：

```bash
pip install pre-commit
pre-commit run --all-files
```

**VSCode 索引**：若 Pylance 找不到扩展模块，在 `.vscode/settings.json` 的 `python.analysis.extraPaths` 里加上 `source/legs_rl_lab` 的路径；若 Pylance 因索引过多崩溃，反过来注释掉一些用不到的 `omni.*` 包路径。

---

## 📝 License / 致谢

- 训练框架基于 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 扩展模板，源码文件头部保留其 SPDX 许可声明，Python 包在 `setup.py` 中声明为 Apache-2.0。
- AMP 走路的专家动作来自 [TienKung-Lab](https://github.com/) 的 walk 轨迹重定向。

使用前请以各源码/数据中的实际许可声明为准。
