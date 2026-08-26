# legs_rl_lab

> 基于 [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) 的 12-DOF 双足机器人（nlegs）强化学习工程，覆盖 rsl_rl PPO 训练 → MuJoCo sim2sim 验证 → ROS 2 实机部署的完整流程。

![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.5%20%7C%205.x-76b900)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue)
![RL](https://img.shields.io/badge/RL-rsl__rl%20PPO-orange)
![Deploy](https://img.shields.io/badge/Deploy-ROS%202%20Humble-22314e)
[![Stars](https://img.shields.io/github/stars/katazen/legs_rl_lab?style=social)](https://github.com/katazen/legs_rl_lab)

---

## 📦 项目内容

- **训练任务** `tasks/nlegs_task`：`nlegs_flat` 平地速度跟踪；`nlegs_rough` rough 地形（继承 flat，台阶 ≤12 cm、坡 ≤17°，带难度课程）。策略盲走（仅 IMU + 关节观测），可直接部署。
- **机器人资产** `assets/nlegs`：MJCF / USD / STL 全部入库，`usd_path` 相对包内解析，克隆即用；执行器用自定义 `DelayedDCMotorCfg`（通信延迟 + 转矩-转速滚降，参数来自实机辨识）。
- **sim2sim** `task/flat/sim2sim.py`：MuJoCo 独立回放训练好的策略做部署前验证，配置全部读训练 run 的 `deploy.yaml`。
- **实机部署** `deploy/`：ROS 2 Humble 三节点（IMU / 电机驱动 / RL 策略），键盘/手柄控速；部署参数从训练 run 的 `params/deploy.yaml` 自动读取，通常只需填一个 `run` 目录名。

---

## 🔧 环境与安装

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

---

## 🚀 快速开始（训练 / 回放）

> 下面命令里的 `python` 均指"装有 Isaac Lab 的解释器"。若不在 conda/venv，请替换为 `FULL_PATH_TO/isaaclab.sh -p`。

```bash
# 训练：平地速度跟踪，4096 环境，无头模式
python scripts/rsl_rl/train.py --task nlegs_flat --headless --num_envs 4096

# 训练：rough 地形（生成器地形 + 难度课程，奖励已按地形调整）
python scripts/rsl_rl/train.py --task nlegs_rough --headless --num_envs 4096

# 回放 / 评估已训练策略（少量环境、可实时观看，同时导出 exported/policy.pt）
python scripts/rsl_rl/play.py --task nlegs_flat --num_envs 32 --real-time

# 冒烟测试：确认环境能正常起（零动作 / 随机动作）
python scripts/zero_agent.py --task nlegs_flat --num_envs 16
python scripts/random_agent.py --task nlegs_flat --num_envs 16
```

常用训练参数：`--task {nlegs_flat,nlegs_rough}`、`--num_envs`、`--max_iterations`、`--seed`、`--headless`、`--video`（录制训练视频）。

训练产物默认写到 `logs/rsl_rl/<experiment_name>/<时间戳>/`（`nlegs_flat` / `nlegs_rough`）：
- `params/env.yaml` —— 完整环境配置（含 `gait` 步态参数）。
- `params/deploy.yaml` —— 部署单一真源（默认站姿 / 观测规格 / action_scale / 步态周期 / PD 增益 / 执行器分组与延迟）。
- `exported/policy.pt`（及 `policy.onnx`，若导出）—— 推理模型。

### sim2sim（MuJoCo 部署前验证）

```bash
python source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/flat/sim2sim.py [--run RUN] [--headless] [--save-data]
```

- **配置**：关节映射 / PD / 力矩模型 / action scale / 观测历史 / 命令范围全部从 `<run>/params/deploy.yaml` 读取；需要手改的只有脚本最前面的标注块（run 目录、场景 xml、键盘绑定、可视化参数）。
- **可视化**：世界原点画 RGB 三轴；机身上方画速度箭头（绿 = 命令、蓝 = 实际）。
- **遥控**：小键盘 `8/2` 前后、`4/6` 左右、`7/9` 转向；`--save-data` 结束时输出关节跟踪 CSV 与 RMSE 图到 `<run>/sim2sim/<时间戳>/`。
- 按文件路径直接运行（不要 `python -m` 走包导入，包初始化会拉起 Isaac Lab）。

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
run: 2026-08-26_13-57-34
logs_root: logs/rsl_rl/nlegs_flat
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

---

## 📁 项目结构

```
legs_rl_lab/
├── scripts/
│   ├── rsl_rl/                    # train.py / play.py / cli_args.py
│   └── list_envs.py  zero_agent.py  random_agent.py
├── deploy/                        # ROS 2 实机部署栈
│   ├── imu_ws/                    # IMU 驱动工作区
│   ├── control_ws/                # armcontrol 电机驱动工作区
│   ├── rl_real_py/                # RL 策略节点 + configs/common.yaml
│   ├── sync_pd.py                 # 从 deploy.yaml 同步 PD 到 armcontrol
│   └── start_real.sh  stop_real.sh
└── source/legs_rl_lab/legs_rl_lab/
    ├── actuators/                 # DelayedDCMotorCfg（延迟 + 转矩-转速滚降）
    ├── assets/nlegs/              # 自包含资产: nlegs.py(ArticulationCfg) + mjcf/ + meshes/
    │   ├── mjcf/nlegs.xml         #   MJCF 源（nlegs_scene.xml 供 sim2sim）
    │   └── mjcf/nlegs/nlegs.usd   #   Isaac 用 USD（已入库, 相对路径加载）
    ├── tasks/nlegs_task/          # 自包含任务包
    │   ├── agents/                #   rsl_rl PPO 配置（flat / rough）
    │   ├── mdp/                   #   rewards / observations / events / gait / symmetry ...
    │   └── task/
    │       ├── flat/              #   nlegs_flat: 全量展开 env cfg + sim2sim.py
    │       └── rough/             #   nlegs_rough: 继承 flat, 地形生成器 + 课程
    └── utils/                     # parser_cfg / export_deploy_cfg（生成 deploy.yaml）
```

---

## 📝 License

训练框架基于 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 扩展模板，源码文件头部保留其 SPDX 许可声明，Python 包在 `setup.py` 中声明为 Apache-2.0。使用前请以各源码/数据中的实际许可声明为准。
