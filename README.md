# legs_rl_lab

> 基于 [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) 的 12-DOF 双足机器人（nlegs）强化学习工程，覆盖 rsl_rl PPO 训练 → MuJoCo sim2sim 验证 → ROS 2 实机部署的完整流程。

![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.5%20%7C%205.x-76b900)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue)
![RL](https://img.shields.io/badge/RL-rsl__rl%20PPO-orange)
![Deploy](https://img.shields.io/badge/Deploy-ROS%202%20Humble-22314e)
[![Stars](https://img.shields.io/github/stars/katazen/legs_rl_lab?style=social)](https://github.com/katazen/legs_rl_lab)

---

## 📦 项目内容

- **行走任务** `tasks/nlegs_task`：`nlegs_flat` 平地速度跟踪；`nlegs_flat_static` 增加零速站稳；`nlegs_flat_crouch` 蹲姿复位；`nlegs_rough` rough 地形（台阶 ≤12 cm、坡 ≤17°，带难度课程）。行走策略盲走，仅使用本体观测。
- **动作跟踪任务** `tasks/mimic_task`：`nlegs_mimic_crouch` 跟踪下蹲 v3，`nlegs_mimic_stand` 跟踪起身 v2；当前参考数据及元数据随仓库同步，见下方数据集说明。
- **机器人资产** `assets/nlegs`：MJCF / USD / STL 全部入库，`usd_path` 相对包内解析，克隆即用；执行器用自定义 `DelayedDCMotorCfg`（通信延迟 + 转矩-转速滚降，参数来自实机辨识）。
- **sim2sim** `task/flat/sim2sim.py`：MuJoCo 独立回放训练好的策略做部署前验证，配置全部读训练 run 的 `deploy.yaml`。
- **实机部署** `deploy/`：ROS 2 Humble 三节点（IMU / 电机驱动 / RL 策略），同一入口切换走路、下蹲、起身；在 `common.yaml` 集中选择三份模型。

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

# 训练：最新下蹲 / 起身参考动作
python scripts/rsl_rl/train.py --task nlegs_mimic_crouch --headless --num_envs 4096
python scripts/rsl_rl/train.py --task nlegs_mimic_stand --headless --num_envs 4096

# 回放 / 评估已训练策略（少量环境、可实时观看，同时导出 exported/policy.pt）
python scripts/rsl_rl/play.py --task nlegs_flat --num_envs 32 --real-time

# 冒烟测试：确认环境能正常起（零动作 / 随机动作）
python tests/zero_agent.py --task nlegs_flat --num_envs 16
python tests/random_agent.py --task nlegs_flat --num_envs 16
```

常用训练参数：`--task`（上列任务名）、`--num_envs`、`--max_iterations`、`--seed`、`--headless`、`--video`（录制训练视频）。`nlegs_rough` 默认将 PPO 数据分为 16 个小批次，以降低 4096 环境更新时的显存峰值。

测试目录（`tests/`、各层 `test/`）、独立诊断/分析脚本和 `deploy/tools/` 仅供本地使用，不随 Git 同步；上面的冒烟测试命令需要本地保留这些脚本。全部 `logs/`（含参数、导出模型和检查点）也不入库，换机器回放或部署需单独复制所需 run 的 `params/` 和 `exported/`。

训练产物默认写到 `logs/rsl_rl/<experiment_name>/<时间戳>/`：
- `params/env.yaml` —— 完整环境配置（含 `gait` 步态参数）。
- `params/deploy.yaml` —— 部署单一真源（默认站姿 / 观测规格 / action_scale / 步态周期 / PD 增益 / 执行器分组与延迟）。
- `exported/policy.pt`（及 `policy.onnx`，若导出）—— 推理模型。

### 当前使用的数据集

| 数据目录 | 用途 | 帧数 / 频率 / 时长 |
|---|---|---|
| [`datasets/stand_to_crouch_v3/`](datasets/stand_to_crouch_v3/) | 当前下蹲参考，对应 `nlegs_mimic_crouch` | 341 帧 / 100 Hz / 3.40 s |
| [`datasets/crouch_to_stand_v2/`](datasets/crouch_to_stand_v2/) | 当前起身参考，对应 `nlegs_mimic_stand` | 336 帧 / 100 Hz / 3.35 s |
| [`datasets/stand_to_crouch_v2/`](datasets/stand_to_crouch_v2/) | 起身生成器引用的历史源数据与溯源记录，不作为当前下蹲训练输入 | 341 帧 / 100 Hz / 3.40 s |

两个当前版本均包含 `motion.npz`（关节、机身、足端及接触计划等原始参考）、`mimic_motion.npz`（训练所需的全身参考格式）和 `metadata.json`（时序、实测参数、模型/源文件 SHA256 与生成检查结果）。历史源目录只同步 `motion.npz` 和 `metadata.json`。

训练配置直接读取包内参考文件，克隆后无需重新生成或转换；它们与数据目录中的 `mimic_motion.npz` 完全一致：

- 下蹲：[`tasks/mimic_task/task/nlegs_crouch/motions/stand_to_crouch_v3.npz`](source/legs_rl_lab/legs_rl_lab/tasks/mimic_task/task/nlegs_crouch/motions/stand_to_crouch_v3.npz)。
- 起身：[`tasks/mimic_task/task/nlegs_stand/motions/crouch_to_stand_v2.npz`](source/legs_rl_lab/legs_rl_lab/tasks/mimic_task/task/nlegs_stand/motions/crouch_to_stand_v2.npz)。

下蹲末帧与起身首帧的关节角、机身位置和姿态一致；起身任务在 3.35 s 参考结束后额外训练保持 5 s。这些 NPZ 是运动学参考，策略权重仍需训练和导出。

#### 参考动作预览

下方动图展示两个 Mimic 任务使用的参考数据运动学回放；点击动图或 MP4 链接查看原始视频。画面左右分别为斜前方和侧视角。

**下蹲 v3 · `nlegs_mimic_crouch` · 3.40 s** — [原始 MP4](datasets/stand_to_crouch_v3/preview.mp4)

[![下蹲 v3 参考动作：站立、双脚向外迈步、下蹲并保持](datasets/stand_to_crouch_v3/preview.gif)](datasets/stand_to_crouch_v3/preview.mp4)

**起身 v2 · `nlegs_mimic_stand` · 3.35 s** — [原始 MP4](datasets/crouch_to_stand_v2/preview.mp4)

[![起身 v2 参考动作：蹲姿起身、依次收脚、站立并保持](datasets/crouch_to_stand_v2/preview.gif)](datasets/crouch_to_stand_v2/preview.mp4)

`.gitignore` 放行上述 8 个数据文件及两段动作的 MP4/GIF 预览；其他预览图片/HTML、历史数据、测试/调试脚本和全部 `logs/` 继续忽略。元数据中的绝对模型路径、`crouch_snapshot` 是生成时的溯源记录；原始实测快照保留在本地，重新生成实测动作时需要它，使用已同步的参考训练不依赖它。

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

### 一份公共配置，三份模型

部署时在 `deploy/rl_real_py/configs/common.yaml` 的 `tasks` 中选择模型，路径相对仓库根目录：

```yaml
tasks:
  walk: logs/rsl_rl/nlegs_flat_static/2026-09-16_11-41-05
  crouch: logs/rsl_rl/nlegs_mimic_crouch/2026-09-17_16-42-11
  rise: logs/rsl_rl/nlegs_mimic_stand/2026-09-17_15-40-44
```

各模型的观测、动作、PD 和参考参数仍从各自 `params/deploy.yaml` 读取，不能合并成一份；策略优先加载 `exported/policy.onnx`，否则加载 `policy.pt`。日志按当前模型存入其 `sim2real/`。公共硬件限位、输入配置和任务切换阈值只在 `common.yaml` 维护，不再叠加专用 YAML。

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
cd deploy
./start_real.sh --check-only  # 先离线预检：不启动驱动、不修改 PD
./start_real.sh              # 现场保护与硬件检查完成后再运行
```

`sync_pd.py` 在启动 `armcontrol` 前，把训练 run 的 `deploy.yaml` 里 `stiffness/damping` 换算顺序后写入 armcontrol 参数 yaml —— **PD 增益与训练严格一致，避免手改漂移**。

流程：保持当前姿态 → 识别站姿/蹲姿就绪 → 按键选择任务。旧单策略入口和上电自动回站立已退役；`P` 现在是中断，不是启动。

```bash
# 关闭：RL → armcontrol → IMU 依次优雅停，兜底强杀
cd deploy && ./stop_real.sh
```

> ⚠️ **安全**：`armcontrol` 退出可能让电机失力，关闭前请先扶稳 / 挂好机器人。

### 遥控

| 输入 | 说明 |
|------|------|
| **键盘**（RL 窗口）| `1/2/3` 走路/下蹲/起身；`W/S A/D Q/E` 控速；`0` 停步减速，再 `Enter` 确认接地收脚；`P` 中断；`R` 重新验收 |
| **手柄** | `LB+A/X/Y` 走路/下蹲/起身；`Start` 停步，再按一次确认接地；`B` 中断；`Back` 重新验收 |

保留硬件与任务限位、反馈/推理超时保护和故障锁存。实机没有足底接触反馈，任务键和停步确认需要操作者确认落地；`P/B` 不是断电急停。完整流程见 [部署说明](deploy/README.md)。

---

## 📁 项目结构

```
legs_rl_lab/
├── datasets/                      # 仅同步当前下蹲 v3、起身 v2 与所需 v2 源数据
├── scripts/
│   ├── rsl_rl/                    # train.py / play.py / cli_args.py
│   ├── sim2sim.py                 # 统一三任务仿真入口
│   └── *.py                       # 任务列表、动作/姿态生成、数据准备与分析
├── tests/                         # 本地离线/仿真检查，不入库
├── deploy/                        # ROS 2 实机部署栈
│   ├── imu_ws/                    # IMU 驱动工作区
│   ├── control_ws/                # armcontrol 电机驱动工作区
│   ├── rl_real_py/                # RL 策略节点 + configs/common.yaml
│   │   └── test/                  # 本地 ROS 离线回归，不入库
│   ├── tools/                     # 本地实验/诊断工具，不入库
│   ├── pose_logs/                 # 本地实验记录，不入库；工具迁移不移动数据
│   ├── sync_pd.py                 # 从 deploy.yaml 同步 PD 到 armcontrol
│   └── start_real.sh  stop_real.sh
└── source/legs_rl_lab/legs_rl_lab/
    ├── actuators/                 # DelayedDCMotorCfg（延迟 + 转矩-转速滚降）
    ├── assets/nlegs/              # 自包含资产: nlegs.py(ArticulationCfg) + mjcf/ + meshes/
    │   ├── mjcf/nlegs.xml         #   MJCF 源（nlegs_scene.xml 供 sim2sim）
    │   └── mjcf/nlegs_limit/      #   当前 Isaac 用 USD（已入库, 相对路径加载）
    ├── tasks/nlegs_task/          # 自包含任务包
    │   ├── agents/                #   rsl_rl PPO 配置（flat / rough）
    │   ├── mdp/                   #   rewards / observations / events / gait / symmetry ...
    │   └── task/
    │       ├── flat/              #   nlegs_flat: 全量展开 env cfg + sim2sim.py
    │       └── rough/             #   nlegs_rough: 继承 flat, 地形生成器 + 课程
    ├── tasks/mimic_task/          # 下蹲 / 起身动作跟踪
    │   └── task/                 # nlegs_crouch / nlegs_stand，含训练参考 motions/
    └── utils/                     # parser_cfg / export_deploy_cfg（生成 deploy.yaml）
```

`tests/check_*sim2sim.py` 使用训练环境中的 MuJoCo/PyTorch；`check_stand_mimic_cfg.py` 不启动 Isaac Sim；其余任务检查通常需要 Isaac Lab。示例：

```bash
python tests/check_multi_sim2sim.py
python tests/check_flat_static.py --headless --device cuda:0
# ROS 模块离线回归，使用系统 Python；不启动电机控制节点
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/deploy/rl_real_py:$PYTHONPATH" /usr/bin/python3 -m pytest deploy/rl_real_py/test -q
```

---

## 📝 License

训练框架基于 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 扩展模板，源码文件头部保留其 SPDX 许可声明，Python 包在 `setup.py` 中声明为 Apache-2.0。使用前请以各源码/数据中的实际许可声明为准。
