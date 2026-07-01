# legs_rl_lab

> 基于 [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) 的双足腿式机器人强化学习工程：用 PPO（rsl_rl）在 GPU 并行仿真中训练腿式本体的速度跟踪 locomotion 策略，内置步态时钟、左右对称数据增强与 MuJoCo sim2sim 部署验证。

![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.5%20%7C%205.x-76b900)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue)
![RL](https://img.shields.io/badge/RL-rsl__rl%20PPO-orange)
[![Stars](https://img.shields.io/github/stars/katazen/legs_rl_lab?style=social)](https://github.com/katazen/legs_rl_lab)

**适合谁用**：正在用 Isaac Lab / Isaac Sim 做腿足运动控制（locomotion）RL 的研究者与工程师。你需要一台带 NVIDIA GPU 的机器，并已装好 Isaac Lab。

---

## ✨ 项目亮点

- **开箱即用的多任务**：`legs`（A1 双腿原型，脚间距 0.36 m）、`nlegs`（窄本体变体，脚间距 0.2 m）、`g1qie`（G1 相关任务），共用一套 MDP 组件。
- **参数化步态时钟**：步态周期 / 支撑相占比 / 左右相位偏移收敛到 `GaitCfg`，会随 `env.yaml` 落盘，训练与部署可复现（`nlegs` 已去掉自定义 env，相位随机化改由 reset 事件实现）。
- **面向真机的奖励设计**：速度跟踪 + 姿态、足部间距 / 离地高度 / 打滑 / 接触力 / 触地相位匹配等成套奖励项。
- **左右对称增强**：内置矢状面镜像的数据增强（rsl_rl symmetry），提升步态对称性与样本效率。
- **sim2sim 验证**：提供 MuJoCo 独立回放脚本，训练完可先在 MuJoCo 里检查策略再上真机。
- **独立扩展结构**：基于 Isaac Lab 扩展模板，在核心仓库之外独立开发。

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

> ⚠️ 机器人 USD 不入库（见 `.gitignore`）。`legs` 需要 `A1_legs_V2_mjcf` 的 USD；`nlegs` 需要先把 `source/legs_rl_lab/legs_rl_lab/assets/legs_URDF/mjcf/A1_legs_V2_narrow_mjcf.xml` 转成 USD，并在 `assets/legs_URDF/nlegs.py` 里填好 `NLEGS_USD_PATH`（当前为 `TODO` 占位）。

---

## 🚀 快速开始

> 下面命令里的 `python` 均指“装有 Isaac Lab 的解释器”。若不在 conda/venv，请替换为 `FULL_PATH_TO/isaaclab.sh -p`。

```bash
# 训练：窄本体变体，4096 环境，无头模式
python scripts/rsl_rl/train.py --task nlegs --headless --num_envs 4096

# 训练：原始双腿任务
python scripts/rsl_rl/train.py --task legs --headless

# 回放 / 评估已训练策略（少量环境、可实时观看）
python scripts/rsl_rl/play.py --task nlegs --num_envs 32 --real-time

# 冒烟测试：确认环境能正常起（零动作 / 随机动作）
python scripts/zero_agent.py --task nlegs --num_envs 16
python scripts/random_agent.py --task nlegs --num_envs 16
```

常用训练参数：`--task {legs,nlegs,g1qie}`、`--num_envs`、`--max_iterations`、`--seed`、`--headless`、`--video`（录制训练视频）。

训练产物默认写到 `logs/rsl_rl/<experiment_name>/<时间戳>/`，其中 `params/env.yaml` 会记录完整环境配置（含 `gait` 步态参数），`exported/policy.pt` 为导出的推理模型。

### sim2sim（MuJoCo 部署验证）

```bash
# 编辑脚本顶部 SimToSimCfg.path 里的 model_path / xml 路径后运行
python source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/nlegs/sim2sim.py
```

> `nlegs` 的 sim2sim 需要窄本体的 scene xml 与 nlegs 导出的 `policy.pt`，脚本里已用 `TODO` 标注待填项。

---

## 📁 项目结构

```
legs_rl_lab/
├── scripts/rsl_rl/           # 训练 / 回放 / CLI 参数
│   ├── train.py  play.py  cli_args.py
├── scripts/                  # list_envs / zero_agent / random_agent 等工具
└── source/legs_rl_lab/legs_rl_lab/
    ├── assets/legs_URDF/     # A1 双腿机器人：MJCF + STL 网格 + 资产配置(legs.py / nlegs.py)
    └── tasks/
        ├── legs_task/        # 任务 "legs"：A1 双腿原型
        ├── nlegs_task/       # 任务 "nlegs"：窄本体变体（脚间距 0.2）
        │   ├── mdp/          # rewards / observations / gait / symmetry ...
        │   ├── agents/       # rsl_rl PPO 配置
        │   └── task/nlegs/   # 环境配置 + sim2sim
        └── g1_task/          # 任务 "g1qie"
```

每个任务通过 `gym.register` 暴露一个 id：`legs` / `nlegs` / `g1qie`。

---

## 🧩 任务一览

| Task id  | 机器人 | 说明 |
|----------|--------|------|
| `legs`   | A1 双腿原型 | 速度跟踪 locomotion，脚间距 0.36 m，带自定义 env 的相位时钟 |
| `nlegs`  | A1 窄本体变体 | 复刻 `legs`，本体换窄立方体、脚间距缩到 0.2 m；步态参数进 `GaitCfg`，相位随机化用 reset 事件（无自定义 env） |
| `g1qie`  | G1 | G1 相关任务 |

`legs` 与 `nlegs` 的差异集中在：机器人资产（USD）、`feet_y_distance` 目标间距（0.36 → 0.2）、以及步态参数的组织方式。

---

## 🛠️ 开发

代码风格用 ruff（配置见根目录 `pyproject.toml`，line-length 120，目标 py310）。可选装 pre-commit 自动格式化：

```bash
pip install pre-commit
pre-commit run --all-files
```

**VSCode 索引**：若 Pylance 找不到扩展模块，在 `.vscode/settings.json` 的 `python.analysis.extraPaths` 里加上 `source/legs_rl_lab` 的路径；若 Pylance 因索引过多崩溃，反过来注释掉一些用不到的 `omni.*` 包路径。

---

## ❓ 适用场景

- 双足 / 腿式机器人 locomotion 策略训练与快速迭代。
- 在同一套 MDP 组件下对比不同本体几何（如站宽）对步态的影响。
- 训练后经 MuJoCo sim2sim 做部署前验证。

---

## 📝 License / 致谢

本项目基于 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 的扩展模板构建，源码文件头部保留其 SPDX 许可声明，Python 包在 `setup.py` 中声明为 Apache-2.0。使用前请以源码中的实际许可声明为准。

> 说明：本 README 中标 `TODO` / “待填” 的部分（如 `nlegs` 的 USD 路径、sim2sim 的 scene xml 与 policy 路径）需在对应资产准备好后补全。
