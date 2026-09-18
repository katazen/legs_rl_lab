# 同一行走策略从下蹲状态起步

`nlegs_flat_crouch` 使用同一个策略学习下蹲起步、恢复站立和继续走路。原 flat/rough 与部署代码不变。

## Reset 逻辑

1. 固定概率选择姿态组：40% 限位蹲姿附近、30% 中间蹲姿、30% 直立附近。
2. 在组内均匀抽取 JSON 的一行，同时使用该行的身体高度、四元数和 12 个关节角。
3. 整体随机旋转 yaw，根节点和关节初速度置零，然后交给正常行走策略。

从第一步开始就开放三组全部状态，没有姿态或速度课程。训练和评估使用相同的 reset 分布。
`stand_probability` 和 `crouch_probability` 分别控制直立与限位附近的概率，剩余概率分配给中间蹲姿。
组内均匀抽取，组概率不受该组在 JSON 中的条数影响。

行走命令沿用 flat：前后 -0.3~0.5 m/s、横向 ±0.3 m/s、转向 ±0.5 rad/s，10% 为零速度命令。
原有奖励、推扰、镜像增强和其他 DR 保留。没有强制起身轨迹或策略切换；动作基准不变，
零动作依然指向默认站立角，不等于保持下蹲。

## JSON 是什么

`assets/nlegs/crouch_pose_bank.json` 是离线生成的有效初始状态集合，不是运动轨迹或策略输出。

- `posture`：`stand`、`intermediate` 或 `crouch`。
- `pitch_deg`：身体前倾角，生成范围 0~24°，间隔 1°。
- `limit_distance_deg`：十个受限关节距下蹲方向限位的最大角度差，不含双踝 roll。
- `height`、`quat_wxyz`、`joint_pos`：相互匹配的身体高度、朝向和关节角。
- `joint_names`：角度排列顺序；加载时按仿真实际关节名称重排。

每个倾角有 16 个样本，共 400 个状态。下蹲参考角和所有硬限位唯一取自 `nlegs_limit.xml`，
不使用历史实机角度：左右 hip pitch、ankle pitch 取下限，左右 knee 取上限，
左 hip roll/yaw 取上限、右 hip roll/yaw 取下限。ankle roll 参与贴地求解，不固定在限位上。

在默认站立角到上述 XML 限位参考的插值附近增加 ±0.03 rad 扰动，再联合求解贴地姿态。
这是优化前的参考扰动，最终关节修正不限定在 ±0.03 rad 内。完成求解后才分组：
十个受限关节均距各自下蹲限位 ≤3° 的归入 `crouch`；否则，均距默认站立角 ≤6° 的归入
`stand`；其余归入 `intermediate`。因此限位组不是仅按身体倾角贴标签。

每个状态检查关节限位、双脚朝向与高度、支撑区域及现有碰撞体。足底间隙为 2 mm；
检查覆盖额外机身质量 1~4 kg 下的静态支撑裕量。
这些是 XML 限位内的模型可行解，不保证十个关节全部同时顶住限位，也不代表动态恢复已验证。

下蹲训练及 Play 在 `FlatCrouchEnvCfg` 中单独覆盖模型路径，加载 `mjcf/nlegs_limit/nlegs_limit.usd`；
公共 `NLEGS_CFG` 和原 flat/rough 仍使用 `mjcf/nlegs/nlegs.usd`。
运行时检查 USD 的机械限位与姿态表一致，不再用姿态表覆盖 PhysX 限位。

`model_sha256` 是文件内容校验清单，不是模型加载配置。它记录姿态生成源 `nlegs_limit.xml`、
入口 `nlegs_limit.usd` 及其 `configuration/` 下的四个 USD 依赖，不再绑定旧 XML/USD。
修改 XML 后应先重新转换新 USD，再生成姿态表并运行仿真检查；哈希只能发现文件变化，
不能单独证明 XML 与 USD 的几何/物理一致。目前模型只有脚的碰撞几何，检查不能覆盖所有腿/机身自碰撞。
有限姿态表不覆盖整个连续空间；需要更多局部样本时可增加生成器的 `--variants`。

## 使用

```bash
conda activate unitree_lab
cd /home/woan/workspace/legs_rl_lab
python scripts/rsl_rl/train.py --task nlegs_flat_crouch --headless --num_envs 4096
```

检查或重新生成：

```bash
python scripts/generate_crouch_pose_bank.py --check
python tests/check_crouch_reset.py --headless --device cuda:0
python scripts/generate_crouch_pose_bank.py
```

只从限位附近出生，可在训练命令后加
`env.events.reset_crouch.params.stand_probability=0 env.events.reset_crouch.params.crouch_probability=1`。
模型保存于 `logs/rsl_rl/nlegs_flat_crouch/`；`*_smoke` 是短测试产物，不是可部署策略。

## MuJoCo sim2sim：限位下蹲按键起步

```bash
conda activate unitree_lab
python source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/flat/sim2sim_crouch.py --run 2026-09-09_18-13-14
```

脚本顶部 `RUN` 可以修改，也可以用 `--run` 指定其他训练目录。需要该目录下的
`params/deploy.yaml` 和 `exported/policy.pt`；没有导出策略时，先运行对应任务的 `scripts/rsl_rl/play.py`。
请按文件路径运行，避免包导入触发 Isaac Lab。此回放使用独立的 `nlegs_limit_scene.xml`，
不改变原 flat 的场景或回放脚本。

- 窗口打开时已经是限位下蹲。等待阶段物理、推理和步态时钟都暂停，不是通电保持或断电稳定性测试。
- 聚焦 MuJoCo 窗口，按小键盘 **5** 开始 RL；等待时长不计入 `--duration`。
- **8/2** 前后、**4/6** 左右、**7/9** 转向，与 flat 的命令增量和限幅相同；主键盘数字也支持。
- 再次按 5 不会重置姿态或相位。关闭窗口退出；`--save-data` 沿用 flat 的跟踪数据与绘图。

出生姿态不从 JSON 的“限位附近”组抽样，而是每次按当前 XML 重新求解：十个关节固定在端点，
只优化身体 roll/pitch 和双踝 roll，并检查足底倾斜、高度差和静态支撑裕量。
保留 2 mm 出生间隙。它不是额外的起身轨迹，启动后完全交给策略。
`default_joint_pos`、动作 offset、观测排列、执行器参数和延迟均沿用该 run 的导出配置。

当前 XML 求解结果（模型改变后以启动打印为准）：

| 仿真关节 | 左腿 / rad | 右腿 / rad |
|---|---:|---:|
| 1 hip pitch | -0.970000 | -0.970000 |
| 2 hip roll | 0.510000 | -0.540000 |
| 3 hip yaw | 0.500000 | -0.630000 |
| 4 knee | 1.140000 | 1.190000 |
| 5 ankle pitch | -0.400000 | -0.430000 |
| 6 ankle roll | -0.110986 | 0.026599 |

机身 roll/pitch/yaw 约为 `[0.6064°, 23.0374°, 0°]`，base 高度约 `0.498892 m`；
四元数 WXYZ 约为 `[0.97984581, 0.00518490, 0.19968524, -0.00105664]`。
仿真 L6/R6 是 ankle roll，对应实机每腿 CAN 7。
最大脚底倾角约 `0.2893°`、足底高度差约 `0.9989 mm`，不是数学上完全共面的解。
真正从十个端点起步比训练表中的限位附近更严格，回放能运行不代表策略已学会从该状态稳定恢复。

```bash
# 只计算初始姿态，不需要导出的策略或窗口
python source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/flat/sim2sim_crouch.py --pose-only

# 有界自动回放：显式跳过按键等待
python source/legs_rl_lab/legs_rl_lab/tasks/nlegs_task/task/flat/sim2sim_crouch.py --headless --auto-start --duration 10 --save-data

# 初姿态、等待/按键、首帧观测及真实策略短回放检查
python tests/check_crouch_sim2sim.py --run 2026-09-09_18-13-14
```
