# nlegs_rough：零速踏步

```bash
./legs_rl_lab.sh -t --task=nlegs_rough
./legs_rl_lab.sh -p --task=nlegs_rough
```

2026-09-19 更新。保留原 rough 的地形、课程、速度命令、0.6 s 步态周期、零速相位和踏步/抬脚/高度奖励；不加入 static 的站姿奖励或零速门控。需重新训练，旧权重不会自动适应新边界。

## 实测单侧限位

今天的实机记录中，膝、踝 pitch 的裁剪值与实测下蹲端点一致。但 rough 实机测试会拆除膝限位块，因此 **L4/R4 不加下蹲限位**，保留 USD 原有膝活动范围。`rough_env_cfg.py` 从 `CROUCH_LIMITS` 中取左右 1、2、3、5 号关节共 8 个单侧端点；另一端保留 USD 原值，踝 roll 不新增限制。rough_static 仍保留 10 个端点。

| 关节 | 限制方向 | rad |
|---|---|---:|
| L5 左踝 pitch | 下限 | -0.3763256275 |
| R5 右踝 pitch | 下限 | -0.4224841688 |

同一张表用于绝对目标角 clip 和 startup 写入 PhysX 物理关节边界，软限位奖励随之更新。XML/USD 不改写，也不从部署配置动态加载训练参数。左右端点不完全对称，因此关闭 rough 的镜像增强与镜像损失。

本次只更新训练任务。统一实机部署当前仍裁剪下蹲膝上限；新 rough 模型实测前需单独调整走路分支，不能直接沿用该膝裁剪，也不能解除 common 硬件边界。

## 辨识设置

| 项目 | 当前范围 |
|---|---|
| 膝目标延迟 | 1–6 步，即 5–30 ms |
| 踝 pitch 目标延迟 | 1–8 步，即 5–40 ms |
| 踝 roll 目标延迟 | 2–6 步，即 10–30 ms |
| 踝 pitch Kp / Kd 倍率 | 0.7–1.1 / 0.7–1.2 |
| 踝 roll Kp / Kd 倍率 | 0.85–1.1 / 0.5–1.0 |

保持名义膝 PD=250/5、踝 pitch=40/2、踝 roll=40/0.5；髋执行器、惯量、摩擦、力矩/速度上限不变。保留原接触摩擦、基座质量、编码器零偏、速度扰动随机化。这些范围是辨识支持的鲁棒性覆盖，不是已确定的物理常数；证据见 [辨识复核](README_static.md#2026-09-18-辨识复核哪些结论可用)。

## 检查

激活 `unitree_lab` 后：

```bash
python tests/check_rough_static.py --task nlegs_rough --headless --device cuda:0
python tests/check_rough_static.py --task nlegs_rough_static --headless --device cuda:0
```

检查 16 环境初始化与有限步运行、物理/软/目标边界、部署标定对应、未改另一端、随机化范围、零速相位继续运行和奖励不门控、导出精度，以及 flat 不受影响。检查通过不代表策略已训练好。

2026-09-19：上述两个任务的检查均通过；rough 膝目标无额外 clip、膝物理范围与原 USD 一致，资产哈希未变化。未启动正式训练或实机控制。
