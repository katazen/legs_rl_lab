# 2026-10-26 实机待测策略

从机器人终端逐个启动。每次切换前先结束上一套控制节点。以下命令会启动实机部署；本次同步只做无电机输出的加载预检。

原 common.yaml 和已测试模型保留。以下独立配置沿用机器人当前设置，仅更换 tasks.walk；启动脚本从对应 params/deploy.yaml 同步 PD。

## 2026-10-22_16-46-40

新导入，从头训练；checkpoint编号未附带

```bash
bash /home/woan/workspace/legs_rl_lab/deploy/start_real.sh --config /home/woan/workspace/legs_rl_lab/deploy/rl_real_py/configs/experiments_20261026/body_164640.yaml
```

## 2026-10-22_17-38-10

新导入，resume训练；checkpoint编号未附带

```bash
bash /home/woan/workspace/legs_rl_lab/deploy/start_real.sh --config /home/woan/workspace/legs_rl_lab/deploy/rl_real_py/configs/experiments_20261026/body_173810.yaml
```

## 2026-10-22_17-38-36_ckpt49999

本地49999；踝pitch Kp200/Kd5

```bash
bash /home/woan/workspace/legs_rl_lab/deploy/start_real.sh --config /home/woan/workspace/legs_rl_lab/deploy/rl_real_py/configs/experiments_20261026/body_173836_ckpt49999.yaml
```

## 2026-10-22_12-24-19_ckpt30800

本地30800；踝pitch Kp40/Kd2

```bash
bash /home/woan/workspace/legs_rl_lab/deploy/start_real.sh --config /home/woan/workspace/legs_rl_lab/deploy/rl_real_py/configs/experiments_20261026/body_122419_ckpt30800.yaml
```
