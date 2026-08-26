"""engineai 风格的 AMP 扩展 (对照本仓已有的 amp/ 模块)。

与 amp/ 的区别 —— 完全照搬 EngineAI 的策略:
  - 时序范式: history-window(默认 5 帧堆叠), 而非 (s, s') transition pair。
  - 判别器输入: 每帧 [joint_pos*9, base_lin_vel_b*7]; nlegs 版 frame_dim = 12+3 = 15,
    5 帧堆叠 -> 判别器输入 75 维。
  - 归一化: 判别器内置 rsl_rl EmpiricalNormalization(逐帧), 而非外置 RunningMeanStd。
  - style reward: 加性 total = task + 0.01*weight*style, 而非 lerp 混合。
  - 判别器每 4 次 PPO update 训练一次, 直接用 storage["amp"] 采样(无 replay buffer)。
  - LSGAN: expert->+1, policy->-1, grad penalty(lambda=10) 把 grad norm 压到 0。

接线方式同 amp/: env 里加 "amp" 观测组 -> obs["amp"]; agent cfg 顶层暴露
frame_dim/frame_length/discriminator_hidden_dims/frame_normalization/dataset_path,
algorithm.class_name 指向本模块的 AMPPPO, 由 construct_algorithm 构建判别器 + 专家数据。
"""

from .amp_ppo import AMPPPO
from .discriminator import Discriminator
from .data_loader import AMPDataLoader

__all__ = ["AMPPPO", "Discriminator", "AMPDataLoader"]
