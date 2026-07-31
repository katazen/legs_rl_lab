"""AMP (Adversarial Motion Priors) 扩展模块, 适配 pip 安装的 rsl_rl 5.0.1。

思路(相对 TienKung 的老 vendored rsl_rl 做了现代化):
  - rsl_rl 5.0.1 里 obs 是 TensorDict, 只要 env 里加一个 "amp" 观测组, runner 就能
    直接拿到 obs["amp"], 无需侧信道 / 无需自定义 runner。
  - AMPPPO 继承官方 PPO, 只重写 act / process_env_step / update / save / load 注入
    判别器逻辑; construct_algorithm 复用 PPO 的再挂上判别器组件。
  - 训练脚本 (scripts/rsl_rl/train.py) 无需改动: 基类 OnPolicyRunner 会从
    agent_cfg.algorithm.class_name 解析出 AMPPPO。
"""

from .amp_ppo import AMPPPO
from .discriminator import Discriminator
from .motion_loader import AMPLoader
from .normalizer import Normalizer
from .replay_buffer import ReplayBuffer

__all__ = ["AMPPPO", "Discriminator", "AMPLoader", "Normalizer", "ReplayBuffer"]
