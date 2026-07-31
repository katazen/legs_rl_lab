"""策略侧 AMP transition 的定长环形缓存。移植自 TienKung rsl_rl 的 ReplayBuffer。

判别器需要"策略产生的 (s, s') 转移"作为负样本。用一个比单次 rollout 大得多的缓存
(默认 10 万)混合近期多批 rollout, 稳定判别器训练。"""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim: int, buffer_size: int, device):
        self.states = torch.zeros(buffer_size, obs_dim, device=device)
        self.next_states = torch.zeros(buffer_size, obs_dim, device=device)
        self.buffer_size = buffer_size
        self.device = device
        self.step = 0
        self.num_samples = 0

    def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
        n = states.shape[0]
        if n == 0:
            return
        start = self.step
        end = self.step + n
        if end > self.buffer_size:  # 环形回绕
            self.states[start:self.buffer_size] = states[: self.buffer_size - start]
            self.next_states[start:self.buffer_size] = next_states[: self.buffer_size - start]
            self.states[: end - self.buffer_size] = states[self.buffer_size - start:]
            self.next_states[: end - self.buffer_size] = next_states[self.buffer_size - start:]
        else:
            self.states[start:end] = states
            self.next_states[start:end] = next_states
        self.num_samples = min(self.buffer_size, max(end, self.num_samples))
        self.step = end % self.buffer_size

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        for _ in range(num_mini_batch):
            idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            yield self.states[idxs], self.next_states[idxs]
