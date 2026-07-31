"""AMP 观测归一化器 (running mean/std)。移植自 TienKung rsl_rl 的 Normalizer,
判别器输入前对 amp_obs 做标准化, 使 style reward 对各通道量纲不敏感。"""

from __future__ import annotations

import numpy as np
import torch


class RunningMeanStd:
    """在线并行算法维护数据流的均值/方差 (Welford / parallel variance)。"""

    def __init__(self, epsilon: float = 1e-4, shape=()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = m_2 / tot_count
        self.count = tot_count


class Normalizer(RunningMeanStd):
    def __init__(self, input_dim, epsilon: float = 1e-4, clip_obs: float = 10.0):
        super().__init__(shape=input_dim)
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + self.epsilon), -self.clip_obs, self.clip_obs)

    def normalize_torch(self, x: torch.Tensor, device) -> torch.Tensor:
        mean = torch.tensor(self.mean, device=device, dtype=torch.float32)
        std = torch.sqrt(torch.tensor(self.var + self.epsilon, device=device, dtype=torch.float32))
        return torch.clamp((x - mean) / std, -self.clip_obs, self.clip_obs)

    # -- checkpoint 存取(np 数组转 list, 方便 torch.save 里塞 dict) --
    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, sd: dict) -> None:
        self.mean = np.asarray(sd["mean"], dtype=np.float64)
        self.var = np.asarray(sd["var"], dtype=np.float64)
        self.count = float(sd["count"])
