"""AMP 判别器 (LSGAN)。移植自 TienKung rsl_rl 的 Discriminator。

判别器吃一个转移 (s, s') 的拼接, 输出一个标量 logit:
  - 训练目标 (LSGAN): 专家转移 -> +1, 策略转移 -> -1;
  - style reward = amp_reward_coef * clamp(1 - 0.25*(d-1)^2, 0), d 越接近专家(+1)奖励越高;
  - compute_grad_pen: 对专家样本做梯度惩罚 (lambda=10) 稳定训练;
  - task_reward_lerp>0 时把 style reward 与任务 reward 线性插值混合。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import autograd


class Discriminator(nn.Module):
    def __init__(self, input_dim, amp_reward_coef, hidden_layer_sizes, device, task_reward_lerp=0.0):
        """input_dim: 单个状态的维度 (amp_obs 维数); 内部会按 2*input_dim 拼接 (s, s')。"""
        super().__init__()
        self.device = device
        self.input_dim = input_dim  # 单状态维度
        self.amp_reward_coef = amp_reward_coef
        self.task_reward_lerp = task_reward_lerp

        layers, curr = [], 2 * input_dim
        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(curr, hidden_dim))
            layers.append(nn.ReLU())
            curr = hidden_dim
        self.trunk = nn.Sequential(*layers).to(device)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)
        self.trunk.train()
        self.amp_linear.train()

    def forward(self, x):
        return self.amp_linear(self.trunk(x))

    def compute_grad_pen(self, expert_state, expert_next_state, lambda_=10):
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True
        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data, grad_outputs=ones,
            create_graph=True, retain_graph=True, only_inputs=True,
        )[0]
        # 让梯度范数趋于 0
        return lambda_ * grad.norm(2, dim=1).pow(2).mean()

    def predict_amp_reward(self, state, next_state, task_reward, normalizer=None):
        """返回 (reward[batch], d[batch,1])。无梯度, 供 rollout 阶段算 style reward。"""
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)
            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
            reward = self.amp_reward_coef * torch.clamp(1 - 0.25 * torch.square(d - 1), min=0)
            if self.task_reward_lerp > 0:
                reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))
            self.train()
        return reward.squeeze(), d

    def _lerp_reward(self, disc_r, task_r):
        return (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
