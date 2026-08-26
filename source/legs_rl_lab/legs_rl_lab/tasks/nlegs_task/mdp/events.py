from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def randomize_joint_zero_bias(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    bias_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """给每个环境抽一组"标0偏置":模拟实机编码器零位标定误差。

    每关节一个恒定偏置, 在 reset 时采样、整幕不变(区别于每步传感器噪声)。
    仅写入 env._joint_zero_bias 缓冲, 由 observations.joint_pos_rel_biased 加到
    策略观测的 joint_pos_rel 上(critic 不加, 保持真值)。

    对称范围(mean=0)是有意为之: 目的是让策略对任意方向的恒定零偏都鲁棒,
    而不是去拟合当前实机那个特定偏置(重标0后偏置就变了)。
    """
    asset = env.scene[asset_cfg.name]
    n = asset.data.joint_pos.shape[1]
    if (not hasattr(env, "_joint_zero_bias")) or env._joint_zero_bias.shape != (env.num_envs, n):
        env._joint_zero_bias = torch.zeros(env.num_envs, n, device=env.device)
    lo, hi = float(bias_range[0]), float(bias_range[1])
    env._joint_zero_bias[env_ids] = torch.empty(
        len(env_ids), n, device=env.device
    ).uniform_(lo, hi)
