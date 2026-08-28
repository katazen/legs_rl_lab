"""nlegs 任务自定义终止函数（isaaclab 内置终止项仍从 isaaclab.envs.mdp 星导入）。"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_height_below_feet(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """base 高出最低那只脚不足 minimum_height 即终止(身体塌到脚边)。

    纯相对量 base_z - min(feet_z)，与地形绝对高度无关，flat/rough 通用；
    替代 isaaclab 的 root_height_below_minimum(世界系绝对 z, 仅适用于平地)。
    asset_cfg.body_ids 应选脚部 body(如 body_names=".*6")。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    return asset.data.root_pos_w[:, 2] - feet_z.min(dim=1).values < minimum_height
