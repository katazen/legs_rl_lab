from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
import torch


def root_height_error(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    max_height = env.scene["robot"].data.default_root_state[:, 2]
    return asset.data.root_pos_w[:, 2] < minimum_height or asset.data.root_pos_w[:, 2] > max_height