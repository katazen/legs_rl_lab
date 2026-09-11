from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils

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


class reset_from_crouch_pose_bank(ManagerTermBase):
    """固定比例抽取直立、中间蹲姿、限位附近的整套状态；不改动作基准。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if env.cfg.scene.terrain.terrain_type != "plane":
            raise ValueError("crouch pose bank 仅支持水平平地")
        path = Path(cfg.params["bank_path"])
        bank = json.loads(path.read_text())
        if bank.get("version") != 3:
            raise ValueError("不支持的 crouch pose bank 版本")
        for relative, expected in bank["model_sha256"].items():
            model_file = (path.parent / relative).resolve()
            if not model_file.is_relative_to(path.parent.resolve()):
                raise ValueError("姿态表的模型路径越出资产目录")
            if hashlib.sha256(model_file.read_bytes()).hexdigest() != expected:
                raise ValueError(f"{relative} 已改变，请重新运行 scripts/generate_crouch_pose_bank.py")

        self.asset = env.scene["robot"]
        usd_path = Path(self.asset.cfg.spawn.usd_path).resolve()
        if (not usd_path.is_relative_to(path.parent.resolve())
                or str(usd_path.relative_to(path.parent.resolve())) not in bank["model_sha256"]):
            raise ValueError("实际加载的 USD 不在姿态表的模型校验清单内")
        names = bank["joint_names"]
        if len(names) != len(set(names)) or set(names) != set(self.asset.joint_names):
            raise ValueError("姿态表必须恰好覆盖机器人全部关节，且名字不能重复")
        order = [names.index(name) for name in self.asset.joint_names]
        rows = bank["poses"]
        if not rows:
            raise ValueError("姿态表为空")
        labels = ("stand", "intermediate", "crouch")
        if {r["posture"] for r in rows} != set(labels):
            raise ValueError("姿态表必须包含 stand、intermediate、crouch 三组")
        self.groups = torch.tensor([labels.index(r["posture"]) for r in rows], device=env.device)
        self.group_sizes = torch.bincount(self.groups, minlength=3)
        self.height = torch.tensor([r["height"] for r in rows], device=env.device)
        self.quat = torch.tensor([r["quat_wxyz"] for r in rows], device=env.device)
        self.q = torch.tensor([r["joint_pos"] for r in rows], device=env.device)
        limits = torch.tensor(bank["joint_limits"], device=env.device)
        if self.q.shape != (len(rows), len(names)) or self.quat.shape != (len(rows), 4):
            raise ValueError("姿态表维度错误")
        if limits.shape != (len(names), 2) or not torch.isfinite(limits).all() or (limits[:, 0] >= limits[:, 1]).any():
            raise ValueError("姿态表关节限位错误")
        if not all(torch.isfinite(t).all() for t in (self.height, self.quat, self.q)):
            raise ValueError("姿态表包含非有限值")
        if (self.height <= 0.2).any() or not torch.allclose(self.quat.norm(dim=1), torch.ones_like(self.height), atol=1e-5):
            raise ValueError("姿态表高度/四元数错误")
        if ((self.q < limits[:, 0]) | (self.q > limits[:, 1])).any():
            raise ValueError("姿态表有关节越限，不能用 clamp 破坏接触约束")
        self.q = self.q[:, order]
        limits = limits[order]
        default = self.asset.data.default_joint_pos
        if ((default < limits[:, 0]) | (default > limits[:, 1])).any():
            raise ValueError("站立动作基准不在新限位内；禁止自动修改动作基准")
        if not torch.allclose(self.asset.data.joint_pos_limits, limits.expand(env.num_envs, -1, -1),
                              rtol=0, atol=1e-5):
            raise ValueError("训练 USD 的关节限位与姿态表不一致，请从 nlegs_limit.xml 重新转换 USD 并生成姿态表")

    def __call__(self, env, env_ids, bank_path: str,
                 stand_probability: float = 0.1, crouch_probability: float = 0.6):
        if not (0 <= stand_probability <= 1 and 0 <= crouch_probability <= 1
                and stand_probability + crouch_probability <= 1):
            raise ValueError("站立/限位附近采样概率必须非负且总和不超过 1")
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)[env_ids if isinstance(env_ids, slice) else slice(None)]
        else:
            env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        if len(env_ids) == 0:
            return
        probabilities = torch.tensor([stand_probability, 1 - (stand_probability + crouch_probability),
                                      crouch_probability], device=env.device)
        weights = (probabilities / self.group_sizes)[self.groups]
        indices = torch.multinomial(weights, len(env_ids), replacement=True)
        q = self.q[indices]
        yaw = torch.empty(len(env_ids), device=env.device).uniform_(-torch.pi, torch.pi)
        zero = torch.zeros_like(yaw)
        root = self.asset.data.default_root_state[env_ids].clone()
        root[:, :3] = env.scene.env_origins[env_ids]
        root[:, 2] += self.height[indices]
        root[:, 3:7] = math_utils.quat_mul(math_utils.quat_from_euler_xyz(zero, zero, yaw), self.quat[indices])
        root[:, 7:13] = 0
        self.asset.write_root_state_to_sim(root, env_ids=env_ids)
        self.asset.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        # 首个策略动作到来前，驱动目标与出生姿态一致。
        self.asset.set_joint_position_target(q, env_ids=env_ids)
        self.asset.set_joint_velocity_target(torch.zeros_like(q), env_ids=env_ids)
        if hasattr(env, "_prev_prev_action"):
            env._prev_prev_action[env_ids] = 0  # 自定义 action_acc 奖励的跨步缓存
