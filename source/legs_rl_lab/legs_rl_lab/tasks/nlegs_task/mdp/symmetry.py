"""Left-right symmetry augmentation for the legs_URDF policy.

Used by rsl-rl's PPO symmetry feature (data augmentation / mirror loss). For each
sample it appends the sagittal-plane mirror image, so the policy is trained to be
left-right symmetric.

Layout assumptions (validated against this task's config):
- Joints, in Isaac/policy order (A1_legs_V2 USD): [R1,L1, R2,L2, R3,L3, R4,L4, R5,L5, R6,L6].
  Mirroring swaps the same-level (R,L) pair and applies a per-joint sign derived from the
  MJCF joint axes (left and right share the same axis) under the sagittal reflection
  M = diag(1,-1,1):
      joint 1 (hip pitch) +1 | 2 (hip roll) -1 | 3 (hip yaw) -1
      joint 4 (knee)      +1 | 5 (ankle pitch) +1 | 6 (ankle roll) -1
  (s1/s4/s5 cross-checked against the mirror-symmetric default pose; the pair-swap perm is
  the same regardless of whether the Isaac order is R-first or L-first.)
- Observations are flat tensors; each group concatenates terms in declaration order,
  and within a term the history is laid out frame-major. The mirror is therefore applied
  per-frame within each term block. History length is per term: entries in ``_TERMS`` may
  carry an explicit 4th field, otherwise they default to ``_HISTORY`` (=10).
- gait_phase mirrors by shifting the gait clock half a cycle (legs swap roles),
  i.e. sin/cos both negate.
- A group may end with a height_scan term. The scan is an nx(x) x ny(y) grid in the
  base-yaw frame (GridPatternCfg with ordering="xy" -> flattened as idx = iy*nx + ix);
  its mirror flips the grid along y (iy -> ny-1-iy), heights keep their sign. Three
  flavours are in use:
    * nlegs_rough      critic only, 17x11=187, group-level history (10 frames) -> 2490
    * nlegs_rough_step critic only, 17x11=187, current frame only            ->  807
    * nlegs_rough_info policy AND critic, 16x11=176, current frame only      ->  646 / 796
  (the 16x11 grid is size=[1.5, 1.0] with offset x=+0.25, i.e. 1.0m ahead / 0.5m behind
  / 0.5m each side of the base.)
  ``_mirror_maps`` picks the layout among the candidates by the observation dimension
  (policy: 470 / 646; critic: 620 / 2490 / 807 / 796).
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]

_HISTORY = 10

# per-joint mirror sign for joint indices 1..6 (see module docstring).
# Derived from the A1_legs_V2 (MJCF) joint axes, where left and right joints share the
# SAME axis convention (unlike the original URDF). Cross-checked against the mirror-symmetric
# default pose (.*4=0.2, .*5=-0.1 identical on both sides => s4=s5=+1).
#   joint 1 hip-pitch +1 | 2 hip-roll -1 | 3 hip-yaw -1 | 4 knee +1 | 5 ankle-pitch +1 | 6 ankle-roll -1
_JOINT_SIGN_1_6 = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0]
# Isaac DOF order alternates sides per level (R1,L1,R2,L2,...); swap each same-level (R,L) pair.
# The pair-swap perm is identical whether the order is R-first or L-first.
_J_PERM = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
_J_SIGN = [s for s in _JOINT_SIGN_1_6 for _ in range(2)]  # [s1,s1,s2,s2,...]

# vector mirror signs under sagittal reflection (y flips):
_ANG = ([0, 1, 2], [-1.0, 1.0, -1.0])   # angular velocity (axial): roll/yaw flip
_GRAV = ([0, 1, 2], [1.0, -1.0, 1.0])   # projected gravity (polar): y flips
_LIN = ([0, 1, 2], [1.0, -1.0, 1.0])    # linear velocity (polar): y flips
_CMD = ([0, 1, 2], [1.0, -1.0, -1.0])   # [vx, vy, wz]: vy and yaw-rate flip
_JNT = (_J_PERM, _J_SIGN)               # any per-joint quantity
_GAIT = ([0, 1], [-1.0, -1.0])          # [sin, cos] of phase -> phase + 0.5

# height_scan 尾项: nx(x) x ny(y) 网格, 展平序 idx = iy*nx + ix (GridPatternCfg ordering="xy")。
# 镜像 = 网格沿 y 翻转(iy -> ny-1-iy), 高度标量不变号。


def _hscan_term(nx: int, ny: int, history: int | None = None):
    """生成 height_scan 的布局元组 (dim, perm, sign[, history])。"""
    perm = [(ny - 1 - iy) * nx + ix for iy in range(ny) for ix in range(nx)]
    sign = [1.0] * (nx * ny)
    term = (nx * ny, perm, sign)
    return term if history is None else (*term, history)


# nlegs_rough: 187 点, 跟着 group 级 history 叠 10 帧
_HSCAN_TERM = _hscan_term(17, 11)
# nlegs_rough_step: 187 点, 仅当前帧
_HSCAN_TERM_CURRENT = _hscan_term(17, 11, 1)
# nlegs_rough_info: 176 点(size=[1.5,1.0] + offset x=+0.25 -> 前1.0m/后0.5m/左右0.5m), 仅当前帧
# 该布局 policy 与 critic 共用(actor 也吃地形)
_HSCAN_TERM_INFO = _hscan_term(16, 11, 1)

# term layout per observation group: list of (dim, perm, sign[, history]), in declaration
# order; history defaults to _HISTORY when omitted
_TERMS = {
    "policy": [
        (3, *_ANG),    # base_ang_vel
        (3, *_GRAV),   # projected_gravity
        (3, *_CMD),    # velocity_commands
        (12, *_JNT),   # joint_pos_rel
        (12, *_JNT),   # joint_vel_rel
        (12, *_JNT),   # last_action
        (2, *_GAIT),   # gait_phase
    ],
    "critic": [
        (3, *_LIN),    # base_lin_vel
        (3, *_ANG),    # base_ang_vel
        (3, *_GRAV),   # projected_gravity
        (3, *_CMD),    # velocity_commands
        (12, *_JNT),   # joint_pos_rel
        (12, *_JNT),   # joint_vel_rel
        (12, *_JNT),   # joint_effort
        (12, *_JNT),   # last_action
        (2, *_GAIT),   # gait_phase
    ],
}

# cache of (perm_idx, sign) tensors per (obs_type, device, obs_dim)
_CACHE: dict = {}


def _term_history(term) -> int:
    """该项的历史帧数: 布局元组给了第 4 个字段就用它, 否则用组级默认 _HISTORY。"""
    return term[3] if len(term) > 3 else _HISTORY


def _mirror_maps(obs_type: str, obs_dim: int, device: torch.device):
    key = (obs_type, obs_dim, device)
    if key in _CACHE:
        return _CACHE[key]
    # 组末尾可能带 height_scan 尾项, 按观测维度自动匹配布局:
    #   policy: 盲走(flat/rough/rough_step) / 带 176 点当前帧(rough_info)
    #   critic: flat 无 / rough 带 187x10 / rough_step 带 187x1 / rough_info 带 176x1
    candidates = [_TERMS[obs_type]]
    if obs_type == "critic":
        candidates.append(_TERMS[obs_type] + [_HSCAN_TERM])
        candidates.append(_TERMS[obs_type] + [_HSCAN_TERM_CURRENT])
    candidates.append(_TERMS[obs_type] + [_HSCAN_TERM_INFO])
    expected = [sum(t[0] * _term_history(t) for t in terms) for terms in candidates]
    if obs_dim not in expected:
        raise ValueError(
            f"symmetry: {obs_type} obs dim {obs_dim} not in expected {expected} "
            f"(sum of per-term dim x history, default history {_HISTORY}). "
            f"Observation layout changed; update tasks/nlegs_task/mdp/symmetry.py."
        )
    terms = candidates[expected.index(obs_dim)]
    perm = []
    sign = []
    off = 0
    for term in terms:
        d, p, s = term[0], term[1], term[2]
        history = _term_history(term)
        for f in range(history):  # term block is frame-major: [frame0 dims, frame1 dims, ...]
            base = off + f * d
            perm.extend(base + p[j] for j in range(d))
            sign.extend(s)
        off += d * history
    perm_t = torch.tensor(perm, dtype=torch.long, device=device)
    sign_t = torch.tensor(sign, dtype=torch.float, device=device)
    _CACHE[key] = (perm_t, sign_t)
    return perm_t, sign_t


@torch.no_grad()
def compute_symmetric_states(
    env: "ManagerBasedRLEnv" = None,
    obs=None,
    actions: torch.Tensor | None = None,
):
    """Append the left-right mirror of each sample (batch -> 2x batch).

    Matches rsl-rl-lib 5.x: ``obs`` is a TensorDict whose keys are the observation
    group names ("policy", "critic"), each a flat ``[N, dim]`` tensor; ``actions`` is
    a ``[N, 12]`` tensor. Returns the augmented TensorDict and action tensor.
    """
    obs_aug = None
    if obs is not None:
        batch = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        for group in obs.keys():
            if group not in _TERMS:
                continue
            g = obs[group]
            perm, sign = _mirror_maps(group, g.shape[1], g.device)
            obs_aug[group][:batch] = g
            obs_aug[group][batch : 2 * batch] = g[:, perm] * sign

    act_aug = None
    if actions is not None:
        j_perm = torch.tensor(_J_PERM, dtype=torch.long, device=actions.device)
        j_sign = torch.tensor(_J_SIGN, dtype=torch.float, device=actions.device)
        act_mirror = actions[:, j_perm] * j_sign
        act_aug = torch.cat([actions, act_mirror], dim=0)

    return obs_aug, act_aug
