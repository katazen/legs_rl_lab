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
  and within a term the history (length 10) is laid out frame-major. The mirror is
  therefore applied per-frame within each term block.
- gait_phase mirrors by shifting the gait clock half a cycle (legs swap roles),
  i.e. sin/cos both negate.
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

# term layout per observation group: list of (dim, perm, sign), in declaration order
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


def _mirror_maps(obs_type: str, obs_dim: int, device: torch.device):
    key = (obs_type, obs_dim, device)
    if key in _CACHE:
        return _CACHE[key]
    terms = _TERMS[obs_type]
    frame_dim = sum(d for d, _, _ in terms)
    expected = frame_dim * _HISTORY
    if obs_dim != expected:
        raise ValueError(
            f"symmetry: {obs_type} obs dim {obs_dim} != expected {expected} "
            f"(frame_dim {frame_dim} x history {_HISTORY}). Observation layout changed; "
            f"update tasks/legs_task/mdp/symmetry.py."
        )
    perm = []
    sign = []
    off = 0
    for d, p, s in terms:
        for f in range(_HISTORY):  # term block is frame-major: [frame0 dims, frame1 dims, ...]
            base = off + f * d
            perm.extend(base + p[j] for j in range(d))
            sign.extend(s)
        off += d * _HISTORY
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
