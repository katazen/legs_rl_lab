"""Rebuild an Isaac articulation's actuator groups from a run deploy.yaml."""
from __future__ import annotations

import copy
import yaml


def load_deploy(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _one(values, name):
    values = list(values)
    if not values or any(v != values[0] for v in values[1:]):
        raise ValueError(f"{name} must be uniform inside one actuator group: {values}")
    return values[0]


def configure_from_deploy(robot_cfg, deploy, fixed_delays=None):
    """Return a copy using the exact actuator groups and nominal joint values in deploy."""
    cfg = copy.deepcopy(robot_cfg)
    names = deploy["joint_names"]
    templates = list(cfg.actuators.values())
    pd_template = next(a for a in templates if not hasattr(a, "saturation_effort"))
    dc_template = next(a for a in templates if hasattr(a, "saturation_effort"))
    fixed_delays = fixed_delays or {}
    groups, covered = {}, []

    for group_name, spec in deploy["actuators"].items():
        ids = list(spec["joint_ids"]); covered += ids
        joint_names = [names[i] for i in ids]
        per_joint = lambda field: {names[i]: deploy[field][i] for i in ids}
        delay = spec["delay"]
        min_delay = max_delay = int(fixed_delays[group_name]) if group_name in fixed_delays else None
        common = dict(
            joint_names_expr=joint_names,
            stiffness=per_joint("stiffness"), damping=per_joint("damping"),
            armature=per_joint("armature"), friction=per_joint("friction"),
            dynamic_friction=per_joint("dynamic_friction"),
            viscous_friction=per_joint("viscous_friction"),
            effort_limit=_one(spec["effort_limit"], f"{group_name}.effort_limit"),
            effort_limit_sim=_one(spec["effort_limit_sim"], f"{group_name}.effort_limit_sim"),
            velocity_limit=_one(spec["velocity_limit"], f"{group_name}.velocity_limit"),
            velocity_limit_sim=_one(spec["velocity_limit_sim"], f"{group_name}.velocity_limit_sim"),
            min_delay=delay[0] if min_delay is None else min_delay,
            max_delay=delay[1] if max_delay is None else max_delay,
        )
        if spec["torque_model"] == "dc_motor":
            common["saturation_effort"] = float(spec["saturation_effort"])
            groups[group_name] = type(dc_template)(**common)
        elif spec["torque_model"] == "ideal_pd":
            groups[group_name] = type(pd_template)(**common)
        else:
            raise ValueError(f"unsupported torque model: {spec['torque_model']}")

    if sorted(covered) != list(range(len(names))):
        raise ValueError(f"actuator groups do not cover every joint exactly once: {covered}")
    cfg.actuators = groups
    cfg.init_state.pos = tuple(deploy["base_init_state"]["pos"])
    cfg.init_state.rot = tuple(deploy["base_init_state"]["rot"])
    # default_joint_pos is in policy/action order; joint_ids_map[action_idx] gives SDK/USD index.
    default_sdk = [0.0] * len(names)
    for action_idx, sdk_idx in enumerate(deploy["joint_ids_map"]):
        default_sdk[sdk_idx] = deploy["default_joint_pos"][action_idx]
    cfg.init_state.joint_pos = {name: default_sdk[i] for i, name in enumerate(names)}
    return cfg
