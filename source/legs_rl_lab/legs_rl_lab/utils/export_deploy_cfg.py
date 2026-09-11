import numpy as np
import os
import yaml

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import class_to_dict
from isaaclab.utils.string import resolve_matching_names


def format_value(x):
    if isinstance(x, float):
        return float(f"{x:.3g}")
    elif isinstance(x, list):
        return [format_value(i) for i in x]
    elif isinstance(x, dict):
        return {k: format_value(v) for k, v in x.items()}
    else:
        return x


def yaml_safe(x):
    """把导出内容变成 yaml.safe_load 能读回来的形式。

    yaml.dump 遇到 tuple / slice / configclass 实例会写成 !!python/... 标签, 而 sim2sim 与
    实机部署都用 safe_load 读, 一碰到就直接报 ConstructorError。典型触发点: commands 的
    ranges.heading(tuple), 观测项 params 里的 SceneEntityCfg(含 joint_ids=slice(None))。
    """
    if isinstance(x, np.generic):  # np.int64 不是 int 子类, 必须先拆
        return x.item()
    if isinstance(x, dict):
        return {k: yaml_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [yaml_safe(v) for v in x]
    if x is None or isinstance(x, (str, bool, int, float)):
        return x
    return str(x)


def export_deploy_cfg(env: ManagerBasedRLEnv, log_dir, policy_action_clip=None):
    asset: Articulation = env.scene["robot"]
    joint_sdk_names = env.cfg.scene.robot.joint_sdk_names
    joint_ids_map, _ = resolve_matching_names(asset.data.joint_names, joint_sdk_names, preserve_order=True)

    cfg = {}  # noqa: SIM904
    cfg["format_version"] = 2
    cfg["joint_names"] = list(joint_sdk_names)
    cfg["joint_ids_map"] = joint_ids_map
    cfg["physics_dt"] = env.cfg.sim.dt
    cfg["step_dt"] = env.cfg.sim.dt * env.cfg.decimation
    cfg["policy_action_clip"] = policy_action_clip
    cfg["base_init_state"] = {
        "pos": list(env.cfg.scene.robot.init_state.pos),
        "rot": list(env.cfg.scene.robot.init_state.rot),
    }
    stiffness = np.zeros(len(joint_sdk_names))
    stiffness[joint_ids_map] = asset.data.default_joint_stiffness[0].detach().cpu().numpy().tolist()
    cfg["stiffness"] = stiffness.tolist()
    damping = np.zeros(len(joint_sdk_names))
    damping[joint_ids_map] = asset.data.default_joint_damping[0].detach().cpu().numpy().tolist()
    cfg["damping"] = damping.tolist()
    cfg["default_joint_pos"] = asset.data.default_joint_pos[0].detach().cpu().numpy().tolist()

    # --- extra robot params for sim2sim (armature / effort limits), in sdk order ---
    # 直接读 asset.data 里已解析的每关节值（robust: cfg 里 armature/effort 可能是 scalar/dict/None），
    # 再按 joint_ids_map 重排到 sdk 顺序（与 stiffness/damping 同一套映射）
    armature = np.zeros(len(joint_sdk_names))
    armature[joint_ids_map] = asset.data.default_joint_armature[0].detach().cpu().numpy().tolist()
    cfg["armature"] = armature.tolist()
    effort = np.zeros(len(joint_sdk_names))
    effort[joint_ids_map] = asset.data.joint_effort_limits[0].detach().cpu().numpy().tolist()
    cfg["effort"] = effort.tolist()

    for key in ["friction", "dynamic_friction", "viscous_friction"]:
        values = np.zeros(len(joint_sdk_names))
        tensor = getattr(asset.data, f"default_joint_{key}_coeff")
        values[joint_ids_map] = tensor[0].detach().cpu().numpy().tolist()
        cfg[key] = values.tolist()

    cfg["actuators"] = {}
    for name, actuator in asset.actuators.items():
        policy_ids = actuator.joint_indices
        if policy_ids == slice(None):
            policy_ids = list(range(len(asset.data.joint_names)))
        elif hasattr(policy_ids, "detach"):
            policy_ids = policy_ids.detach().cpu().tolist()
        sdk_ids = [joint_ids_map[i] for i in policy_ids]
        group = {
            "joint_ids": sdk_ids,
            "torque_model": "dc_motor" if hasattr(actuator, "_saturation_effort") else "ideal_pd",
            "effort_limit": actuator.effort_limit[0].detach().cpu().tolist(),
            "effort_limit_sim": actuator.effort_limit_sim[0].detach().cpu().tolist(),
            "velocity_limit": actuator.velocity_limit[0].detach().cpu().tolist(),
            "velocity_limit_sim": actuator.velocity_limit_sim[0].detach().cpu().tolist(),
            "saturation_effort": getattr(actuator, "_saturation_effort", None),
            "delay": [int(getattr(actuator.cfg, "min_delay", 0)), int(getattr(actuator.cfg, "max_delay", 0))],
        }
        cfg["actuators"][name] = group

    # --- gait clock period (nlegs GaitCfg, or a custom env with .period) ---
    gait_cfg = getattr(env.cfg, "gait", None)
    if gait_cfg is not None and getattr(gait_cfg, "period", None) is not None:
        cfg["gait_period"] = gait_cfg.period
    elif getattr(env, "period", None) is not None:
        cfg["gait_period"] = env.period

    # --- action delay (DelayedPDActuatorCfg min/max_delay), in physics steps ---
    min_delays = [a.min_delay for a in env.cfg.scene.robot.actuators.values() if getattr(a, "min_delay", None) is not None]
    max_delays = [a.max_delay for a in env.cfg.scene.robot.actuators.values() if getattr(a, "max_delay", None) is not None]
    if min_delays and max_delays:
        cfg["action_delay"] = [int(min(min_delays)), int(max(max_delays))]

    # --- commands ---
    cfg["commands"] = {}
    if hasattr(env.cfg.commands, "base_velocity"):  # some environments do not have base_velocity command
        cfg["commands"]["base_velocity"] = {}
        if hasattr(env.cfg.commands.base_velocity, "limit_ranges"):
            ranges = env.cfg.commands.base_velocity.limit_ranges.to_dict()
        else:
            ranges = env.cfg.commands.base_velocity.ranges.to_dict()
        # 所有 tuple 都要转 list: yaml.safe_dump 会把 tuple 写成 !!python/tuple, safe_load 读不回来
        # (heading 只在 heading_command=True 时非 None, 漏转会让 deploy.yaml 直接 load 失败)
        ranges = {
            key: list(value) if isinstance(value, tuple) else value for key, value in ranges.items()
        }
        cfg["commands"]["base_velocity"]["ranges"] = ranges

    if hasattr(env.cfg.commands, "crouch_progress"):
        command_cfg = env.cfg.commands.crouch_progress.to_dict()
        command_cfg.pop("class_type")
        cfg["commands"]["crouch_progress"] = command_cfg

    # --- actions ---
    action_names = env.action_manager.active_terms
    action_terms = zip(action_names, env.action_manager._terms.values())
    cfg["actions"] = {}
    for action_name, action_term in action_terms:
        term_cfg = action_term.cfg.copy()
        if isinstance(term_cfg.scale, float):
            term_cfg.scale = [term_cfg.scale for _ in range(action_term.action_dim)]
        else:  # dict
            term_cfg.scale = action_term._scale[0].detach().cpu().numpy().tolist()

        if term_cfg.clip is not None:
            term_cfg.clip = action_term._clip[0].detach().cpu().numpy().tolist()

        if action_name in ["JointPositionAction", "JointVelocityAction"]:
            if term_cfg.use_default_offset:
                term_cfg.offset = action_term._offset[0].detach().cpu().numpy().tolist()
            else:
                term_cfg.offset = [0.0 for _ in range(action_term.action_dim)]

        # clean cfg
        term_cfg = term_cfg.to_dict()

        for _ in ["class_type", "asset_name", "debug_vis", "preserve_order", "use_default_offset"]:
            del term_cfg[_]
        cfg["actions"][action_name] = term_cfg

        if action_term._joint_ids == slice(None):
            cfg["actions"][action_name]["joint_ids"] = None
        else:
            cfg["actions"][action_name]["joint_ids"] = action_term._joint_ids

    # --- height scanner geometry (sim2sim / 实机建图需要知道采样网格长什么样) ---
    # 只有 actor 吃高度图时这些参数才是部署必需的; critic 用不影响部署, 导了也无害。
    scanner_cfg = getattr(env.cfg.scene, "height_scanner", None)
    if scanner_cfg is not None:
        cfg["height_scanner"] = {
            "offset": list(scanner_cfg.offset.pos),
            "ray_alignment": scanner_cfg.ray_alignment,
            "resolution": scanner_cfg.pattern_cfg.resolution,
            "size": list(scanner_cfg.pattern_cfg.size),
            "ordering": scanner_cfg.pattern_cfg.ordering,
        }

    # --- observations ---
    obs_names = env.observation_manager.active_terms["policy"]
    obs_cfgs = env.observation_manager._group_obs_term_cfgs["policy"]
    obs_terms = zip(obs_names, obs_cfgs)
    cfg["observations"] = {}
    for obs_name, obs_cfg in obs_terms:
        obs_dims = tuple(obs_cfg.func(env, **obs_cfg.params).shape)
        term_cfg = obs_cfg.copy()
        if term_cfg.scale is not None:
            scale = term_cfg.scale.detach().cpu().numpy().tolist()
            if isinstance(scale, float):
                term_cfg.scale = [scale for _ in range(obs_dims[1])]
            else:
                term_cfg.scale = scale
        else:
            term_cfg.scale = [1.0 for _ in range(obs_dims[1])]
        if term_cfg.clip is not None:
            term_cfg.clip = list(term_cfg.clip)
        if term_cfg.history_length == 0:
            term_cfg.history_length = 1

        # clean cfg
        term_cfg = term_cfg.to_dict()
        for _ in ["func", "modifiers", "noise", "flatten_history_dim"]:
            del term_cfg[_]
        cfg["observations"][obs_name] = term_cfg

    # --- save config file ---
    filename = os.path.join(log_dir, "params", "deploy.yaml")
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not isinstance(cfg, dict):
        cfg = class_to_dict(cfg)
    cfg = format_value(yaml_safe(cfg))
    with open(filename, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)
