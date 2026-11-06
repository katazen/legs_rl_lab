"""W1196 的软站立目标，与相位观测共用运动到站立的渐变权重。"""

from isaaclab.managers import SceneEntityCfg

from legs_rl_lab.tasks.nlegs_task import mdp


def moving_base_height_l2(
    env, target_height: float, command_threshold: float, stop_transition_time: float,
    sensor_cfg: SceneEntityCfg | None = None, yaw_command_threshold: float | None = None,
):
    weight = mdp.gait_motion_weight(env, command_threshold, stop_transition_time, yaw_command_threshold)
    return mdp.base_height_l2(env, target_height, sensor_cfg=sensor_cfg) * weight


def standing_joint_pos_l1(
    env, command_threshold: float, stop_transition_time: float, yaw_command_threshold: float | None = None,
):
    weight = 1.0 - mdp.gait_motion_weight(env, command_threshold, stop_transition_time, yaw_command_threshold)
    return mdp.joint_deviation_l1(env) * weight


def standing_joint_vel_l2(
    env, command_threshold: float, stop_transition_time: float, yaw_command_threshold: float | None = None,
):
    weight = 1.0 - mdp.gait_motion_weight(env, command_threshold, stop_transition_time, yaw_command_threshold)
    return mdp.joint_vel_l2(env) * weight


def standing_feet_contact(
    env, sensor_cfg: SceneEntityCfg, command_threshold: float, stop_transition_time: float,
    yaw_command_threshold: float | None = None,
):
    weight = 1.0 - mdp.gait_motion_weight(env, command_threshold, stop_transition_time, yaw_command_threshold)
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return (forces > 1.0).all(dim=1).float() * weight
