"""flat_static 的命令门控奖励；静止目标是资产默认站姿，不是 reset 后随机姿态。"""

from isaaclab.managers import SceneEntityCfg

from legs_rl_lab.tasks.nlegs_task import mdp


def moving_base_height_l2(env, target_height: float, command_threshold: float):
    return mdp.base_height_l2(env, target_height) * mdp.command_is_moving(env, command_threshold)


def standing_joint_pos_l1(env, command_threshold: float):
    return mdp.joint_deviation_l1(env) * ~mdp.command_is_moving(env, command_threshold)


def standing_joint_vel_l2(env, command_threshold: float):
    return mdp.joint_vel_l2(env) * ~mdp.command_is_moving(env, command_threshold)


def standing_feet_contact(env, sensor_cfg: SceneEntityCfg, command_threshold: float):
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return (forces > 1.0).all(dim=1).float() * ~mdp.command_is_moving(env, command_threshold)
