"""nlegs_flat_step_to_crouch：从站立踏步进入限位下蹲，最后落脚停稳。"""

from pathlib import Path
import math

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils import configclass

from legs_rl_lab.assets.nlegs import nlegs
from legs_rl_lab.tasks.nlegs_task import mdp
from legs_rl_lab.tasks.nlegs_task.agents.rsl_rl_ppo_cfg import NlegsFlatPPORunnerCfg
from .flat_env_cfg import FlatEnvCfg
from . import step_to_crouch_mdp as crouch


class LocalFlatTerrain(TerrainImporter):
    """固定平板顶面 z=0；GPU 接触过滤不支持默认无限 Plane 碰撞体。"""

    def import_ground_plane(self, name, size=(2.0e6, 2.0e6)):
        path = f"{self.cfg.prim_path}/{name}"
        span = max(20.0, 2 * math.ceil(math.sqrt(self.cfg.num_envs)) * self.cfg.env_spacing)
        cfg = sim_utils.CuboidCfg(
            size=(span, span, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=self.cfg.physics_material,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
        )
        cfg.func(path, cfg, translation=(0, 0, -0.05))
        self.terrain_prim_paths.append(path)


@configclass
class CommandsCfg:
    crouch_progress = crouch.CrouchProgressCfg()


@configclass
class RewardsCfg:
    posture = RewTerm(func=crouch.posture, weight=3.0)
    body_pose = RewTerm(func=crouch.body_pose, weight=2.0)
    stepping = RewTerm(func=crouch.stepping, weight=2.0)
    settling = RewTerm(func=crouch.settling, weight=3.0)
    loaded_slip = RewTerm(func=crouch.loaded_slip, weight=-5.0)
    flight = RewTerm(func=crouch.flight, weight=-2.0)
    drift = RewTerm(func=crouch.drift, weight=-10.0)
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.15)
    action_acc = RewTerm(func=mdp.action_acc_l2, weight=-0.05)
    joint_limits = RewTerm(func=crouch.joint_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    feet_contact_forces = RewTerm(
        func=mdp.contact_forces, weight=-0.0002,
        params={"threshold": 200, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*6")},
    )


@configclass
class FlatStepToCrouchEnvCfg(FlatEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.usd_path = str(Path(nlegs.__file__).parent / "mjcf/nlegs_limit/nlegs_limit.usd")
        self.scene.height_scanner = None
        self.scene.terrain.class_type = LocalFlatTerrain
        self.scene.sky_light.spawn.texture_file = None
        self.scene.terrain.visual_material = None
        for side, name in (("L", "left_foot_contact"), ("R", "right_foot_contact")):
            setattr(self.scene, name, ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/base/Link_{side}6",
                filter_prim_paths_expr=["/World/ground/terrain"],
                track_contact_points=True, max_contact_data_count_per_prim=24,
                update_period=self.sim.dt,
            ))
        self.events.push_robot = None
        # 保留摩擦/执行器/零偏随机化，但不把近乎冰面的工况作为下蹲的常态。
        self.events.physics_material.params.update(
            static_friction_range=(0.6, 1.3), dynamic_friction_range=(0.6, 1.3), make_consistent=True,
        )
        self.curriculum = None
        for group in (self.observations.policy, self.observations.critic):
            group.velocity_commands = None
            group.crouch_progress = ObsTerm(func=mdp.generated_commands, params={"command_name": "crouch_progress"})
            group.gait_phase = ObsTerm(func=crouch.gait_obs)
        command = self.commands.crouch_progress
        self.episode_length_s = command.prepare_s + command.lower_s + command.hold_s
        self.terminations.time_out = DoneTerm(func=crouch.time_out, time_out=True)
        self.terminations.excessive_slip = DoneTerm(func=crouch.excessive_slip)
        self.terminations.bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.05})


@configclass
class FlatStepToCrouchPlayEnvCfg(FlatStepToCrouchEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False


@configclass
class NlegsFlatStepToCrouchPPORunnerCfg(NlegsFlatPPORunnerCfg):
    experiment_name = "nlegs_flat_step_to_crouch"

    def __post_init__(self):
        self.algorithm.symmetry_cfg = None
