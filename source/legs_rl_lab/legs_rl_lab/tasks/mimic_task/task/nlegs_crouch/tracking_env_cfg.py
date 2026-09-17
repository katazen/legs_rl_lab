"""nlegs v2 crouch tracking, adapted from unitree_rl_lab/tasks/mimic (Apache-2.0)."""

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from legs_rl_lab.assets.nlegs import nlegs
from legs_rl_lab.tasks.mimic_task import mdp

ASSET = Path(nlegs.__file__).parent
ROBOT = nlegs.NLEGS_CFG.copy()
ROBOT.spawn.usd_path = str(ASSET / "mjcf/nlegs_limit/nlegs_limit.usd")
ROBOT.soft_joint_pos_limit_factor = 1.0
ROBOT.init_state.pos = (0, 0, .581834209777198)
BODIES = ["base", "Link_L2", "Link_L4", "Link_L6", "Link_R2", "Link_R4", "Link_R6"]


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


def joint_ranges(xml_path):
    joints = ET.parse(xml_path).getroot().findall(".//worldbody//joint")
    ranges = {j.attrib["name"]: tuple(map(float, j.attrib["range"].split())) for j in joints}
    for name, bounds in ranges.items():
        if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] >= bounds[1]:
            raise ValueError(f"{name} 的 XML 限位无效: {bounds}")
    return ranges


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", class_type=LocalFlatTerrain,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1., dynamic_friction=1., restitution=0.,
        ),
    )
    robot: ArticulationCfg = ROBOT.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1500.))
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/base/.*", history_length=3)
    left_foot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/Link_L6", filter_prim_paths_expr=["/World/ground/terrain"],
        track_contact_points=True, max_contact_data_count_per_prim=24,
    )
    right_foot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/Link_R6", filter_prim_paths_expr=["/World/ground/terrain"],
        track_contact_points=True, max_contact_data_count_per_prim=24,
    )


@configclass
class CommandsCfg:
    motion = mdp.MotionCommandCfg(
        asset_name="robot", motion_file=str(Path(__file__).parent / "motions/stand_to_crouch_v2.npz"),
        model_file=str(ASSET / "mjcf/nlegs_limit.xml"),
        anchor_body_name="base", body_names=BODIES, resampling_time_range=(1.e9, 1.e9),
        debug_vis=False, start_probability=.5, sample_until_s=2.28,
        pose_range={"z": (.001, .003), "roll": (-.015, .015), "pitch": (-.015, .015)},
        velocity_range={"x": (-.03, .03), "y": (-.03, .03)}, joint_position_range=(-.01, .01),
    )


@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=.25, use_default_offset=True,
        clip=joint_ranges(ASSET / "mjcf/nlegs_limit.xml"),
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"},
                                    noise=Unoise(n_min=-.02, n_max=.02))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-.1, n_max=.1))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-.01, n_max=.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-.3, n_max=.3))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    physics_material = EventTerm(func=mdp.randomize_rigid_body_material, mode="startup", params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"), "static_friction_range": (.6, 1.3),
        "dynamic_friction_range": (.6, 1.3), "restitution_range": (0., 0.), "num_buckets": 64,
        "make_consistent": True,
    })
    add_base_mass = EventTerm(func=mdp.randomize_rigid_body_mass, mode="startup", params={
        "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        "mass_distribution_params": (1., 4.), "operation": "add",
    })
    base_com = EventTerm(func=mdp.randomize_rigid_body_com, mode="startup", params={
        "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        "com_range": {"x": (-.03, .03), "y": (-.01, .01), "z": (-.02, .02)},
    })
    push_robot = EventTerm(
        func=mdp.PushDuringCrouch, mode="interval", interval_range_s=(0., 0.), is_global_time=True,
        params={"command_name": "motion", "motion_time_range_s": (1.48, 2.28),
                "velocity_range": {"x": (-.10, .10), "y": (-.05, .05)}},
    )  # 每步检查参考进度，不按回合时长周期推扰。


@configclass
class RewardsCfg:
    motion_anchor_pos = RewTerm(func=mdp.motion_global_anchor_position_error_exp, weight=.5,
                               params={"command_name": "motion", "std": .10})
    motion_anchor_ori = RewTerm(func=mdp.motion_global_anchor_orientation_error_exp, weight=.5,
                               params={"command_name": "motion", "std": .35})
    motion_body_pos = RewTerm(func=mdp.motion_relative_body_position_error_exp, weight=1.,
                             params={"command_name": "motion", "std": .08})
    motion_body_ori = RewTerm(func=mdp.motion_relative_body_orientation_error_exp, weight=1.,
                             params={"command_name": "motion", "std": .35})
    motion_body_lin_vel = RewTerm(func=mdp.motion_global_body_linear_velocity_error_exp, weight=.5,
                                 params={"command_name": "motion", "std": 1.})
    motion_body_ang_vel = RewTerm(func=mdp.motion_global_body_angular_velocity_error_exp, weight=.5,
                                 params={"command_name": "motion", "std": 3.14})
    motion_joint_pos = RewTerm(func=mdp.motion_joint_position_error_exp, weight=2.,
                              params={"command_name": "motion", "std": .2})
    loaded_slip = RewTerm(func=mdp.loaded_foot_slip, weight=-5.)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.e-5)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-.02)
    joint_limit = RewTerm(func=mdp.joint_pos_limits, weight=-10.)
    failure = RewTerm(func=mdp.is_terminated, weight=-2.)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.motion_time_out, time_out=True)
    anchor_pos = DoneTerm(func=mdp.bad_anchor_pos, params={"command_name": "motion", "threshold": .25})
    fall = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.05})
    feet_pos = DoneTerm(func=mdp.bad_motion_body_pos_z_only, params={
        "command_name": "motion", "threshold": .15, "body_names": ["Link_L6", "Link_R6"],
    })


@configclass
class NlegsCrouchEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 3.4
        self.sim.dt = .005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        for name in ("contact_forces", "left_foot_contact", "right_foot_contact"):
            getattr(self.scene, name).update_period = self.sim.dt
        self.viewer.eye = (1.5, 1.5, 1.)
        self.viewer.lookat = (0., 0., .3)


@configclass
class NlegsCrouchPlayEnvCfg(NlegsCrouchEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.motion.start_probability = 1.
        self.commands.motion.pose_range = {}
        self.commands.motion.velocity_range = {}
        self.commands.motion.joint_position_range = (0., 0.)
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
