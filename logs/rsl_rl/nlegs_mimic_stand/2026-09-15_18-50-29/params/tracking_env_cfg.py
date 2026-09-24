"""下蹲到站立：复用下蹲 Mimic，只覆盖参考动作和对应时序。"""

from pathlib import Path

from isaaclab.utils import configclass

from ..nlegs_crouch.tracking_env_cfg import NlegsCrouchEnvCfg


@configclass
class NlegsStandEnvCfg(NlegsCrouchEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = str(Path(__file__).parent / "motions/crouch_to_stand_v1.npz")
        self.commands.motion.sample_until_s = 2.08
        self.events.push_robot.params["motion_time_range_s"] = (.4, 2.08)
        self.episode_length_s = 2.68


@configclass
class NlegsStandPlayEnvCfg(NlegsStandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.motion.start_probability = 1.
        self.commands.motion.pose_range = {}
        self.commands.motion.velocity_range = {}
        self.commands.motion.joint_position_range = (0., 0.)
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
