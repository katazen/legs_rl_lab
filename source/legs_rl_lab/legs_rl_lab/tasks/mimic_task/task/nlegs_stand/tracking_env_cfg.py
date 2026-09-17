"""实测蹲姿到站立：复用下蹲 Mimic，使用抬脚 2 cm、放慢后的 v2 动作。"""

from pathlib import Path

from isaaclab.utils import configclass

from ..nlegs_crouch.tracking_env_cfg import NlegsCrouchEnvCfg


@configclass
class NlegsStandEnvCfg(NlegsCrouchEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = str(Path(__file__).parent / "motions/crouch_to_stand_v2.npz")
        self.commands.motion.sample_until_s = 2.6
        self.commands.motion.track_heading = False
        self.commands.motion.end_hold_s = 5.0
        self.events.push_robot.params["motion_time_range_s"] = (.5, 2.6)
        self.episode_length_s = 3.35 + self.commands.motion.end_hold_s


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
