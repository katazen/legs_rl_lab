"""nlegs_flat_crouch：同一行走策略从站立/下蹲起步，不改变原 flat/rough。"""

from pathlib import Path

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from legs_rl_lab.assets.nlegs import nlegs
from legs_rl_lab.tasks.nlegs_task import mdp
from legs_rl_lab.tasks.nlegs_task.agents.rsl_rl_ppo_cfg import NlegsFlatPPORunnerCfg
from .flat_env_cfg import FlatEnvCfg


@configclass
class FlatCrouchEnvCfg(FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.usd_path = str(
            Path(nlegs.__file__).parent / "mjcf/nlegs_limit/nlegs_limit.usd"
        )
        self.events.reset_base = None
        self.events.reset_robot_joints = None
        self.events.reset_crouch = EventTerm(
            func=mdp.reset_from_crouch_pose_bank,
            mode="reset",
            params={
                "bank_path": str(Path(nlegs.__file__).parent / "crouch_pose_bank.json"),
                "stand_probability": 0.3,
                "crouch_probability": 0.4,
            },
        )
        self.curriculum = None
        self.rewards.feet_slide.weight = -1.0


@configclass
class FlatCrouchPlayEnvCfg(FlatCrouchEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32


@configclass
class NlegsFlatCrouchPPORunnerCfg(NlegsFlatPPORunnerCfg):
    experiment_name = "nlegs_flat_crouch"
