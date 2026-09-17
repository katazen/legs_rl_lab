"""Original mimic PPO settings adapted to this project's RSL-RL 5.x interface."""

from isaaclab.utils import configclass
from legs_rl_lab.tasks.nlegs_task.agents.rsl_rl_ppo_cfg import NlegsFlatPPORunnerCfg


@configclass
class NlegsCrouchPPORunnerCfg(NlegsFlatPPORunnerCfg):
    experiment_name = "nlegs_mimic_crouch"
    max_iterations = 30000

    def __post_init__(self):
        self.algorithm.symmetry_cfg = None
        self.algorithm.entropy_coef = 0.005
        self.actor.distribution_cfg["init_std"] = 0.3


@configclass
class NlegsStandPPORunnerCfg(NlegsCrouchPPORunnerCfg):
    experiment_name = "nlegs_mimic_stand"
