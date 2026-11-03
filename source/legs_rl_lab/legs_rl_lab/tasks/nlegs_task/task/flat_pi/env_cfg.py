"""Pi release_v1.0.0 training settings, with W1196 geometry and actuators.

Reference: HighTorque-Robotics/Mini_Pi_RL_Baseline, e8b787afa1e30da4f28f8cd7fefe542a3912ac27.
The task is independent of the flat/rough task configurations.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from legs_rl_lab.assets.nlegs_v2.nlegs_v2 import (
    W1196_CFG, W1196_FEET_WIDTH, W1196_FOOT_HEIGHT_OFFSET, W1196_NOMINAL_BASE_HEIGHT,
)


PI_MASS = 6.93973513
V2_MASS = 10.8311180249
MASS_RATIO = V2_MASS / PI_MASS
WIDTH_RATIO = W1196_FEET_WIDTH / 0.1602


@configclass
class PiGaitCfg:
    period: float = 0.4


@configclass
class PiRewardsCfg:
    base_height_target: float = W1196_NOMINAL_BASE_HEIGHT
    min_dist: float = 0.15 * WIDTH_RATIO
    max_dist: float = 0.2 * WIDTH_RATIO
    foot_height_offset: float = W1196_FOOT_HEIGHT_OFFSET
    clearance_height_offset: float = W1196_FOOT_HEIGHT_OFFSET + 0.05
    target_joint_pos_scale: float = 0.08
    target_feet_height: float = 0.06
    tracking_sigma_lin: float = 0.1
    tracking_sigma_ang: float = 0.1
    max_contact_force: float = 100.0 * MASS_RATIO
    contact_threshold: float = 5.0 * MASS_RATIO
    clearance_contact_threshold: float = 0.1 * MASS_RATIO
    collision_threshold: float = 0.1 * MASS_RATIO
    only_positive_rewards: bool = True
    scales: dict = {
        "joint_pos": 1.6,
        "feet_clearance": 5.0,
        "feet_contact_number": 1.2,
        "feet_air_time": 1.0,
        "foot_slip": -0.05,
        "feet_distance": 0.16,
        "knee_distance": 0.16,
        "feet_contact_forces": -0.001,
        "tracking_lin_vel": 10.0,
        "tracking_ang_vel": 20.0,
        "vel_mismatch_exp": 0.5,
        "low_speed": 0.05,
        "track_vel_hard": 0.2,
        "default_hip_roll_joint_pos": 4.0,
        "default_thigh_joint_pos": 1.0,
        "default_ankle_roll_pos": 0.5,
        "orientation": 0.5,
        "base_height": 0.5,
        "base_acc": 0.2,
        "action_smoothness": -0.002,
        "torques": -1.0e-5,
        "dof_vel": -5.0e-5,
        "dof_acc": -1.0e-8,
        "collision": -1.0,
        "termination": 1.0,
    }


@configclass
class PiSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.6, dynamic_friction=0.6, restitution=0.0,
            friction_combine_mode="average", restitution_combine_mode="average",
        ),
    )
    robot = W1196_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/.*", update_period=0.0, history_length=1,
    )
    light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )


@configclass
class PiEnvCfg(DirectRLEnvCfg):
    task_type: str = "nlegs_v2_pi"
    seed: int = 5
    decimation: int = 20
    episode_length_s: float = 12.0
    action_space: int = 12
    observation_space: int = 705
    state_space: int = 219
    actor_history: int = 15
    critic_history: int = 3
    action_scale: float = 0.25
    clip_actions: float = 18.0
    clip_observations: float = 18.0
    robot_contact_offset: float = 0.01
    robot_rest_offset: float = 0.0
    scene: PiSceneCfg = PiSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.001, render_interval=20,
        physx=sim_utils.PhysxCfg(solver_type=1, bounce_threshold_velocity=0.1),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.6, dynamic_friction=0.6, restitution=0.0,
            friction_combine_mode="average", restitution_combine_mode="average",
        ),
    )
    gait: PiGaitCfg = PiGaitCfg()
    rewards: PiRewardsCfg = PiRewardsCfg()
    # Original Pi heading controller clips yaw to +/-1, despite its +/-0.3 range field.
    heading_command: bool = True
    heading_gain: float = 0.5
    heading_yaw_limit: float = 1.0
    command_ranges: dict = {
        "lin_vel_x": (-0.3, 0.6), "lin_vel_y": (-0.3, 0.3),
        "ang_vel_z": (-0.3, 0.3), "heading": (-3.14, 3.14),
    }
    command_resampling_time: float = 8.0
    small_command_threshold: float = 0.2
    noise_enabled: bool = True
    noise_level: float = 0.6
    # Standard deviations before applying noise_level, already observation-scaled.
    noise_scales: dict = {"joint_pos": 0.05, "joint_vel": 0.5 * 0.05, "ang_vel": 0.1, "euler": 0.03}
    randomization: bool = True
    friction_range: tuple = (0.1, 2.0)
    friction_buckets: int = 256
    added_base_mass_range: tuple = (-1.0, 1.0)
    push_interval_s: float = 4.0
    max_push_vel_xy: float = 0.2
    max_push_ang_vel: float = 0.4
    action_randomization: bool = True
    dynamic_randomization: float = 0.02
    reset_joint_position_range: tuple = (-0.1, 0.1)
    termination_height: float = 0.3 / 0.3453 * W1196_NOMINAL_BASE_HEIGHT
    termination_contact_threshold: float = 1.0 * MASS_RATIO
    termination_gravity_xy: float = 0.8

    def __post_init__(self):
        # Convert the existing 5--25 ms delay to this task's 1 ms physics ticks.
        for actuator in self.scene.robot.actuators.values():
            actuator.min_delay = 5
            actuator.max_delay = 25
        self.viewer.eye = (2.5, 2.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)

@configclass
class PiPlayEnvCfg(PiEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64


@configclass
class PiActorCfg:
    class_name: str = "MLPModel"
    hidden_dims: list = [512, 256, 128]
    activation: str = "elu"
    obs_normalization: bool = False
    distribution_cfg: dict = {"class_name": "GaussianDistribution", "init_std": 1.0}


@configclass
class PiCriticCfg:
    class_name: str = "MLPModel"
    hidden_dims: list = [768, 256, 128]
    activation: str = "elu"
    obs_normalization: bool = False


@configclass
class PiPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyRunner"
    seed = 5
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 10001
    save_interval = 100
    # Pi clips AFTER its random action blend inside the environment.
    clip_actions = None
    check_for_nan = True
    experiment_name = "nlegs_v2_pi"
    run_name = "v1"
    logger = "tensorboard"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
    actor = PiActorCfg()
    critic = PiCriticCfg()
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO", value_loss_coef=1.0, use_clipped_value_loss=True,
        clip_param=0.2, entropy_coef=0.001, num_learning_epochs=2, num_mini_batches=4,
        learning_rate=1.0e-5, schedule="adaptive", gamma=0.994, lam=0.9,
        desired_kl=0.01, max_grad_norm=1.0, symmetry_cfg=None,
    )
