# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""nlegs_task 的 PPO runner 配置。

NlegsFlatPPORunnerCfg 不继承 legs_task 的 BasePPORunnerCfg, 所有字段就地列出(与 nlegs
run 2026-08-25_20-27-40 的 agent.yaml 一致); 其他任务变体在其上继承。
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

from legs_rl_lab.tasks.nlegs_task.mdp.symmetry import compute_symmetric_states


@configclass
class MLPActorCfg:
    """Only the fields that rsl-rl 5.x MLPModel.__init__ accepts."""
    class_name: str = "MLPModel"
    hidden_dims: list = None
    activation: str = "elu"
    obs_normalization: bool = False
    distribution_cfg: dict = None


@configclass
class MLPCriticCfg:
    class_name: str = "MLPModel"
    hidden_dims: list = None
    activation: str = "elu"
    obs_normalization: bool = False


@configclass
class NlegsFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    clip_actions = 5.0
    experiment_name = "nlegs_flat"
    actor = MLPActorCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )
    critic = MLPCriticCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=1.0,
            data_augmentation_func=compute_symmetric_states,
        ),
    )


@configclass
class NlegsRoughPPORunnerCfg(NlegsFlatPPORunnerCfg):
    """rough 地形变体：拆小 PPO 更新批次，降低 2490 维 critic 的显存峰值。"""
    experiment_name = "nlegs_rough"

    def __post_init__(self):
        self.algorithm.num_mini_batches = 16


@configclass
class NlegsRoughStepPPORunnerCfg(NlegsFlatPPORunnerCfg):
    """专用上台阶变体，仅日志实验名不同。

    critic 的 height_scan 只用当前帧（观测维度 807），镜像布局由 mdp/symmetry.py
    按维度自动匹配，故算法配置与 flat 完全一致。
    """
    experiment_name = "nlegs_rough_step"


@configclass
class NlegsRoughInfoPPORunnerCfg(NlegsFlatPPORunnerCfg):
    """非盲走变体：actor 也吃 176 点高度图（观测维度 646，critic 796）。

    网络结构刻意与 flat/rough 保持一致（512-256-128），这样和 nlegs_rough 的对比里
    唯一变量就是"有没有地形信息"；镜像布局由 mdp/symmetry.py 按维度自动匹配。
    """
    experiment_name = "nlegs_rough_info"
