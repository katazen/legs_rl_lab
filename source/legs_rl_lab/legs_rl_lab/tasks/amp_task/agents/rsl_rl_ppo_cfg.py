# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

from legs_rl_lab.tasks.amp_task.mdp.symmetry import compute_symmetric_states

import os

# 专家动作数据目录 (convert_tienkung_motion.py 的输出 nlegs_walk.txt 所在处)
_AMP_MOTION_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets", "motion_amp_expert",
)


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
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "legs"
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
class AmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """在 PPO 配置上追加 AMP 专属字段。

    class_name 指向本仓的 AMPPPO(继承官方 PPO); to_dict() 后这些 amp_* 字段随 cfg["algorithm"]
    一起传入, 由 AMPPPO.construct_algorithm 弹出并构造判别器/专家数据/归一化器/回放缓存。
    数值默认对齐 TienKung walk (reward_coef=0.3, task_reward_lerp=0.7, 判别器 [1024,512,256])。
    """
    class_name: str = "legs_rl_lab.amp.amp_ppo.AMPPPO"
    amp_reward_coef: float = 0.3
    amp_reward_lerp: float = 0.7          # 最终 reward = (1-lerp)*style + lerp*task
    amp_discr_hidden_dims: list = None
    amp_replay_buffer_size: int = 100000
    amp_learning_rate: float = 1.0e-3     # 判别器独立优化器, 固定 lr(不随 PPO 的 KL 自适应)
    amp_grad_pen_lambda: float = 10.0
    amp_num_preload_transitions: int = 200000
    amp_motion_files: list = None         # None -> 用 amp_data_dir 下所有文件
    amp_data_dir: str = ""


@configclass
class NlegsAmpPPORunnerCfg(BasePPORunnerCfg):
    """nlegs + AMP: 复用 Base 的 actor/critic/超参, 算法换成 AMPPPO。"""
    experiment_name = "nlegs_amp"
    algorithm = AmpPpoAlgorithmCfg(
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
        amp_discr_hidden_dims=[1024, 512, 256],
        amp_data_dir=_AMP_MOTION_DIR,
    )