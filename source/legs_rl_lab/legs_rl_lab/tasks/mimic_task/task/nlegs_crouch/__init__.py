import gymnasium as gym

gym.register(
    id="nlegs_mimic_crouch",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:NlegsCrouchEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:NlegsCrouchPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "legs_rl_lab.tasks.mimic_task.agents.rsl_rl_ppo_cfg:NlegsCrouchPPORunnerCfg",
    },
)
