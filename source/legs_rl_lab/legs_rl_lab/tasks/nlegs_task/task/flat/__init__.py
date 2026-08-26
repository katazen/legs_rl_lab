import gymnasium as gym

gym.register(
    id="nlegs_flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:FlatEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.flat_env_cfg:FlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "legs_rl_lab.tasks.nlegs_task.agents.rsl_rl_ppo_cfg:NlegsFlatPPORunnerCfg",
    },
)
