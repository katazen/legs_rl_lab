import gymnasium as gym

gym.register(
    id="nlegs_rough_static",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.static_env_cfg:RoughStaticEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.static_env_cfg:RoughStaticPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.static_env_cfg:NlegsRoughStaticPPORunnerCfg",
    },
)

gym.register(
    id="nlegs_rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:RoughEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.rough_env_cfg:RoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "legs_rl_lab.tasks.nlegs_task.agents.rsl_rl_ppo_cfg:NlegsRoughPPORunnerCfg",
    },
)
