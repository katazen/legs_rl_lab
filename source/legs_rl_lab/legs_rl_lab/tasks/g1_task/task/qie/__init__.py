import gymnasium as gym

gym.register(
    id="g1qie",
    entry_point=f"{__name__}.qie_env:QieEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qie_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.qie_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"legs_rl_lab.tasks.g1_task.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
