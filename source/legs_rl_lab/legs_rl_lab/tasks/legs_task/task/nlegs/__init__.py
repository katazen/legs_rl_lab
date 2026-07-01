import gymnasium as gym

gym.register(
    id="nlegs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nlegs_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.nlegs_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "legs_rl_lab.tasks.legs_task.agents.rsl_rl_ppo_cfg:NlegsPPORunnerCfg",
    },
)
