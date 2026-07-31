import gymnasium as gym

# amp_task 专注 AMP 走路任务:只注册 nlegs_amp(自包含, 见 nlegs_env_cfg.py)。
gym.register(
    id="nlegs_amp",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nlegs_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.nlegs_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "legs_rl_lab.tasks.amp_task.agents.rsl_rl_ppo_cfg:NlegsAmpPPORunnerCfg",
    },
)
