import gymnasium as gym

# EngineAI 风格 AMP 走路任务: 与 nlegs_amp 同任务, 换 AMP 算法(history-window)+数据(engineai)。
gym.register(
    id="nlegs_amp_engineai",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nlegs_env_cfg:RobotEngineaiEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.nlegs_env_cfg:RobotEngineaiPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "legs_rl_lab.tasks.amp_task.agents.rsl_rl_ppo_cfg:NlegsAmpEngineaiPPORunnerCfg",
    },
)
