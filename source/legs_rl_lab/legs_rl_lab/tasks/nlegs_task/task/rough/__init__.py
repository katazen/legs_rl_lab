import gymnasium as gym

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

# 专用上台阶任务: 只有反金字塔台阶(坑, 四面上行) + 少量平地, 命令退化为恒定前进 + 朝向纠偏
gym.register(
    id="nlegs_rough_step",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_step_env_cfg:RoughStepEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.rough_step_env_cfg:RoughStepPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "legs_rl_lab.tasks.nlegs_task.agents.rsl_rl_ppo_cfg:NlegsRoughStepPPORunnerCfg",
    },
)

# 非盲走 rough: 楼梯比例 15% -> 40%, actor 也吃当前帧高度图(176 点), 保留完整速度跟踪命令
gym.register(
    id="nlegs_rough_info",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_info_env_cfg:RoughInfoEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.rough_info_env_cfg:RoughInfoPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "legs_rl_lab.tasks.nlegs_task.agents.rsl_rl_ppo_cfg:NlegsRoughInfoPPORunnerCfg",
    },
)
