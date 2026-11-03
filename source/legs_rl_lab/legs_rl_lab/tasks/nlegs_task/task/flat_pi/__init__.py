"""Independent Mini Pi style task for the nlegs v2 robot."""

import gymnasium as gym

gym.register(
    id="nlegs_v2_pi",
    entry_point=f"{__name__}.env:PiEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PiEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.env_cfg:PiPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg:PiPPORunnerCfg",
    },
)
