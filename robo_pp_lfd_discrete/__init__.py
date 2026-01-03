"""LfD polishing package with discrete action controller."""

from gymnasium.envs.registration import register

from .Polish_Env_LfD_Discrete import PolishLfDDiscreteEnv, PolishLfDDiscreteEnvCfg

register(
    id="Robo_PP-LfD-Discrete-v0",
    entry_point="vttRL.tasks.robo_pp_lfd_discrete.Polish_Env_LfD_Discrete:PolishLfDDiscreteEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "vttRL.tasks.robo_pp_lfd_discrete.Polish_Env_LfD_Discrete:PolishLfDDiscreteEnvCfg",
        "rl_games_cfg_entry_point": "vttRL.tasks.robo_pp_lfd_discrete.agents:rl_games_ppo_cfg.yaml",
    },
)

__all__ = ["PolishLfDDiscreteEnv", "PolishLfDDiscreteEnvCfg"]
