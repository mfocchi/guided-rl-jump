# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, jump_platform_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Jump-Unitree-Go2-v0",
    entry_point="omni.isaac.orbit.envs:RLPlanningTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": jump_platform_env_cfg.UnitreeGo2JumpEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeGo2PPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_td3_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Jump-Unitree-Go2-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLPlanningTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": jump_platform_env_cfg.UnitreeGo2JumpEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeGo2PPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_td3_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Jump-Unitree-Go2-Test-v0",
    entry_point="omni.isaac.orbit.envs:RLPlanningTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": jump_platform_env_cfg.UnitreeGo2JumpEnvCfg_TEST,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeGo2PPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_td3_cfg.yaml",
    },
)
