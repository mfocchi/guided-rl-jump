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
    id="Isaac-Jump-Unitree-Aliengo-v0",
    entry_point="omni.isaac.orbit.envs:RLPlanningTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": jump_platform_env_cfg.UnitreeAliengoJumpEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeAliengoPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_td3_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Jump-Unitree-Aliengo-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLPlanningTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": jump_platform_env_cfg.UnitreeAliengoJumpEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeAliengoPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_td3_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Jump-Unitree-Aliengo-Test-v0",
    entry_point="omni.isaac.orbit.envs:RLPlanningTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": jump_platform_env_cfg.UnitreeAliengoJumpEnvCfg_TEST,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeAliengoPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_td3_cfg.yaml",
    },
)