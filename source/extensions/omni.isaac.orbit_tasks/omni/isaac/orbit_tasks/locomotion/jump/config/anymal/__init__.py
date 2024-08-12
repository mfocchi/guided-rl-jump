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
    id="Isaac-Jump-Anymal-v0",
    entry_point="omni.isaac.orbit.envs:RLPlanningTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": jump_platform_env_cfg.AnymalJumpEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_td3_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Jump-Anymal-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLPlanningTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": jump_platform_env_cfg.AnymalJumpEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_td3_cfg.yaml",
    },
)
