# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from omni.isaac.orbit.utils import configclass

from .rl_task_env_cfg import RLTaskEnvCfg


@configclass
class RLPlanningTaskEnvCfg(RLTaskEnvCfg):
    """Configuration for a reinforcement learning environment."""

    running_rewards: object = MISSING
    negative_rewards: object = MISSING
    """Running reward settings."""
