# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`omni.isaac.orbit.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
import numpy as np
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.terrains import TerrainImporter

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


def jump_curriculum(env: RLTaskEnv, env_ids: Sequence[int], term_name: str, start: float, num_steps: int, num_steps_rp: int, pos_x, pos_y, pos_z, roll, pitch, yaw, activate: bool = False):

    if activate:
        if env.common_step_counter <= num_steps:
            coeff = start + np.clip((env.common_step_counter / num_steps) - start, 0, 1 - start)

            curr_pos_x = coeff * np.array(pos_x)
            curr_pos_y = coeff * np.array(pos_y)
            curr_pos_z = coeff * np.array(pos_z)
            

            env.command_manager.get_term(term_name).cfg.ranges.pos_x = tuple(curr_pos_x)
            env.command_manager.get_term(term_name).cfg.ranges.pos_y = tuple(curr_pos_y)
            env.command_manager.get_term(term_name).cfg.ranges.pos_z = tuple(curr_pos_z)

        elif env.common_step_counter <= num_steps_rp:

            coeff = np.clip(((env.common_step_counter - num_steps) / (num_steps_rp - num_steps)), 0, 1)
            
            curr_roll = coeff * np.array(roll)
            curr_pitch = coeff * np.array(pitch)
            curr_yaw = coeff * np.array(yaw)

            env.command_manager.get_term(term_name).cfg.ranges.roll = tuple(curr_roll)
            env.command_manager.get_term(term_name).cfg.ranges.pitch = tuple(curr_pitch)
            env.command_manager.get_term(term_name).cfg.ranges.yaw = tuple(curr_yaw)

        print("Term: ", env.command_manager.get_term(term_name).cfg)
