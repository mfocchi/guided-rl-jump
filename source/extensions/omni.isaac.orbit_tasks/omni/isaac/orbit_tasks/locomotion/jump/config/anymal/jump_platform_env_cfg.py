# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.utils import configclass

from omni.isaac.orbit_tasks.locomotion.jump.jump_env_cfg_anymal import LocomotionJumpEnvCfg

##
# Pre-defined configs
##
from omni.isaac.orbit_assets import ANYMAL_C_CFG

@configclass
class AnymalJumpEnvCfg(LocomotionJumpEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")



@configclass
class AnymalJumpEnvCfg_PLAY(AnymalJumpEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # disable randomization for play
        self.observations.policy.enable_corruption = False