# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import torch
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.managers import SceneEntityCfg

from omni.isaac.orbit_tasks.locomotion.jump.jump_env_cfg import LocomotionJumpEnvCfg

##
# Pre-defined configs
##
from omni.isaac.orbit_assets.solo import SOLO_CFG  # isort: skip


@configclass
class SoloJumpEnvCfg(LocomotionJumpEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = SOLO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Robot params
        trunk_name = "base_link"
        foot_name = "FOOT"
        legs_name = "base_legs"
        robot_height = 0.24
        foot_offset = 0.01

        fl_joint_names = ["FL.*"]
        fl_body_names = ["FL_FOOT"]
        fr_joint_names = ["FR.*"]
        fr_body_names = ["FR_FOOT"]
        rl_joint_names = ["HL.*"]
        rl_body_names = ["HL_FOOT"]
        rr_joint_names = ["HR.*"]
        rr_body_names = ["HR_FOOT"]

        x_limit = 0.15
        y_limit = 0.15
        z_limit = 0.34

        q_0_lo = torch.tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.7854, 0.7854, -0.7854, -0.7854, -1.5708, -1.5708, 1.5708, 1.5708])
        # q_0_lo = torch.tensor([0.0000, 0.0000, 0.0000, 0.0000, 1.5854, 1.5854, -1.5854, -1.5854, 1.5708, 1.5708, -1.5708, -1.5708])

        self.commands.trunk_target.body_name = trunk_name
        self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names=trunk_name)
        self.rewards.target_position_error.params["asset_cfg"] = SceneEntityCfg("robot", body_names=trunk_name)
        self.negative_rewards.target_orientation_error.params["asset_cfg"] = SceneEntityCfg("robot", body_names=trunk_name)

        self.actions.jump_traj.robot_height = robot_height
        self.negative_rewards.apex_z_regularization.params["robot_height"] = robot_height

        self.events.detect_apex.params["foot_height_offset"] = foot_offset
        self.rewards.target_position_error.params["foot_height_offset"] = foot_offset

        self.actions.jump_traj.fl_joint_names = fl_joint_names
        self.actions.jump_traj.fl_body_names = fl_body_names
        self.actions.jump_traj.fr_joint_names = fr_joint_names
        self.actions.jump_traj.fr_body_names = fr_body_names
        self.actions.jump_traj.rl_joint_names = rl_joint_names
        self.actions.jump_traj.rl_body_names = rl_body_names
        self.actions.jump_traj.rr_joint_names = rr_joint_names
        self.actions.jump_traj.rr_body_names = rr_body_names

        self.negative_rewards.singularity_penalty.params["x_limit"] = x_limit
        self.negative_rewards.singularity_penalty.params["y_limit"] = y_limit
        self.negative_rewards.singularity_penalty.params["z_limit"] = z_limit

        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/.*(?:" + foot_name + ")$"
        self.events.detect_apex.params["foot_name"] = ".*" + foot_name
        self.events.detect_touchdown.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=".*" + foot_name)
        self.running_rewards.friction_constraint.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=".*" + foot_name)
        self.rewards.target_position_error.params["foot_name"] = ".*" + foot_name

        self.actions.jump_traj.q_0_lo = q_0_lo
        self.actions.jump_traj.legs_name = legs_name

        self.actions.jump_traj.x_r_min = 0.1
        self.actions.jump_traj.x_r_max = 0.34


@configclass
class SoloJumpEnvCfg_PLAY(SoloJumpEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        mode = "play"

        self.rewards.target_position_error.params["mode"] = mode


@configclass
class SoloJumpEnvCfg_TEST(SoloJumpEnvCfg_PLAY):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        mode = "test"

        self.rewards.target_position_error.params["mode"] = mode
