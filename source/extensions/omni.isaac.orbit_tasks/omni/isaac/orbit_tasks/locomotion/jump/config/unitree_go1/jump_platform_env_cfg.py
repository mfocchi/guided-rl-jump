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
from omni.isaac.orbit_assets.unitree import UNITREE_GO1_CFG  # isort: skip


@configclass
class UnitreeGo1JumpEnvCfg(LocomotionJumpEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Robot params
        trunk_name = "trunk"
        foot_name = "foot"
        legs_name = "base_legs"
        legs_name_calf = "base_legs_calf"
        thigh_name = "thigh"
        robot_height = 0.31
        foot_offset = 0.02

        fl_joint_names = ["FL.*"]
        fl_body_names = ["FL_foot"]
        fr_joint_names = ["FR.*"]
        fr_body_names = ["FR_foot"]
        rl_joint_names = ["RL.*"]
        rl_body_names = ["RL_foot"]
        rr_joint_names = ["RR.*"]
        rr_body_names = ["RR_foot"]

        x_limit = 0.15
        y_limit = 0.15
        z_limit = 0.4

        q_0_lo = torch.tensor([0.2316, -0.2316, 0.2343, -0.2338, 1.3818, 1.3816, 1.6860, 1.6858, -2.4620, -2.4617, -2.3162, -2.3163])

        mass_range = 0
        stiffness_division = 1

        self.commands.trunk_target.body_name = trunk_name
        # self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names=trunk_name)
        self.rewards.target_position_error.params["asset_cfg"] = SceneEntityCfg("robot", body_names=trunk_name)
        self.negative_rewards.target_orientation_error.params["asset_cfg"] = SceneEntityCfg("robot", body_names=trunk_name)
        self.negative_rewards.touchdown_bounce_penalization.params["asset_cfg"] = SceneEntityCfg("robot", body_names=trunk_name)

        self.actions.jump_traj.robot_height = robot_height
        # self.negative_rewards.apex_z_regularization.params["robot_height"] = robot_height

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
        self.scene.nonfoot_forces.prim_path = "{ENV_REGEX_NS}/Robot/.*?(" + thigh_name + "|" + trunk_name + ").*"
        self.actions.jump_traj.sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*")
        self.events.detect_apex.params["foot_name"] = ".*" + foot_name
        self.events.detect_touchdown.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=".*")
        self.events.detect_fail.params["sensor_cfg"] = SceneEntityCfg("nonfoot_forces", body_names=".*")
        self.running_rewards.friction_constraint.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=".*")
        self.rewards.target_position_error.params["foot_name"] = ".*" + foot_name
        self.events.physics_material.params["asset_cfg"] = SceneEntityCfg("robot", body_names=".*")

        self.actions.jump_traj.q_0_lo = q_0_lo
        self.actions.jump_traj.legs_name = legs_name
        self.actions.jump_traj.legs_name_calf = legs_name_calf

        # self.events.add_base_mass.params["mass_range"] = (-mass_range, mass_range)
        self.actions.jump_traj.stiffness_division = stiffness_division


@configclass
class UnitreeGo1JumpEnvCfg_PLAY(UnitreeGo1JumpEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        mode = "play"

        self.rewards.target_position_error.params["mode"] = mode
        self.actions.jump_traj.mode = mode
        self.episode_length_s = 2
        self.curriculum.jump_complexity.params["activate"] = False
        self.commands.trunk_target.roll_yaw_shufle = False


@configclass
class UnitreeGo1JumpEnvCfg_TEST(UnitreeGo1JumpEnvCfg_PLAY):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        mode = "test"

        self.rewards.target_position_error.params["mode"] = mode
        self.actions.jump_traj.mode = mode
        self.episode_length_s = 2
        self.curriculum.jump_complexity.params["activate"] = False
        self.commands.trunk_target.roll_yaw_shufle = False
