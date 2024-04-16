# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.sensors import ContactSensor
from omni.isaac.orbit.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


def computeActivationFunction(activationType, values, lower, upper):
    if activationType == 'linear':
        return torch.abs(torch.min(values - lower, torch.tensor(0.0))) + torch.max(values - upper, torch.tensor(0.0))
    elif activationType == 'quadratic':
        return torch.pow(torch.min(values - lower, torch.tensor(0.0)), 2) / 2.0 + torch.pow(torch.max(values - upper, torch.tensor(0.0)), 2) / 2.0
    else:
        raise ValueError("Invalid activation type")


def target_position_error(env: RLTaskEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore

    # TODO:experiment with this function
    # return 1.0 / torch.norm(curr_pos_w - des_pos_w, dim=1)
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def target_orientation_error(env: RLTaskEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore

    return quat_error_magnitude(curr_quat_w, des_quat_w)


def friction_constraint(env: RLTaskEnv, sensor_cfg: SceneEntityCfg, mu: float = 0.8) -> torch.Tensor:
    """Penalize contact forces out of the friction cone

    Args:
        sensor_cfg: The contact sensor configuration
        mu: the friction coefficient
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w
    # compute out of limits constraints

    torch.norm(net_contact_forces[:, sensor_cfg.body_ids], dim=-1)

    # From jumpleg -> residual = np.linalg.norm(self.contactForceW[:2]) - p.mu * p.contactForceW[2]
    residuals = torch.norm(net_contact_forces[:, sensor_cfg.body_ids, :2], dim=-1) - mu * net_contact_forces[:, sensor_cfg.body_ids, 2]

    # Compute the violation values of friction cone constraint
    #  #evns,#sensors (n, 4)
    # sum along each robot to get the total violation cost
    costs = torch.sum(computeActivationFunction('linear', residuals, -torch.inf, 0.0), dim=1)

    return costs


def feet_contact_time(env: RLTaskEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    reward = torch.sum(contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids], dim=1)

    return reward


def air_time(env: RLTaskEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    reward = torch.sum(contact_sensor.data.current_air_time[:, sensor_cfg.body_ids], dim=1)

    return reward
