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


def target_position_error(env: RLTaskEnv, command_name: str, asset_cfg: SceneEntityCfg, z_threshold: float = 0.1, coeff: float = 1., dist_coeff: float = 1., err_coeff: float = 1., bias: float = 1) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # obtain the desired and current positions

    target_distance = torch.norm(env.command_manager.get_command(command_name)[:, 0:3], dim=1) + 1e-12

    des_pos_w = env.extras["trunk_tg"] + env.scene.env_origins
    des_pos_w[..., 2] -= asset.data.default_root_state[..., 2]

    foot_idx = asset.find_bodies(".*foot")[0]

    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore

    # Calculate foot center, remove foot padding (2cm)
    foot_pos_center = (torch.sum(asset.data.body_state_w[:, foot_idx, 2], dim=1) / 4) - 0.02

    curr_pos_w[..., 2] = foot_pos_center

    # print(f"Target: {des_pos_w[...,:3]}")
    # print(f"Landing: {curr_pos_w[...,:3]}")

    # Calculate percentual_error to normalize jump performance
    target_z_error = torch.abs(des_pos_w[..., 2] - curr_pos_w[..., 2])
    target_z_error = torch.where(target_z_error <= z_threshold, torch.tensor(0.0), target_z_error)

    target_error = torch.norm(des_pos_w[..., :2] - curr_pos_w[..., :2], dim=1) + target_z_error

    # the norm of the target becaus is alredy relative
    # jump_error = target_error * torch.exp(-target_distance)

    cost = 1.0 / ((coeff * target_error) + 1e-12)
    cost = torch.log(1 + cost)
    cost = torch.clip((((cost + (dist_coeff * torch.exp(target_distance))) * torch.pow((1 - target_error), err_coeff)) - bias), 0, torch.inf)

    # print(f"Avg jump_error: {target_error.mean()}")
    env.extras["avg_abs_err"] = target_error.mean()
    env.extras["avg_rpe_err"] = (target_error / target_distance).mean()

    # TODO:experiment with this function
    # cost = (1.0 / ((50 * (percentual_error ** 3)) + 1e-15)) - 0.02
    return cost
    # return torch.tanh(cost)
    # return torch.norm(curr_pos_w - des_pos_w, dim=1)


def target_orientation_error(env: RLTaskEnv, command_name: str, asset_cfg: SceneEntityCfg, coeff: float = 1., dist_coeff: float = 1., err_coeff: float = 1., bias: float = 1) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    target_distance = torch.norm(env.command_manager.get_command(command_name)[:, 0:3], dim=1) + 1e-12

    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    # TODO: add quaternion conversion
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore

    target_error = quat_error_magnitude(curr_quat_w, des_quat_w)

    # print(target_error)

    # cost = 1.0 / ((coeff * target_error) + 1e-12)
    # cost = torch.log(1 + cost)
    # cost = torch.clip((((cost + (dist_coeff * torch.exp(target_distance))) * torch.pow((1 - target_error), err_coeff)) - bias), 0, torch.inf)

    # return cost
    return target_error


def liftoff_z_regularization(env: RLTaskEnv, limit: float = 0.3) -> torch.Tensor:

    # des_lo_z = env.extras["trunk_x_lo"][..., 2]
    des_lo_z = env.extras["trunk_x_exp"][..., 2]
    return torch.square(des_lo_z - limit)


def liftoff_position_error(env: RLTaskEnv) -> torch.Tensor:
    # obtain the desired and current positions

    # des_lo_pos_w = env.extras["trunk_x_lo"] + env.scene.env_origins
    des_lo_pos_w = env.extras["trunk_x_exp"] + env.scene.env_origins
    curr_lo_pos_w = env.extras["actual_lo_config"][..., 0:3]

    return torch.norm(des_lo_pos_w - curr_lo_pos_w, dim=1)


def liftoff_orientation_error(env: RLTaskEnv) -> torch.Tensor:
    # obtain the desired and current positions

    des_lo_o_w = env.extras["trunk_o_lo"]
    curr_lo_o_w = env.extras["actual_lo_config"][..., 3:7]

    target_error = quat_error_magnitude(des_lo_o_w, curr_lo_o_w)

    return target_error


def liftoff_linear_velocity_error(env: RLTaskEnv) -> torch.Tensor:
    # obtain the desired and current positions

    # des_lo_lvel_w = env.extras["trunk_xd_lo"]
    des_lo_lvel_w = env.extras["trunk_xd_exp"]
    curr_lo_lvel_w = env.extras["actual_lo_config"][..., 7:10]

    return torch.norm(des_lo_lvel_w - curr_lo_lvel_w, dim=1)


def liftoff_angular_velocity_error(env: RLTaskEnv) -> torch.Tensor:
    # obtain the desired and current positions

    des_lo_lvel_w = env.extras["trunk_od_lo"]
    curr_lo_lvel_w = env.extras["actual_lo_config"][..., 10:13]

    return torch.norm(des_lo_lvel_w - curr_lo_lvel_w, dim=1)


def friction_constraint(env: RLTaskEnv, sensor_cfg: SceneEntityCfg, mu: float = 0.8) -> torch.Tensor:
    """Penalize contact forces out of the friction cone

    Args:
        sensor_cfg: The contact sensor configuration
        mu: the friction coefficient
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w

    # From jumpleg -> residual = np.linalg.norm(self.contactForceW[:2]) - p.mu * p.contactForceW[2]
    residuals = torch.norm(net_contact_forces[:, sensor_cfg.body_ids, :2], dim=-1) - mu * net_contact_forces[:, sensor_cfg.body_ids, 2]

    # Compute the violation values of friction cone constraint
    #  #evns,#sensors (n, 4)
    # sum along each robot to get the total violation cost
    costs = torch.sum(computeActivationFunction('linear', residuals, -torch.inf, 0.0), dim=1)

    return costs


def contact_constraint(env: RLTaskEnv, sensor_cfg: SceneEntityCfg, contact_threshold: float = 1) -> torch.Tensor:

    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w

    costs = ~torch.all(torch.norm(net_contact_forces[:, sensor_cfg.body_ids], dim=-1) > contact_threshold, dim=1).reshape(-1, 1)
    costs = torch.sum(costs, dim=1)

    return costs


def unilateral_constraint(env: RLTaskEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces out of the friction cone

    Args:
        sensor_cfg: The contact sensor configuration
        mu: the friction coefficient
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w

    # sum along each robot to get the total violation cost
    costs = torch.sum(computeActivationFunction('linear', net_contact_forces[:, sensor_cfg.body_ids, 2], 0.0, torch.inf), dim=1)

    return costs


def no_touchdown(env: RLTaskEnv) -> torch.Tensor:
    """
    Penalize env where no touchdown is reached
    """

    no_touchdown_penalty = torch.ones(env.num_envs, device=env.device)

    touchdown_end_ids = torch.tensor(list(env.extras['touchdown'].keys()), device=env.device, dtype=torch.int)
    no_touchdown_penalty[touchdown_end_ids] = 0

    return no_touchdown_penalty


def action_regularization(env: RLTaskEnv, action: int, limit: float = 0.0) -> torch.Tensor:
    """Penalize big action to constrain the range"""
    return torch.square(env.action_manager.action[..., action] - limit)


def action_limit_penalization(env: RLTaskEnv, min_action, max_action) -> torch.Tensor:
    """Penalize big action to constrain the range"""
    return torch.sum(computeActivationFunction('linear', env.action_manager.action, min_action, max_action), dim=1)
