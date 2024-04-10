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

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


def computeActivationFunction(activationType, values, lower, upper):
    if activationType == 'linear':
        return torch.abs(torch.min(values - lower, torch.tensor(0.0))) + torch.max(values - upper, torch.tensor(0.0))
    elif activationType == 'quadratic':
        return torch.pow(torch.min(values - lower, torch.tensor(0.0)), 2) / 2.0 + torch.pow(torch.max(values - upper, torch.tensor(0.0)), 2) / 2.0
    else:
        raise ValueError("Invalid activation type")


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
