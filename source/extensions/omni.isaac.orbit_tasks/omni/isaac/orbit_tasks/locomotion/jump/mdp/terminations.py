
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv
    from omni.isaac.orbit.managers.command_manager import CommandTerm


"""
Contact sensor.
"""


# def touch_down(env: RLTaskEnv, air_time_threshold: float, contact_threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Terminate when the contact force on the sensor exceeds the force threshold."""
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     net_contact_forces = contact_sensor.data.net_forces_w
#     net_last_air_time = contact_sensor.data.last_air_time

#     flew_env_ids = torch.all(net_last_air_time[:, sensor_cfg.body_ids] > air_time_threshold, dim=1)
#     in_contact_env_ids = torch.all(torch.norm(net_contact_forces[:, sensor_cfg.body_ids], dim=-1) > contact_threshold, dim=1)

#     touchdown_env = flew_env_ids & in_contact_env_ids

#     return touchdown_env
