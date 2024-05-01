from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import quat_from_euler_xyz
from omni.isaac.orbit.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


def reset_robot_state(
    env: RLTaskEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to the default position and velocity.

    This function reset the root position and velocity of the asset.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]
    orientations = root_states[:, 3:7]

    # velocities
    velocities = root_states[:, 7:13]

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

    # reset touchdown info
    env.extras['touchdown'] = []


def reset_landing_platform(
    env: RLTaskEnv,
    env_ids: torch.Tensor,
    landing_pltform_cfg: SceneEntityCfg = SceneEntityCfg("landing_platform"),
):
    """Reset the asset root state to the default position and velocity.

    This function reset the root position and velocity of the asset.
    """
    # extract the used quantities (to enable type-hinting)
    landing_pltform: RigidObject | Articulation = env.scene[landing_pltform_cfg.name]

    root_states = landing_pltform.data.default_root_state[env_ids].clone()

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]
    orientations = root_states[:, 3:7]

    landing_pltform.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)


def move_landing_platform(
    env: RLTaskEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    landing_platform_cfg: SceneEntityCfg = SceneEntityCfg("landing_platform"),
    base_lin_vel_threshold: float = 0,
    foot_z_threshold: float = 0.02,
    base_z_threshold: float = 0.35,
    base_heigth: float = 0.3

):
    """Reset the asset root state to the default position and velocity.

    This function reset the root position and velocity of the asset.
    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject | Articulation = env.scene[robot_cfg.name]
    landing_platform: RigidObject | Articulation = env.scene[landing_platform_cfg.name]

    # Get foot positions
    foot_idx = robot.find_bodies(".*foot")[0]
    fl_foot_z_pos_w = robot.data.body_state_w[:, foot_idx[0], 2]
    fr_foot_z_pos_w = robot.data.body_state_w[:, foot_idx[1], 2]
    rl_foot_z_pos_w = robot.data.body_state_w[:, foot_idx[2], 2]
    rr_foot_z_pos_w = robot.data.body_state_w[:, foot_idx[3], 2]

    # Get foots z position
    foots_z_pos_w = torch.cat((fl_foot_z_pos_w.unsqueeze(1),
                               fr_foot_z_pos_w.unsqueeze(1),
                               rl_foot_z_pos_w.unsqueeze(1),
                               rr_foot_z_pos_w.unsqueeze(1)), dim=1)

    # Get base z position
    base_pose_w = robot.data.root_state_w[:, 0:3]

    # Get base z linear velocity
    base_lin_vel_w = robot.data.root_state_w[env_ids, 7:10]

    root_states = landing_platform.data.default_root_state[env_ids].clone()
    # Get com_target command
    com_target = env.command_manager.get_command("com_target")

    # Obtain env ids of robot with foots heigher than the threshold
    foot_lifted_off_env_ids = torch.nonzero(torch.all(foots_z_pos_w > foot_z_threshold, dim=1)).reshape(1, -1)[0]
    # Obtain env ids of robot with base heigher than the threshold
    base_lifted_off_env_ids = torch.nonzero(base_pose_w[:, 2] > base_z_threshold).reshape(1, -1)[0]
    # Obtain env ids of robot that are experiencing a negative z linear velocity
    base_negative_lin_vel_env_ids = torch.nonzero(base_lin_vel_w[:, 2] < base_lin_vel_threshold).reshape(1, -1)[0]

    # Get the apex apex_env_ids from intersections of all conditions
    apex_env_ids = foot_lifted_off_env_ids[(foot_lifted_off_env_ids.view(1, -1) == base_lifted_off_env_ids.view(-1, 1)).any(dim=0)]
    apex_env_ids = apex_env_ids[(apex_env_ids.view(1, -1) == base_negative_lin_vel_env_ids.view(-1, 1)).any(dim=0)]

    # Calculate the landing platform default position and orientation
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]
    orientations = root_states[:, 3:7]

    # Move the landing platform of env_ids where apex is reached
    if len(apex_env_ids):
        # Transform landing platform position and orientation accordingly to the com_targert command
        positions[apex_env_ids] += com_target[apex_env_ids, 0:3]
        positions[apex_env_ids, 2] -= base_heigth
        orientations[apex_env_ids] = com_target[apex_env_ids, 3:7]

        # Write the changes to the simulator
        landing_platform.write_root_pose_to_sim(torch.cat([positions[apex_env_ids], orientations[apex_env_ids]], dim=-1), env_ids=apex_env_ids)


def touch_down(env: RLTaskEnv, env_ids: torch.Tensor, air_time_threshold: float, contact_threshold: float, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg):
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # asset.write_root_pose_to_sim(torch.tensor([0., 0., 1, 1, 0, 0, 0], device='cuda'), env_ids=torch.tensor([0], device='cuda').to(torch.int))
    # asset.write_root_velocity_to_sim(torch.tensor([0.], device='cuda'), env_ids=torch.tensor([0], device='cuda').to(torch.int))

    net_contact_forces = contact_sensor.data.net_forces_w
    net_last_air_time = contact_sensor.data.last_air_time

    flew_env_ids = torch.all(net_last_air_time[:, sensor_cfg.body_ids] > air_time_threshold, dim=1)
    in_contact_env_ids = torch.all(torch.norm(net_contact_forces[:, sensor_cfg.body_ids], dim=-1) > contact_threshold, dim=1)

    touchdown_env = torch.nonzero(flew_env_ids & in_contact_env_ids).reshape(1, -1)[0]

    existing_touchdown_ids = set(env.extras.get('touchdown', []))
    existing_touchdown_ids.update(touchdown_env.tolist())
    env.extras['touchdown'] = list(existing_touchdown_ids)

    if len(env.extras['touchdown']):
        touchdown_env_ids = torch.tensor(env.extras['touchdown'], device=env.device, dtype=torch.int)
        print(touchdown_env_ids, torch.zeros_like(touchdown_env_ids))
        asset.write_root_velocity_to_sim(torch.zeros((len(touchdown_env_ids), 6), device=env.device, dtype=torch.float), env_ids=touchdown_env_ids)
