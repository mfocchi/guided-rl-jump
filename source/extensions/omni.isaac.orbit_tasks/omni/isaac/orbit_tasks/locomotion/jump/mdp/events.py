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
    initial_z: float = 0.0,
):
    """Reset the asset root state to the default position and velocity.

    This function reset the root position and velocity of the asset.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    asset.reset()
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()
    # move initial z according to params
    root_states[..., 2] += initial_z
    root_states[..., 0:3] += env.scene.env_origins[env_ids]

    # set into the physics simulation
    asset.write_root_pose_to_sim(root_states[..., 0:7])
    asset.write_root_velocity_to_sim(root_states[..., 7:])
    asset.write_joint_state_to_sim(asset.data.default_joint_pos.clone(), asset.data.default_joint_vel.clone())
    asset.set_joint_position_target(asset.data.default_joint_pos.clone())
    asset.set_joint_velocity_target(asset.data.default_joint_vel.clone())
    asset.set_joint_effort_target(torch.zeros_like(asset.data.default_joint_pos))

    # reset apex, touchdown info
    env.extras['apex'] = {}
    env.extras['touchdown'] = {}
    env.extras['after_t_th_total'] = torch.tensor([], device=env.device)
    env.extras['actual_lo_config'] = torch.zeros_like(asset.data.root_state_w)
    env.extras['t_th_q'] = torch.zeros_like(asset.data.joint_pos)
    env.extras['apex_q'] = torch.zeros_like(asset.data.joint_pos)
    env.extras['apex_dt'] = torch.zeros(env.num_envs, device=env.device)
    env.extras['apex_z'] = torch.zeros(env.num_envs, device=env.device)
    env.extras['landing_z'] = torch.zeros(env.num_envs, device=env.device)
    env.extras['wbc'] = torch.zeros((env.num_envs, 1), device=env.device)
    env.extras['fail'] = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    env.extras['forces'] = torch.zeros((env.num_envs, 4, 3), device=env.device)


def reset_landing_platform(
    env: RLTaskEnv,
    env_ids: torch.Tensor,
    landing_pltform_cfg: SceneEntityCfg = SceneEntityCfg("landing_platform"),
    initial_z: float = 0.0,

):
    """Reset the asset root state to the default position and velocity.

    This function reset the root position and velocity of the asset.
    """
    # extract the used quantities (to enable type-hinting)
    landing_pltform: RigidObject = env.scene[landing_pltform_cfg.name]
    landing_pltform.reset()

    root_states = landing_pltform.data.default_root_state[env_ids].clone()
    # move initial z according to params
    root_states[..., 2] += initial_z

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]
    orientations = root_states[:, 3:7]

    landing_pltform.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)


def detect_apex(
    env: RLTaskEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    landing_platform_cfg: SceneEntityCfg = SceneEntityCfg("landing_platform"),
    base_lin_vel_threshold: float = -0.2,
    foot_height_offset: float = 0.02,
    offset: float = 0.05,
    initial_z: float = 0.0,
    foot_name: str = ".*foot"

):
    """Reset the asset root state to the default position and velocity.

    This function reset the root position and velocity of the asset.
    """
    # extract the used quantities (to enable type-hinting)
    robot: Articulation = env.scene[robot_cfg.name]
    landing_platform: RigidObject | Articulation = env.scene[landing_platform_cfg.name]

    # Get foot positions
    foot_idx = robot.find_bodies(foot_name)[0]
    fl_foot_z_pos_w = robot.data.body_state_w[:, foot_idx[0], 2].clone()
    fr_foot_z_pos_w = robot.data.body_state_w[:, foot_idx[1], 2].clone()
    rl_foot_z_pos_w = robot.data.body_state_w[:, foot_idx[2], 2].clone()
    rr_foot_z_pos_w = robot.data.body_state_w[:, foot_idx[3], 2].clone()

    # Get foots z position
    foots_z_pos_w = torch.cat((fl_foot_z_pos_w.unsqueeze(1),
                               fr_foot_z_pos_w.unsqueeze(1),
                               rl_foot_z_pos_w.unsqueeze(1),
                               rr_foot_z_pos_w.unsqueeze(1)), dim=1)

    foots_z_pos_w -= foot_height_offset

    # Get base z linear velocity
    base_lin_vel_w = robot.data.root_state_w[env_ids, 7:10].clone()

    root_states = landing_platform.data.default_root_state[env_ids].clone()
    # Get trunk_target command
    trunk_target = env.command_manager.get_command("trunk_target")

    # Obtain env ids of robot that are experiencing a negative z linear velocity
    base_negative_lin_vel_env_ids = torch.nonzero(base_lin_vel_w[:, 2] < base_lin_vel_threshold).reshape(1, -1)[0]

    after_t_th_total_ids = env.extras.get('after_t_th_total')

    # Get the apex apex_env_ids from intersections of all conditions
    apex_env_ids = base_negative_lin_vel_env_ids[(base_negative_lin_vel_env_ids.view(1, -1) == after_t_th_total_ids.view(-1, 1)).any(dim=0)]

    foot_target_z = trunk_target[..., 2] + initial_z

    foot_height_z = torch.min(foots_z_pos_w, dim=1).values
    landing_z = torch.clip(foot_height_z - offset, min=torch.zeros_like(foot_target_z), max=foot_target_z)

    # Calculate the landing platform default position and orientation
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]
    orientations = trunk_target[env_ids, 3:7]

    positions[..., 0:2] += trunk_target[..., 0:2]
    # correct landing platform offset
    positions[..., 2] = env.extras['landing_z'] - 0.025

    existing_apex_ids = env.extras.get('apex', {})

    for apex_env in apex_env_ids:
        apex_env = int(apex_env.item())
        if apex_env not in existing_apex_ids:
            # adding the touchdown state
            env.extras['apex'][apex_env] = True
            # env.extras['apex_q'][apex_env] = robot.data.joint_pos[apex_env].clone()
            env.extras['apex_q'][apex_env] = env.extras['q_des'][apex_env].clone()
            env.extras['apex_z'][apex_env] = robot.data.root_state_w[apex_env][..., 2].clone()

            # get if the reached foot z is greather than the saved ona but lower/equal than the target one
            if (landing_z[apex_env] > env.extras['landing_z'][apex_env]) and (landing_z[apex_env] <= foot_target_z[apex_env]):
                env.extras['landing_z'][apex_env] = landing_z[apex_env]

    existing_apex_ids = torch.tensor(list(env.extras['apex'].keys()), device=env.device, dtype=torch.int)
    landing_platform.write_root_pose_to_sim(torch.cat([positions[existing_apex_ids], orientations[existing_apex_ids]], dim=-1), env_ids=existing_apex_ids)

    # print('apex', torch.tensor(list(env.extras['apex'].keys()), device=env.device, dtype=torch.int))


def detect_touchdown(env: RLTaskEnv, env_ids: torch.Tensor, contact_threshold: float, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg):
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    net_contact_forces = contact_sensor.data.net_forces_w.clone()

    # near_foot_env_ids = torch.std(asset.data.body_state_w[:, foot_idx, 2], dim=1) <= foot_pos_threshold
    in_contact_env_ids = torch.all(torch.norm(net_contact_forces[:, sensor_cfg.body_ids], dim=-1) > contact_threshold, dim=1)

    # touchdown_env_ids = torch.nonzero(near_foot_env_ids & in_contact_env_ids).reshape(1, -1)[0]
    touchdown_env_ids = torch.nonzero(in_contact_env_ids).reshape(1, -1)[0]

    apex_env_ids = torch.tensor(list(env.extras['apex'].keys()), device=env.device, dtype=torch.int)

    # Save touchdown pose
    existing_touchdown_ids = env.extras.get('touchdown', {})
    for touchdown_env in touchdown_env_ids:
        touchdown_env = touchdown_env.item()
        # Check if the env is in the apex and not have done touchdown
        if touchdown_env not in existing_touchdown_ids and touchdown_env in apex_env_ids:
            root_state = asset.data.root_state_w[touchdown_env].clone()
            joint_pos = asset.data.joint_pos[touchdown_env].clone()
            # adding the touchdown state
            env.extras['touchdown'][touchdown_env] = torch.cat((root_state, joint_pos), dim=0)

    # print(f"detected touchdow: {env.extras['touchdown'].keys()}")
    # try to pause the simulation for the env that are in touchdown
    # enable wbc if touchdown is detected
    if len(env.extras['touchdown']):
        existing_touchdown_ids = torch.tensor(list(env.extras['touchdown'].keys()), device=env.device, dtype=torch.int)
        env.extras['wbc'][existing_touchdown_ids] = 1
    #     values = torch.stack(list(env.extras['touchdown'].values())).to(env.device)
    #     asset.write_root_pose_to_sim(values[..., 0:7], env_ids=existing_touchdown_ids)
    #     asset.write_root_velocity_to_sim(torch.zeros((len(existing_touchdown_ids), 6), device=env.device, dtype=torch.float), env_ids=existing_touchdown_ids)
    #     asset.write_joint_state_to_sim(values[..., 13:25], torch.zeros((len(existing_touchdown_ids), 12), device=env.device, dtype=torch.float), env_ids=existing_touchdown_ids)


def detect_fail(env: RLTaskEnv, env_ids: torch.Tensor, contact_threshold: float, sensor_cfg: SceneEntityCfg):
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    net_contact_forces = contact_sensor.data.net_forces_w.clone()

    # near_foot_env_ids = torch.std(asset.data.body_state_w[:, foot_idx, 2], dim=1) <= foot_pos_threshold
    in_contact_env_ids = torch.any(torch.norm(net_contact_forces[:, sensor_cfg.body_ids], dim=-1) > contact_threshold, dim=1)
    env.extras['fail'] = env.extras['fail'] | in_contact_env_ids
