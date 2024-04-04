# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates different legged robots.

.. code-block:: bash

    # Usage
    ./orbit.sh -p source/standalone/demos/quadrupeds.py

"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates different legged robots.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.math import subtract_frame_transforms
##
# Pre-defined configs
##
from omni.isaac.orbit_assets.unitree import UNITREE_GO1_CFG

torch.set_printoptions(threshold=float('inf'), precision=5, linewidth=10000, sci_mode=False)


@configclass
class SceneCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    robot = scene["robot"]

    foot_idx = robot.find_bodies(".*foot")[0]
    robot_joint_ids = robot.find_joints(".*joint")[0]

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    fl_joints = np.array(robot.find_joints("FL.*")[0])
    fr_joints = np.array(robot.find_joints("FR.*")[0])
    rl_joints = np.array(robot.find_joints("RL.*")[0])
    rr_joints = np.array(robot.find_joints("RR.*")[0])

    # Simulate physics
    while simulation_app.is_running():

        # 'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'
        fl_jacobian = robot.root_physx_view.get_jacobians()[:, foot_idx[0], 0:3, fl_joints + 6]
        fr_jacobian = robot.root_physx_view.get_jacobians()[:, foot_idx[1], 0:3, fr_joints + 6]
        rl_jacobian = robot.root_physx_view.get_jacobians()[:, foot_idx[2], 0:3, rl_joints + 6]
        rr_jacobian = robot.root_physx_view.get_jacobians()[:, foot_idx[3], 0:3, rr_joints + 6]

        # foot position in wf
        fl_foot_pos_w = robot.data.body_state_w[:, foot_idx[0], 0:3]
        fr_foot_pos_w = robot.data.body_state_w[:, foot_idx[1], 0:3]
        rl_foot_pos_w = robot.data.body_state_w[:, foot_idx[2], 0:3]
        rr_foot_pos_w = robot.data.body_state_w[:, foot_idx[3], 0:3]

        # foot orientation in wf
        fl_foot_orient_w = robot.data.body_state_w[:, foot_idx[0], 3:7]
        fr_foot_orient_w = robot.data.body_state_w[:, foot_idx[1], 3:7]
        rl_foot_orient_w = robot.data.body_state_w[:, foot_idx[2], 3:7]
        rr_foot_orient_w = robot.data.body_state_w[:, foot_idx[3], 3:7]

        base_pose_w = robot.data.root_state_w[:, 0:3]
        base_orient_w = robot.data.root_state_w[:, 3:7]

        base_lin_vel_w = robot.data.root_state_w[:, 7:10]
        base_ang_vel_w = robot.data.root_state_w[:, 10:13]

        # foot position, orientation in bf
        fl_foot_pos_b, fl_foot_orient_b = subtract_frame_transforms(base_pose_w, base_orient_w, fl_foot_pos_w, fl_foot_orient_w)
        fr_foot_pos_b, fr_foot_orient_b = subtract_frame_transforms(base_pose_w, base_orient_w, fr_foot_pos_w, fr_foot_orient_w)
        rl_foot_pos_b, rl_foot_orient_b = subtract_frame_transforms(base_pose_w, base_orient_w, rl_foot_pos_w, rl_foot_orient_w)
        rr_foot_pos_b, rr_foot_orient_b = subtract_frame_transforms(base_pose_w, base_orient_w, rr_foot_pos_w, rr_foot_orient_w)

        # foot joint position
        fl_joint_pos = robot.data.joint_pos[:, fl_joints]
        fr_joint_pos = robot.data.joint_pos[:, fr_joints]
        rl_joint_pos = robot.data.joint_pos[:, rl_joints]
        rr_joint_pos = robot.data.joint_pos[:, rr_joints]

        # foot joint velocity
        fl_joint_vel = robot.data.joint_vel[:, fl_joints]
        fr_joint_vel = robot.data.joint_vel[:, fr_joints]
        rl_joint_vel = robot.data.joint_vel[:, rl_joints]
        rr_joint_vel = robot.data.joint_vel[:, rr_joints]

        joint_pos_des = robot.data.default_joint_pos
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_joint_ids)

        # write data to sim
        robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)


def main():
    """Main function."""

    # Initialize the simulation context
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, substeps=1))
    # Set main camera
    sim.set_camera_view(eye=(2.5, 2.5, 2.5), target=(0.0, 0.0, 0.0))
    # design scene
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
