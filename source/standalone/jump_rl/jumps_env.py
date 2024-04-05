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
parser.add_argument("--env_spacing", type=int, default=7, help="Number of environments to spawn.")
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
from omni.isaac.orbit.assets import AssetBaseCfg, RigidObjectCfg
from omni.isaac.orbit.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.math import subtract_frame_transforms
##
# Pre-defined configs
##
from omni.isaac.orbit_assets.unitree import UNITREE_GO1_CFG

from JumpEnv import *

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

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 0.0)),
    )


def main():
    """Main function."""

    # Initialize the simulation context
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, substeps=1))
    # Set main camera
    sim.set_camera_view(eye=(2.5, 2.5, 2.5), target=(0.0, 0.0, 0.0))
    # design scene
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing)
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    env = JumpEnv(simulation_app, sim, scene)
    # Run the simulator
    env.run_simulator()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
