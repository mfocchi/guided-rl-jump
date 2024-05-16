from __future__ import annotations

"""Launch Isaac Sim Simulator first."""
import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates different legged robots.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--env_spacing", type=int, default=7, help="Spacing between each origin")
parser.add_argument("--debug", type=bool, default=False, help="Enable debugging stuff")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Jump task"""
import numpy as np
import torch

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.scene import InteractiveScene
from omni.isaac.orbit_assets.unitree import UNITREE_GO1_CFG

from SceneCfg import SceneCfg
from Env import *


def main():

    if args_cli.debug:
        torch.set_printoptions(threshold=float('inf'), precision=5, linewidth=10000, sci_mode=False)

    # Initialize the simulation context
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, substeps=1))
    sim.set_camera_view(eye=(2.5, 2.5, 2.5), target=(0.0, 0.0, 0.0))

    # Design the scene
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing)
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()
    print("[INFO]: Setup complete...")

    env = Env(simulation_app, sim, scene)

    # Run the simulator
    env.run_simulator()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
