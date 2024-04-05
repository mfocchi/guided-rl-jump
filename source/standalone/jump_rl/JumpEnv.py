import numpy as np
import torch

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import AssetBaseCfg, RigidObjectCfg
from omni.isaac.orbit.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.math import subtract_frame_transforms
from omni.isaac.orbit_assets.unitree import UNITREE_GO1_CFG

class JumpEnv():

    def __init__(self, simulation_app, sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
        super(JumpEnv, self).__init__()

        self.simulation_app = simulation_app
        self.sim = sim
        self.scene = scene

    def compute_quantities(self, robot):
        fl_joints = np.array(robot.find_joints("FL.*")[0])
        fr_joints = np.array(robot.find_joints("FR.*")[0])
        rl_joints = np.array(robot.find_joints("RL.*")[0])
        rr_joints = np.array(robot.find_joints("RR.*")[0])

        foot_idx = robot.find_bodies(".*foot")[0]

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

    def run_simulator(self):
        """Runs the simulation loop."""
        robot = self.scene["robot"]

        robot_joint_ids = robot.find_joints(".*joint")[0]

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.set_joint_position_target(joint_pos, joint_ids=robot_joint_ids)

        robot.reset()

        # Define simulation stepping
        sim_dt = self.sim.get_physics_dt()
        sim_time = 0.0
        max_episode_time = 2.0
        count = 0

        while True:
            # Simulate physics
            while self.simulation_app.is_running():

                if sim_time > max_episode_time:
                    sim_time = 0.0
                    print("-" * 10)
                    print("Resetting the base")

                    root_state = robot.data.default_root_state.clone()
                    root_state[:, :3] += self.scene.env_origins

                    robot.write_root_state_to_sim(root_state)
                    robot.reset()

                self.compute_quantities(robot)

                joint_pos_des = robot.data.default_joint_pos
                robot.set_joint_position_target(joint_pos_des, joint_ids=robot_joint_ids)

                # write data to sim
                robot.write_data_to_sim()
                # perform step
                self.sim.step()
                # update sim-time
                sim_time += sim_dt
                count += 1
                # update buffers
                self.scene.update(sim_dt)

            print("Simulation concluded :)")
            break
