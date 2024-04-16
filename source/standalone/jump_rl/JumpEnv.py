import numpy as np
import torch
import time

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import AssetBaseCfg, RigidObjectCfg
from omni.isaac.orbit.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.math import subtract_frame_transforms
from omni.isaac.orbit_assets.unitree import UNITREE_GO1_CFG
from omni.isaac.orbit.controllers import DifferentialIKController, DifferentialIKControllerCfg


class JumpEnv():

    def __init__(self, simulation_app, sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
        super(JumpEnv, self).__init__()

        self.simulation_app = simulation_app
        self.sim = sim
        self.scene = scene

        diff_ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")

        self.fl_diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
        self.fr_diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
        self.rl_diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
        self.rr_diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

        self.base_des = torch.tensor([-0.0003, 0.0017, 0.3, 1, 0, 0, 0], device=sim.device)
        # fl_foot_pos_b, fl_foot_orient_b = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], fl_pose_w[:, 0:3], fl_pose_w[:, 3:7])

        # [0,  0, 0.25,  1, 0, 0,  0]

        self.fl_entity_cfg = SceneEntityCfg("robot", joint_names=["FL.*"], body_names=["FL_foot"])
        self.fl_entity_cfg.resolve(self.scene)
        self.fl_jacobi_idx = self.fl_entity_cfg.body_ids[0]

        self.fr_entity_cfg = SceneEntityCfg("robot", joint_names=["FR.*"], body_names=["FR_foot"])
        self.fr_entity_cfg.resolve(self.scene)
        self.fr_jacobi_idx = self.fr_entity_cfg.body_ids[0]

        self.rl_entity_cfg = SceneEntityCfg("robot", joint_names=["RL.*"], body_names=["RL_foot"])
        self.rl_entity_cfg.resolve(self.scene)
        self.rl_jacobi_idx = self.rl_entity_cfg.body_ids[0]

        self.rr_entity_cfg = SceneEntityCfg("robot", joint_names=["RR.*"], body_names=["RR_foot"])
        self.rr_entity_cfg.resolve(self.scene)
        self.rr_jacobi_idx = self.rr_entity_cfg.body_ids[0]

        # Default configuration for having the robot at z = 0.3
        foot_orient_def = torch.tensor([[1, 0, 0, 0]], device=self.sim.device)

        self.fl_diff_ik_controller.set_command(torch.tensor([[0.1995, 0.1708, -0.2641]], device=self.sim.device), ee_quat=foot_orient_def)
        self.fr_diff_ik_controller.set_command(torch.tensor([[0.2009, -0.1701, -0.2643]], device=self.sim.device), ee_quat=foot_orient_def)
        self.rl_diff_ik_controller.set_command(torch.tensor([[-0.2355, 0.1642, -0.2678]], device=self.sim.device), ee_quat=foot_orient_def)
        self.rr_diff_ik_controller.set_command(torch.tensor([[-0.2368, -0.1640, -0.2680]], device=self.sim.device), ee_quat=foot_orient_def)

        # self.fl_diff_ik_controller.set_command(torch.tensor([[0.1995, 0.1708, -0.2641, 0.8993, 0.0727, -0.4298, -0.0348]], device=self.sim.device))
        # self.fr_diff_ik_controller.set_command(torch.tensor([[0.2009, -0.1701, -0.2643, 0.8983, -0.0715, -0.4321, 0.0344]], device=self.sim.device))
        # self.rl_diff_ik_controller.set_command(torch.tensor([[-0.2355, 0.1642, -0.2678, 0.9422, 0.0643, -0.3281, -0.0224]], device=self.sim.device))
        # self.rr_diff_ik_controller.set_command(torch.tensor([[-0.2368, -0.1640, -0.2680, 0.9431, -0.0639, -0.3256, 0.0221]], device=self.sim.device))
        # self.fl_diff_ik_controller.set_command(torch.tensor([[0.1995, 0.1708, -0.2641, 0.8993, 0.0727, -0.4298, -0.0348], [0.1995, 0.1708, -0.2641, 1, 0, 0, 0]], device=self.sim.device))
        # self.fr_diff_ik_controller.set_command(torch.tensor([[0.2009, -0.1701, -0.2643, 0.8983, -0.0715, -0.4321, 0.0344], [0.2009, -0.1701, -0.2643, 1, 0, 0, 0]], device=self.sim.device))
        # self.rl_diff_ik_controller.set_command(torch.tensor([[-0.2355, 0.1642, -0.2678, 0.9422, 0.0643, -0.3281, -0.0224], [-0.2355, 0.1642, -0.2678, 1, 0, 0, 0]], device=self.sim.device))
        # self.rr_diff_ik_controller.set_command(torch.tensor([[-0.2368, -0.1640, -0.2680, 0.9431, -0.0639, -0.3256, 0.0221], [-0.2368, -0.1640, -0.2680, 1, 0, 0, 0]], device=self.sim.device))

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
        max_episode_time = 5.0
        count = 0

        joint_pos_des = torch.zeros_like(robot.data.default_joint_pos)

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

                    self.fl_diff_ik_controller.reset()
                    self.fr_diff_ik_controller.reset()
                    self.rl_diff_ik_controller.reset()
                    self.rr_diff_ik_controller.reset()

                root_pose_w = robot.data.root_state_w[:, 0:7]

                start = time.time()
                fl_jacobian = robot.root_physx_view.get_jacobians()[:, self.fl_jacobi_idx, :, np.array(self.fl_entity_cfg.joint_ids) + 6]
                fl_pose_w = robot.data.body_state_w[:, self.fl_entity_cfg.body_ids[0], 0:7]
                fl_joint_pos = robot.data.joint_pos[:, self.fl_entity_cfg.joint_ids]
                fl_foot_pos_b, fl_foot_orient_b = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], fl_pose_w[:, 0:3], fl_pose_w[:, 3:7])
                joint_pos_des[:, self.fl_entity_cfg.joint_ids] = self.fl_diff_ik_controller.compute(fl_foot_pos_b, fl_foot_orient_b, fl_jacobian, fl_joint_pos)

                fr_jacobian = robot.root_physx_view.get_jacobians()[:, self.fr_jacobi_idx, :, np.array(self.fr_entity_cfg.joint_ids) + 6]
                fr_pose_w = robot.data.body_state_w[:, self.fr_entity_cfg.body_ids[0], 0:7]
                fr_joint_pos = robot.data.joint_pos[:, self.fr_entity_cfg.joint_ids]
                fr_foot_pos_b, fr_foot_orient_b = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], fr_pose_w[:, 0:3], fr_pose_w[:, 3:7])
                joint_pos_des[:, self.fr_entity_cfg.joint_ids] = self.fr_diff_ik_controller.compute(fr_foot_pos_b, fr_foot_orient_b, fr_jacobian, fr_joint_pos)

                rl_jacobian = robot.root_physx_view.get_jacobians()[:, self.rl_jacobi_idx, :, np.array(self.rl_entity_cfg.joint_ids) + 6]
                rl_pose_w = robot.data.body_state_w[:, self.rl_entity_cfg.body_ids[0], 0:7]
                rl_joint_pos = robot.data.joint_pos[:, self.rl_entity_cfg.joint_ids]
                rl_foot_pos_b, rl_foot_orient_b = v(root_pose_w[:, 0:3], root_pose_w[:, 3:7], rl_pose_w[:, 0:3], rl_pose_w[:, 3:7])
                joint_pos_des[:, self.rl_entity_cfg.joint_ids] = self.rl_diff_ik_controller.compute(rl_foot_pos_b, rl_foot_orient_b, rl_jacobian, rl_joint_pos)

                rr_jacobian = robot.root_physx_view.get_jacobians()[:, self.rr_jacobi_idx, :, np.array(self.rr_entity_cfg.joint_ids) + 6]
                rr_pose_w = robot.data.body_state_w[:, self.rr_entity_cfg.body_ids[0], 0:7]
                rr_joint_pos = robot.data.joint_pos[:, self.rr_entity_cfg.joint_ids]
                rr_foot_pos_b, rr_foot_orient_b = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], rr_pose_w[:, 0:3], rr_pose_w[:, 3:7])
                joint_pos_des[:, self.rr_entity_cfg.joint_ids] = self.rr_diff_ik_controller.compute(rr_foot_pos_b, rr_foot_orient_b, rr_jacobian, rr_joint_pos)

                print('-' * 10)
                print(fl_pose_w[:, 0:3], fl_pose_w[:, 3:7])
                print(fr_pose_w[:, 0:3], fr_pose_w[:, 3:7])
                print(rl_pose_w[:, 0:3], rl_pose_w[:, 3:7])
                print(rr_pose_w[:, 0:3], rr_pose_w[:, 3:7])

                robot.set_joint_position_target(joint_pos_des, joint_ids=robot_joint_ids)
                # print(root_pose_w, time.time() - start)

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
