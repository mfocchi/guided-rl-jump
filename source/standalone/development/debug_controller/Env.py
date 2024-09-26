import numpy as np
import torch
import time

import matplotlib.pyplot as plt

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import AssetBaseCfg, RigidObjectCfg
from omni.isaac.orbit.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.math import subtract_frame_transforms, quat_from_euler_xyz
from omni.isaac.orbit_assets.unitree import UNITREE_GO1_CFG
from omni.isaac.orbit.controllers import DifferentialIKController, DifferentialIKControllerCfg


class Env():

    def __init__(self, simulation_app, sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
        super(Env, self).__init__()

        self.simulation_app = simulation_app
        self.sim = sim
        self.scene = scene
        self.sim_dt = self.sim.get_physics_dt()
        print("Simulation time step:", self.sim_dt)

        diff_ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")

        self.fl_diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
        self.fr_diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
        self.rl_diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
        self.rr_diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

        self.fl_entity_cfg = SceneEntityCfg("robot", joint_names=["FL.*"], body_names=["FL_foot"])
        self.fl_entity_cfg.resolve(self.scene)
        self.fl_body_idx = self.fl_entity_cfg.body_ids[0]

        self.fr_entity_cfg = SceneEntityCfg("robot", joint_names=["FR.*"], body_names=["FR_foot"])
        self.fr_entity_cfg.resolve(self.scene)
        self.fr_body_idx = self.fr_entity_cfg.body_ids[0]

        self.rl_entity_cfg = SceneEntityCfg("robot", joint_names=["RL.*"], body_names=["RL_foot"])
        self.rl_entity_cfg.resolve(self.scene)
        self.rl_body_idx = self.rl_entity_cfg.body_ids[0]

        self.rr_entity_cfg = SceneEntityCfg("robot", joint_names=["RR.*"], body_names=["RR_foot"])
        self.rr_entity_cfg.resolve(self.scene)
        self.rr_body_idx = self.rr_entity_cfg.body_ids[0]

    def reset(self, robot):
        print("-" * 10)
        print("Resetting the base")

        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += self.scene.env_origins

        robot.reset()
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)

        self.fl_diff_ik_controller.reset()
        self.fr_diff_ik_controller.reset()
        self.rl_diff_ik_controller.reset()
        self.rr_diff_ik_controller.reset()

    def plot_traj(self, actual_traj, des_traj, title=""):
        fig, ax = plt.subplots(3, 4, figsize=(10, 8))
        fig.suptitle(title)

        actual_traj = torch.stack(actual_traj, dim=1)[0]
        des_traj = torch.stack(des_traj, dim=1)[0]

        time = np.arange(0, len(des_traj)) * 0.005

        # FL
        ax[0, 0].set_title("FL")
        ax[0, 0].plot(time, actual_traj[..., self.fl_entity_cfg.joint_ids[0]], color="blue", label="actual")
        ax[0, 0].plot(time, des_traj[..., self.fl_entity_cfg.joint_ids[0]], color="red", label="desired")

        ax[1, 0].plot(time, actual_traj[..., self.fl_entity_cfg.joint_ids[1]], color="blue")
        ax[1, 0].plot(time, des_traj[..., self.fl_entity_cfg.joint_ids[1]], color="red")

        ax[2, 0].plot(time, actual_traj[..., self.fl_entity_cfg.joint_ids[2]], color="blue")
        ax[2, 0].plot(time, des_traj[..., self.fl_entity_cfg.joint_ids[2]], color="red")

        ax[0, 0].legend()

        # FR
        ax[0, 1].set_title("FR")
        ax[0, 1].plot(time, actual_traj[..., self.fr_entity_cfg.joint_ids[0]], color="blue")
        ax[0, 1].plot(time, des_traj[..., self.fr_entity_cfg.joint_ids[0]], color="red")

        ax[1, 1].plot(time, actual_traj[..., self.fr_entity_cfg.joint_ids[1]], color="blue")
        ax[1, 1].plot(time, des_traj[..., self.fr_entity_cfg.joint_ids[1]], color="red")

        ax[2, 1].plot(time, actual_traj[..., self.fr_entity_cfg.joint_ids[2]], color="blue")
        ax[2, 1].plot(time, des_traj[..., self.fr_entity_cfg.joint_ids[2]], color="red")

        # RL
        ax[0, 2].set_title("RL")
        ax[0, 2].plot(time, actual_traj[..., self.rl_entity_cfg.joint_ids[0]], color="blue")
        ax[0, 2].plot(time, des_traj[..., self.rl_entity_cfg.joint_ids[0]], color="red")

        ax[1, 2].plot(time, actual_traj[..., self.rl_entity_cfg.joint_ids[1]], color="blue")
        ax[1, 2].plot(time, des_traj[..., self.rl_entity_cfg.joint_ids[1]], color="red")

        ax[2, 2].plot(time, actual_traj[..., self.rl_entity_cfg.joint_ids[2]], color="blue")
        ax[2, 2].plot(time, des_traj[..., self.rl_entity_cfg.joint_ids[2]], color="red")

        # RR
        ax[0, 3].set_title("RR")
        ax[0, 3].plot(time, actual_traj[..., self.rr_entity_cfg.joint_ids[0]], color="blue")
        ax[0, 3].plot(time, des_traj[..., self.rr_entity_cfg.joint_ids[0]], color="red")

        ax[1, 3].plot(time, actual_traj[..., self.rr_entity_cfg.joint_ids[1]], color="blue")
        ax[1, 3].plot(time, des_traj[..., self.rr_entity_cfg.joint_ids[1]], color="red")

        ax[2, 3].plot(time, actual_traj[..., self.rr_entity_cfg.joint_ids[2]], color="blue")
        ax[2, 3].plot(time, des_traj[..., self.rr_entity_cfg.joint_ids[2]], color="red")

    def run_simulator(self):
        """Runs the simulation loop."""
        robot = self.scene["robot"]

        self.reset(robot)

        # Define simulation stepping

        old_q_des = robot.data.default_joint_pos.clone()

        q_actual_traj = []
        q_des_traj = []

        qd_actual_traj = []
        qd_des_traj = []

        sim_time = 0.0
        max_episode_time = 2
        count = 0

        while True:
            # Simulate physics
            while self.simulation_app.is_running():

                if sim_time > max_episode_time:

                    self.plot_traj(q_actual_traj, q_des_traj, "q")
                    self.plot_traj(qd_actual_traj, qd_des_traj, "qd")

                    plt.show()

                    # Reset sim time
                    sim_time = 0.0

                    old_q_des = robot.data.default_joint_pos.clone()

                    # reset traj logger
                    q_actual_traj = []
                    q_des_traj = []

                    qd_actual_traj = []
                    qd_des_traj = []

                    # reset the robot
                    self.reset(robot)

                # set the robot trunk in air
                trunk_air_state = robot.data.default_root_state.clone()
                trunk_air_state[..., 2] += 0.5

                robot.write_root_pose_to_sim(trunk_air_state[..., 0:7])
                robot.write_root_velocity_to_sim(trunk_air_state[..., 7:14])

                # q_des = robot.data.default_joint_pos.clone()
                # q_des = q_des.clone() + 0.1 * torch.sin(torch.tensor(3 * 2 * np.pi * sim_time))

                # qd_des = (q_des.clone() - old_q_des) / self.sim_dt
                # old_q_des = q_des.clone()
                q_des = torch.zeros_like(robot.data.default_joint_pos)
                qd_des = robot.data.default_joint_vel

                # write data to sim
                # robot.write(q_des)
                # robot.set_joint_velocity_target(qd_des)
                robot.write_joint_state_to_sim(q_des, qd_des)

                robot.write_data_to_sim()

                # perform step
                self.sim.step()
                # update sim-time
                sim_time += self.sim_dt
                count += 1
                # update buffers
                self.scene.update(self.sim_dt)

                q_actual_traj.append(robot.data.joint_pos.clone().detach().cpu())
                q_des_traj.append(q_des.detach().cpu())

                qd_actual_traj.append(robot.data.joint_vel.clone().detach().cpu())
                qd_des_traj.append(qd_des.detach().cpu())

            print("Simulation concluded :)")
            break
