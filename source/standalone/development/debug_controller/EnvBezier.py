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


class BezierCurveAction():

    def __init__(self,
                 t_th_min,
                 t_th_max,
                 x_theta_min,
                 x_theta_max,
                 x_r_min,
                 x_r_max,
                 xd_theta_min,
                 xd_theta_max,
                 xd_r_min,
                 xd_r_max,
                 psi_min,
                 psi_max,
                 theta_min,
                 theta_max,
                 phi_min,
                 phi_max,
                 psid_min,
                 psid_max,
                 thetad_min,
                 thetad_max,
                 phid_min,
                 phid_max) -> None:

        self.t_th_min = t_th_min
        self.t_th_max = t_th_max

        self.x_theta_min = x_theta_min
        self.x_theta_max = x_theta_max

        self.x_r_min = x_r_min
        self.x_r_max = x_r_max

        self.xd_theta_min = xd_theta_min
        self.xd_theta_max = xd_theta_max

        self.xd_r_min = xd_r_min
        self.xd_r_max = xd_r_max

        self.psi_min = psi_min
        self.psi_max = psi_max

        self.theta_min = theta_min
        self.theta_max = theta_max

        self.phi_min = phi_min
        self.phi_max = phi_max

        self.psid_min = psid_min
        self.psid_max = psid_max

        self.thetad_min = thetad_min
        self.thetad_max = thetad_max

        self.phid_min = phid_min
        self.phid_max = phid_max

    def torch_cart2sph(self, pos: torch.Tensor):
        # Extract x, y, z components
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]

        # Compute spherical coordinates
        hxy = torch.hypot(x, y)
        r = torch.hypot(hxy, z)
        el = torch.atan2(z, hxy)
        az = torch.atan2(y, x)

        # Concatenate azimuth, elevation, and radius
        spherical = torch.stack((az, el, r), dim=1)

        return spherical

    def torch_sph2cart(self, pos: torch.Tensor):
        # Extract az, el, r components
        az = pos[:, 0]
        el = pos[:, 1]
        r = pos[:, 2]

        rcos_theta = r * torch.cos(el)
        x = rcos_theta * torch.cos(az)
        y = rcos_theta * torch.sin(az)
        z = r * torch.sin(el)

        # Concatenate x, y, and z
        cartesian = torch.stack((x, y, z), dim=1)

        return cartesian

    def compute_bezier_w(self, start: torch.Tensor, start_v: torch.Tensor, end: torch.Tensor, end_v: torch.Tensor, t_th: torch.Tensor):

        w = torch.stack((
            start.unsqueeze(1),
            (((t_th / 3) * start_v) + start).unsqueeze(1),
            ((-(t_th / 3) * end_v) + end).unsqueeze(1),
            end.unsqueeze(1)), dim=2).squeeze()

        w = w.reshape(-1, 4, 3)

        w_d = torch.stack((
            ((3 / t_th) * (w[:, 1] - w[:, 0])).unsqueeze(1),
            ((3 / t_th) * (w[:, 2] - w[:, 1])).unsqueeze(1),
            ((3 / t_th) * (w[:, 3] - w[:, 2])).unsqueeze(1)), dim=2).squeeze()

        return w, w_d

    def bezier_trajectory(self, w: torch.Tensor, w_d: torch.Tensor, t_ex: float, t_th: torch.Tensor):

        t = t_ex / t_th

        bezier_curve_3 = torch.cat((
            (1.0) * (t**0) * (1 - t)**(3 - 0),
            (3.0) * (t**1) * (1 - t)**(3 - 1),
            (3.0) * (t**2) * (1 - t)**(3 - 2),
            (1.0) * (t**3) * (1 - t)**(3 - 3),
        ), dim=-1)

        bezier_curve_2 = torch.cat((
            (1.0) * (t**0) * (1 - t)**(2 - 0),
            (2.0) * (t**1) * (1 - t)**(2 - 1),
            (1.0) * (t**2) * (1 - t)**(2 - 2),
        ), dim=-1)

        bezier_position = torch.cat((
            (w[..., 0] * bezier_curve_3).sum(dim=-1).reshape(-1, 1),
            (w[..., 1] * bezier_curve_3).sum(dim=-1).reshape(-1, 1),
            (w[..., 2] * bezier_curve_3).sum(dim=-1).reshape(-1, 1)), dim=1)

        bezier_velocity = torch.cat((
            (w_d[..., 0] * bezier_curve_2).sum(dim=-1).reshape(-1, 1),
            (w_d[..., 1] * bezier_curve_2).sum(dim=-1).reshape(-1, 1),
            (w_d[..., 2] * bezier_curve_2).sum(dim=-1).reshape(-1, 1)), dim=1)

        return bezier_position, bezier_velocity

    def process_actions(self, trunk_x_0, trunk_xd_0, actions: torch.Tensor, target: torch.Tensor):

        trunk_tg = trunk_x_0 + target

        self.t_th = (self.t_th_max - self.t_th_min) * 0.5 * (actions[..., 0] + 1) + self.t_th_min
        self.t_th = self.t_th.reshape(-1, 1)

        # Phi is the same for x and xd
        x_xd_phi = self.torch_cart2sph(trunk_tg)[..., 0].clone()

        # Calculate X_lo
        x_theta = (self.x_theta_max - self.x_theta_min) * 0.5 * (actions[..., 1] + 1) + self.x_theta_min
        x_r = (self.x_r_max - self.x_r_min) * 0.5 * (actions[..., 2] + 1) + self.x_r_min

        trunk_x_lo = self.torch_sph2cart(torch.stack((x_xd_phi, x_theta, x_r), dim=1))

        # Calculate Xd_lo

        xd_theta = (self.xd_theta_max - self.xd_theta_min) * 0.5 * (actions[..., 3] + 1) + self.xd_theta_min
        xd_r = (self.xd_r_max - self.xd_r_min) * 0.5 * (actions[..., 4] + 1) + self.xd_r_min

        trunk_xd_lo = self.torch_sph2cart(torch.stack((x_xd_phi, xd_theta, xd_r), dim=1))

        self.w_x, self.w_xd = self.compute_bezier_w(trunk_x_0, trunk_xd_0, trunk_x_lo, trunk_xd_lo, self.t_th)

        print(trunk_x_0, trunk_xd_0, trunk_x_lo, trunk_xd_lo, self.t_th)

    def eval_bezier(self, dt):

        x, xd = self.bezier_trajectory(self.w_x, self.w_xd, dt, self.t_th)

        return x, xd


class EnvBezier():

    def __init__(self, simulation_app, sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
        super(EnvBezier, self).__init__()

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

        self.bezierAction = BezierCurveAction(t_th_min=0.1,
                                              t_th_max=1,
                                              x_theta_min=np.pi / 4,
                                              x_theta_max=np.pi / 2,
                                              x_r_min=0.2,
                                              x_r_max=0.4,
                                              xd_theta_min=np.pi / 6,
                                              xd_theta_max=np.pi / 2,
                                              xd_r_min=0.1,
                                              xd_r_max=5,
                                              psi_min=-2 * np.pi,
                                              psi_max=2 * np.pi,
                                              theta_min=- 2 * np.pi,
                                              theta_max=2 * np.pi,
                                              phi_min=- 2 * np.pi,
                                              phi_max=2 * np.pi,
                                              psid_min=-4,
                                              psid_max=4,
                                              thetad_min=-4,
                                              thetad_max=4,
                                              phid_min=-4,
                                              phid_max=4,)

    def ik(self, robot, pose, old_q_des):
        x = pose[..., 0:3]
        o_quat = pose[..., 3:7]

        q_des = torch.zeros_like(robot.data.default_joint_pos)

        root_pose_w = robot.data.root_state_w[:, 0:7].clone()

        fl_jacobian = robot.root_physx_view.get_jacobians()[:, self.fl_body_idx, :, np.array(self.fl_entity_cfg.joint_ids) + 6].clone()
        fl_pose_w = robot.data.body_state_w[:, self.fl_entity_cfg.body_ids[0], 0:7].clone()
        fl_joint_pos = robot.data.joint_pos[:, self.fl_entity_cfg.joint_ids].clone()
        fl_foot_pos_b, fl_foot_orient_b = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], fl_pose_w[:, 0:3], fl_pose_w[:, 3:7])
        fl_foot_pos_des_b, fl_foot_orient_des_b = subtract_frame_transforms(x, o_quat, fl_pose_w[:, 0:3])

        fr_jacobian = robot.root_physx_view.get_jacobians()[:, self.fr_body_idx, :, np.array(self.fr_entity_cfg.joint_ids) + 6].clone()
        fr_pose_w = robot.data.body_state_w[:, self.fr_entity_cfg.body_ids[0], 0:7].clone()
        fr_joint_pos = robot.data.joint_pos[:, self.fr_entity_cfg.joint_ids].clone()
        fr_foot_pos_b, fr_foot_orient_b = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], fr_pose_w[:, 0:3], fr_pose_w[:, 3:7])
        fr_foot_pos_des_b, fr_foot_orient_des_b = subtract_frame_transforms(x, o_quat, fr_pose_w[:, 0:3])

        rl_jacobian = robot.root_physx_view.get_jacobians()[:, self.rl_body_idx, :, np.array(self.rl_entity_cfg.joint_ids) + 6].clone()
        rl_pose_w = robot.data.body_state_w[:, self.rl_entity_cfg.body_ids[0], 0:7].clone()
        rl_joint_pos = robot.data.joint_pos[:, self.rl_entity_cfg.joint_ids].clone()
        rl_foot_pos_b, rl_foot_orient_b = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], rl_pose_w[:, 0:3], rl_pose_w[:, 3:7])
        rl_foot_pos_des_b, rl_foot_orient_des_b = subtract_frame_transforms(x, o_quat, rl_pose_w[:, 0:3])

        rr_jacobian = robot.root_physx_view.get_jacobians()[:, self.rr_body_idx, :, np.array(self.rr_entity_cfg.joint_ids) + 6].clone()
        rr_pose_w = robot.data.body_state_w[:, self.rr_entity_cfg.body_ids[0], 0:7].clone()
        rr_joint_pos = robot.data.joint_pos[:, self.rr_entity_cfg.joint_ids].clone()
        rr_foot_pos_b, rr_foot_orient_b = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], rr_pose_w[:, 0:3], rr_pose_w[:, 3:7])
        rr_foot_pos_des_b, rr_foot_orient_des_b = subtract_frame_transforms(x, o_quat, rr_pose_w[:, 0:3])

        # Orientation is just for visualization (ignore)
        self.fl_diff_ik_controller.set_command(fl_foot_pos_des_b, ee_quat=fl_foot_orient_des_b)
        self.fr_diff_ik_controller.set_command(fr_foot_pos_des_b, ee_quat=fr_foot_orient_des_b)
        self.rl_diff_ik_controller.set_command(rl_foot_pos_des_b, ee_quat=rl_foot_orient_des_b)
        self.rr_diff_ik_controller.set_command(rr_foot_pos_des_b, ee_quat=rr_foot_orient_des_b)

        q_des[:, self.fl_entity_cfg.joint_ids] = self.fl_diff_ik_controller.compute(fl_foot_pos_b, fl_foot_orient_b, fl_jacobian, fl_joint_pos)
        q_des[:, self.fr_entity_cfg.joint_ids] = self.fr_diff_ik_controller.compute(fr_foot_pos_b, fr_foot_orient_b, fr_jacobian, fr_joint_pos)
        q_des[:, self.rl_entity_cfg.joint_ids] = self.rl_diff_ik_controller.compute(rl_foot_pos_b, rl_foot_orient_b, rl_jacobian, rl_joint_pos)
        q_des[:, self.rr_entity_cfg.joint_ids] = self.rr_diff_ik_controller.compute(rr_foot_pos_b, rr_foot_orient_b, rr_jacobian, rr_joint_pos)

        qd_des = (q_des - old_q_des) / self.sim_dt

        return q_des, qd_des

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

    def plot_trunk_traj(self, actual_traj, des_traj, title=""):
        fig, ax = plt.subplots(3, 1, figsize=(10, 8))
        fig.suptitle(title)

        actual_traj = torch.stack(actual_traj, dim=1)[0]
        des_traj = torch.stack(des_traj, dim=1)[0]

        time = np.arange(0, len(des_traj)) * 0.005

        ax[0].plot(time, actual_traj[..., 0], color="blue", label="actual")
        ax[0].plot(time, des_traj[..., 0], color="red", label="desired")

        ax[1].plot(time, actual_traj[..., 1], color="blue")
        ax[1].plot(time, des_traj[..., 1], color="red")

        ax[2].plot(time, actual_traj[..., 2], color="blue")
        ax[2].plot(time, des_traj[..., 2], color="red")
        ax[2].axhline(0.15, color='purple')

        ax[0].legend()
        plt.show()

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

        tau_actual_traj = []
        tau_des_traj = []

        trunk_actual_traj = []
        trunk_des_traj = []

        sim_time = 0.0
        start_time = 1
        max_episode_time = 2 + start_time
        count = 0

        initial__trunk = robot.data.root_state_w[..., 0:7].clone()

        trunk_x_0 = initial__trunk[..., 0:3]
        trunk_xd_0 = robot.data.root_state_w[..., 7:10].clone()

        action = torch.tensor([[0.5, 0, 1, -0.2, -0.2]], device=self.sim.device)
        target = torch.tensor([[0.5, 0, 0]], device=self.sim.device)

        self.bezierAction.process_actions(trunk_x_0, trunk_xd_0, action, target)

        while True:
            # Simulate physics
            while self.simulation_app.is_running():

                if sim_time > max_episode_time:

                    print(robot.data.root_state_w[..., :7])

                    self.plot_traj(q_actual_traj, q_des_traj, "q")
                    self.plot_traj(qd_actual_traj, qd_des_traj, "qd")
                    self.plot_traj(tau_actual_traj, tau_des_traj, "tau")
                    self.plot_trunk_traj(trunk_actual_traj, trunk_des_traj, "trunk")

                    plt.show()

                    # Reset sim time
                    sim_time = 0.0

                    old_q_des = robot.data.default_joint_pos.clone()

                    # reset traj logger
                    q_actual_traj = []
                    q_des_traj = []

                    qd_actual_traj = []
                    qd_des_traj = []

                    tau_actual_traj = []
                    tau_des_traj = []

                    trunk_actual_traj = []
                    trunk_des_traj = []

                    # reset the robot
                    self.reset(robot)

                trunk_des = initial__trunk.clone()
                dt = 0

                if sim_time > start_time:
                    dt = sim_time - start_time

                x, xd = self.bezierAction.eval_bezier(dt)
                trunk_des[..., 0:3] = x

                if dt <= self.bezierAction.t_th:
                    q_des, qd_des = self.ik(robot, trunk_des, old_q_des)
                    old_q_des = q_des.clone()
                else:
                    q_des = robot.data.default_joint_pos.clone()
                    qd_des = robot.data.default_joint_vel.clone()

                robot.set_joint_position_target(q_des)
                robot.set_joint_velocity_target(qd_des)

                # write data to sim
                robot.write_data_to_sim()

                # perform step
                self.sim.step()
                # update sim-time
                sim_time += self.sim_dt
                count += 1
                # update buffers
                self.scene.update(self.sim_dt)

                if sim_time > start_time and dt <= self.bezierAction.t_th:

                    q_actual_traj.append(robot.data.joint_pos.clone().detach().cpu())
                    q_des_traj.append(q_des.detach().cpu())

                    qd_actual_traj.append(robot.data.joint_vel.clone().detach().cpu())
                    qd_des_traj.append(qd_des.detach().cpu())

                    tau_actual_traj.append(robot.data.applied_torque.clone().detach().cpu())
                    tau_des_traj.append(robot.data.computed_torque.clone().detach().cpu())

                    trunk_actual_traj.append(robot.data.root_state_w[..., 0:3].clone().detach().cpu())
                    trunk_des_traj.append(trunk_des[..., 0:3].detach().cpu())

            print("Simulation concluded :)")
            break
