import numpy as np
import torch
import time

import matplotlib.pyplot as plt

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import AssetBaseCfg, RigidObjectCfg
from omni.isaac.orbit.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat
from omni.isaac.orbit_assets.unitree import UNITREE_GO1_CFG
from omni.isaac.orbit.controllers import DifferentialIKController, DifferentialIKControllerCfg


class BezierCurveAction():

    def __init__(self,
                 min_action,
                 max_action,
                 lerp_time,
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

        self.min_action = min_action
        self.max_action = max_action

        self.lerp_time = lerp_time

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

    def torch_cart2sph(self, pos: torch.Tensor, threshold: float = 1e-5):
        # Extract x, y, z components
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]

        # deal with precision problem
        x = torch.where(torch.abs(x) < threshold, torch.tensor(0.0, dtype=x.dtype, device=x.device), x)
        y = torch.where(torch.abs(y) < threshold, torch.tensor(0.0, dtype=y.dtype, device=y.device), y)

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

    def map_range(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def process_actions(self, robot, trunk_x_0, trunk_xd_0, trunk_o_0, trunk_od_0, actions: torch.Tensor, target: torch.Tensor):

        # Reset qd computation
        self.old_q_des = robot.data.default_joint_pos.clone()

        trunk_tg = trunk_x_0 + target

        self.t_th = self.map_range(actions[..., 0], self.min_action, self.max_action, self.t_th_min, self.t_th_max)
        self.t_th = self.t_th.reshape(-1, 1)

        # Phi is the same for x and xd
        x_xd_phi = self.torch_cart2sph(trunk_tg)[..., 0].clone()

        # Calculate X_lo
        x_theta = self.map_range(actions[..., 1], self.min_action, self.max_action, self.x_theta_min, self.x_theta_max)
        x_r = self.map_range(actions[..., 2], self.min_action, self.max_action, self.x_r_min, self.x_r_max)

        self.trunk_x_lo = self.torch_sph2cart(torch.stack((x_xd_phi, x_theta, x_r), dim=1))

        # Calculate Xd_lo

        xd_theta = self.map_range(actions[..., 3], self.min_action, self.max_action, self.xd_theta_min, self.xd_theta_max)
        xd_r = self.map_range(actions[..., 4], self.min_action, self.max_action, self.xd_r_min, self.xd_r_max)

        self.trunk_xd_lo = self.torch_sph2cart(torch.stack((x_xd_phi, xd_theta, xd_r), dim=1))

        # Calculate Phi_lo

        psi = self.map_range(actions[..., 5], self.min_action, self.max_action, self.psi_min, self.psi_max)
        theta = self.map_range(actions[..., 6], self.min_action, self.max_action, self.theta_min, self.theta_max)
        phi = self.map_range(actions[..., 7], self.min_action, self.max_action, self.phi_min, self.phi_max)

        self.trunk_o_lo = torch.stack((psi, theta, phi), dim=1)

        # Calculate Phid_lo

        psid = self.map_range(actions[..., 8], self.min_action, self.max_action, self.psid_min, self.psid_max)
        thetad = self.map_range(actions[..., 9], self.min_action, self.max_action, self.thetad_min, self.thetad_max)
        phid = self.map_range(actions[..., 10], self.min_action, self.max_action, self.phid_min, self.phid_max)

        self.trunk_od_lo = torch.stack((psid, thetad, phid), dim=1)

        # TODO: implement scaling
        xd_mult = actions[..., 11]
        l_expl = actions[..., 12]

        # NOTE: debug override
        self.trunk_x_lo = torch.tensor([[0., 0., 0.2]], device="cuda")
        self.trunk_xd_lo = torch.tensor([[0, 0, 1.]], device="cuda")
        self.trunk_o_lo = torch.tensor([[0, -0.1, 0]], device="cuda")
        self.trunk_od_lo = torch.tensor([[0, 0., 0]], device="cuda")
        self.t_th = torch.tensor([[0.5]], device="cuda")

        self.trunk_xd_exp = self.trunk_xd_lo * xd_mult
        trunk_xd_0_un = self.trunk_xd_lo / torch.norm(self.trunk_xd_lo)

        self.trunk_x_exp = self.trunk_x_lo + trunk_xd_0_un * l_expl

        vf_n = torch.norm(self.trunk_xd_exp)
        v0_n = torch.norm(self.trunk_xd_lo)
        sf_n = torch.norm(self.trunk_x_exp)
        s0_n = torch.norm(self.trunk_x_lo)

        a = 0.5 * ((torch.pow(vf_n, 2) - torch.pow(v0_n, 2)) / (sf_n - s0_n))

        self.t_exp = (vf_n - v0_n) / a
        self.t_th_total = self.t_th + self.t_exp

        print(f"t_th: {self.t_th}, x_lo: {self.trunk_x_lo}, xd_lo: {self.trunk_xd_lo}\n\
              o_lo: {self.trunk_o_lo}, od_lo: {self.trunk_od_lo}, xd_mult: {xd_mult}, l_exp:{l_expl}\n\
                x_exp: {self.trunk_x_exp}, xd_exp: {self.trunk_xd_exp}, t_exp: {self.t_exp}")

        self.w_x, self.w_xd = self.compute_bezier_w(trunk_x_0, trunk_xd_0, self.trunk_x_lo, self.trunk_xd_lo, self.t_th)
        self.w_o, self.w_od = self.compute_bezier_w(trunk_o_0, trunk_od_0, self.trunk_o_lo, self.trunk_od_lo, self.t_th_total)

    def eval_bezier(self, dt):
        if dt < self.t_th:
            x, _ = self.bezier_trajectory(self.w_x, self.w_xd, dt, self.t_th)
        else:
            t = dt - self.t_th
            x = torch.lerp(self.trunk_x_lo, self.trunk_x_exp, t / self.t_exp)
        o, _ = self.bezier_trajectory(self.w_o, self.w_od, dt, self.t_th_total)

        return x, o


class EnvBezier():

    def __init__(self, simulation_app, sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
        super(EnvBezier, self).__init__()

        self.simulation_app = simulation_app
        self.sim = sim
        self.scene = scene
        self.sim_dt = self.sim.get_physics_dt()
        print("Simulation time step:", self.sim_dt)

        self.q_0_lo = torch.tensor([0.2187, -0.2191, 0.2343, -0.2364, 1.3717, 1.3716, 1.6770, 1.6784, -2.4063, -2.4061, -2.2808, -2.2778], device=sim.device)

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

        self.bezierAction = BezierCurveAction(
            min_action=-5,
            max_action=5,
            lerp_time=0.1,
            t_th_min=0.3,
            t_th_max=1,
            x_theta_min=np.pi / 4,
            x_theta_max=np.pi / 2,
            x_r_min=0.2,
            x_r_max=0.4,
            xd_theta_min=np.pi / 6,
            xd_theta_max=np.pi / 2,
            xd_r_min=0.1,
            xd_r_max=5,
            psi_min=-np.pi / 4,
            psi_max=np.pi / 4,
            theta_min=-np.pi / 4,
            theta_max=np.pi / 4,
            phi_min=-np.pi,
            phi_max=np.pi,
            psid_min=-4,
            psid_max=4,
            thetad_min=-4,
            thetad_max=4,
            phid_min=-4,
            phid_max=4)

    def ik(self, robot, x, o, old_q_des):
        x += self.scene.env_origins
        o_quat = quat_from_euler_xyz(o[..., 0], o[..., 1], o[..., 2])

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

    def plot_trunk_traj(self, actual_traj, des_traj, t_th, title=""):
        fig, ax = plt.subplots(3, 1, figsize=(10, 8))
        fig.suptitle(title)

        actual_traj = torch.stack(actual_traj, dim=1)[0]
        des_traj = torch.stack(des_traj, dim=1)[0]

        time = np.arange(0, len(des_traj)) * 0.005
        time_act = np.arange(0, len(actual_traj)) * 0.005
        z_min = torch.argmin(des_traj[..., 2])
        t_zmin = time[z_min]

        ax[0].plot(time_act, actual_traj[..., 0], color="blue", label="actual")
        ax[0].plot(time, des_traj[..., 0], color="red", label="desired")

        ax[1].plot(time_act, actual_traj[..., 1], color="blue")
        ax[1].plot(time, des_traj[..., 1], color="red")

        ax[2].plot(time_act, actual_traj[..., 2], color="blue")
        ax[2].plot(time, des_traj[..., 2], color="red")
        ax[2].axhline(0.15, color='purple')
        ax[2].axvline(t_zmin, color='gray')

        ax[0].axvline(t_th, color='orange')
        ax[1].axvline(t_th, color='orange')
        ax[2].axvline(t_th, color='orange')

        ax[0].legend()

        return t_zmin

    def plot_traj(self, actual_traj, des_traj, t_th, title=""):
        fig, ax = plt.subplots(3, 4, figsize=(10, 8))
        fig.suptitle(title)

        actual_traj = torch.stack(actual_traj, dim=1)[0]
        des_traj = torch.stack(des_traj, dim=1)[0]

        time = np.arange(0, len(des_traj)) * 0.005

        # FL
        ax[0, 0].set_title("FL")
        ax[0, 0].plot(time, actual_traj[..., self.fl_entity_cfg.joint_ids[0]], color="blue", label="actual")
        ax[0, 0].plot(time, des_traj[..., self.fl_entity_cfg.joint_ids[0]], color="red", label="desired")
        ax[0, 0].axvline(t_th, color='orange')

        ax[1, 0].plot(time, actual_traj[..., self.fl_entity_cfg.joint_ids[1]], color="blue")
        ax[1, 0].plot(time, des_traj[..., self.fl_entity_cfg.joint_ids[1]], color="red")
        ax[1, 0].axvline(t_th, color='orange')

        ax[2, 0].plot(time, actual_traj[..., self.fl_entity_cfg.joint_ids[2]], color="blue")
        ax[2, 0].plot(time, des_traj[..., self.fl_entity_cfg.joint_ids[2]], color="red")
        ax[2, 0].axvline(t_th, color='orange')

        ax[0, 0].legend()

        # FR
        ax[0, 1].set_title("FR")
        ax[0, 1].plot(time, actual_traj[..., self.fr_entity_cfg.joint_ids[0]], color="blue")
        ax[0, 1].plot(time, des_traj[..., self.fr_entity_cfg.joint_ids[0]], color="red")
        ax[0, 1].axvline(t_th, color='orange')

        ax[1, 1].plot(time, actual_traj[..., self.fr_entity_cfg.joint_ids[1]], color="blue")
        ax[1, 1].plot(time, des_traj[..., self.fr_entity_cfg.joint_ids[1]], color="red")
        ax[1, 1].axvline(t_th, color='orange')

        ax[2, 1].plot(time, actual_traj[..., self.fr_entity_cfg.joint_ids[2]], color="blue")
        ax[2, 1].plot(time, des_traj[..., self.fr_entity_cfg.joint_ids[2]], color="red")
        ax[2, 1].axvline(t_th, color='orange')

        # RL
        ax[0, 2].set_title("RL")
        ax[0, 2].plot(time, actual_traj[..., self.rl_entity_cfg.joint_ids[0]], color="blue")
        ax[0, 2].plot(time, des_traj[..., self.rl_entity_cfg.joint_ids[0]], color="red")
        ax[0, 2].axvline(t_th, color='orange')

        ax[1, 2].plot(time, actual_traj[..., self.rl_entity_cfg.joint_ids[1]], color="blue")
        ax[1, 2].plot(time, des_traj[..., self.rl_entity_cfg.joint_ids[1]], color="red")
        ax[1, 2].axvline(t_th, color='orange')

        ax[2, 2].plot(time, actual_traj[..., self.rl_entity_cfg.joint_ids[2]], color="blue")
        ax[2, 2].plot(time, des_traj[..., self.rl_entity_cfg.joint_ids[2]], color="red")
        ax[2, 2].axvline(t_th, color='orange')

        # RR
        ax[0, 3].set_title("RR")
        ax[0, 3].plot(time, actual_traj[..., self.rr_entity_cfg.joint_ids[0]], color="blue")
        ax[0, 3].plot(time, des_traj[..., self.rr_entity_cfg.joint_ids[0]], color="red")
        ax[0, 3].axvline(t_th, color='orange')

        ax[1, 3].plot(time, actual_traj[..., self.rr_entity_cfg.joint_ids[1]], color="blue")
        ax[1, 3].plot(time, des_traj[..., self.rr_entity_cfg.joint_ids[1]], color="red")
        ax[1, 3].axvline(t_th, color='orange')

        ax[2, 3].plot(time, actual_traj[..., self.rr_entity_cfg.joint_ids[2]], color="blue")
        ax[2, 3].plot(time, des_traj[..., self.rr_entity_cfg.joint_ids[2]], color="red")
        ax[2, 3].axvline(t_th, color='orange')


    def plot_grf_traj(self, grf_traj, t_min, names, t_th, title="grf"):

        fig, ax = plt.subplots(4, 1, figsize=(10, 8))
        fig.suptitle(title)

        actual_traj = torch.stack(grf_traj, dim=1)[0]

        time = np.arange(0, len(actual_traj)) * 0.005

        ax[0].plot(time, actual_traj[..., 0, 0], color="red", label="x")
        ax[0].plot(time, actual_traj[..., 0, 1], color="green", label="y")
        ax[0].plot(time, actual_traj[..., 0, 2], color="blue", label="z")
        ax[0].set_ylabel(names[0])
        ax[0].axvline(t_min, color='gray')

        ax[1].plot(time, actual_traj[..., 1, 0], color="red")
        ax[1].plot(time, actual_traj[..., 1, 1], color="green")
        ax[1].plot(time, actual_traj[..., 1, 2], color="blue")
        ax[1].set_ylabel(names[1])
        ax[1].axvline(t_min, color='gray')

        ax[2].plot(time, actual_traj[..., 2, 0], color="red")
        ax[2].plot(time, actual_traj[..., 2, 1], color="green")
        ax[2].plot(time, actual_traj[..., 2, 2], color="blue")
        ax[2].set_ylabel(names[2])
        ax[2].axvline(t_min, color='gray')

        ax[3].plot(time, actual_traj[..., 3, 0], color="red")
        ax[3].plot(time, actual_traj[..., 3, 1], color="green")
        ax[3].plot(time, actual_traj[..., 3, 2], color="blue")
        ax[3].set_ylabel(names[3])
        ax[3].axvline(t_min, color='gray')

        ax[0].axvline(t_th, color='orange')
        ax[1].axvline(t_th, color='orange')
        ax[2].axvline(t_th, color='orange')
        ax[3].axvline(t_th, color='orange')

        ax[0].legend()

        fig = plt.figure()
        fig.suptitle(f"avg z {title}")
        plt.plot(time, torch.mean(actual_traj[..., 2], dim=1), color="purple")
        plt.axvline(t_min, color='gray')
        plt.axvline(t_th, color='orange')


    def run_simulator(self):
        """Runs the simulation loop."""
        robot = self.scene["robot"]
        contact_sensor = self.scene["contact_forces"]

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

        grf_traj = []

        sim_time = 0.0
        start_time = 0.0
        max_episode_time = 2 + start_time
        count = 0

        initial__trunk = robot.data.root_state_w[..., 0:7].clone()

        trunk_x_0 = initial__trunk[..., 0:3]
        trunk_xd_0 = robot.data.root_state_w[..., 7:10].clone()
        trunk_o_0 = torch.stack(euler_xyz_from_quat(robot.data.root_state_w[:, 3:7].clone()), dim=1)
        trunk_od_0 = robot.data.root_ang_vel_b.clone()

        action = torch.tensor([[0.2514, -1.2007, 3.0652, -2.5994, -1.2312, -0.2063, -0.4556, -0.0423, -0.0882, 1.8352, -0.4276, 3., 0.2]], device=self.sim.device)
        target = torch.tensor([[0.0, 0, 0]], device=self.sim.device)

        self.bezierAction.process_actions(robot, trunk_x_0, trunk_xd_0, trunk_o_0, trunk_od_0, action, target)

        while True:
            # Simulate physics
            while self.simulation_app.is_running():

                if sim_time > max_episode_time:

                    print(f"landing: {robot.data.root_state_w[..., :7]}")

                    t_th = self.bezierAction.t_th.detach().cpu()

                    self.plot_traj(q_actual_traj, q_des_traj, t_th, "q")
                    self.plot_traj(qd_actual_traj, qd_des_traj, t_th, "qd")
                    self.plot_traj(tau_actual_traj, tau_des_traj, t_th, "tau")
                    t_min = self.plot_trunk_traj(trunk_actual_traj, trunk_des_traj, t_th, "trunk")
                    self.plot_grf_traj(grf_traj, t_min, contact_sensor.body_names, t_th, "grf z")

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

                    grf_traj = []

                    # reset the robot
                    self.reset(robot)

                trunk_des = initial__trunk.clone()
                self.dt = 0

                if sim_time > start_time:
                    self.dt = sim_time - start_time

                if self.dt <= self.bezierAction.t_th_total:
                    x, o = self.bezierAction.eval_bezier(self.dt)
                    trunk_des[..., 0:3] = x
                    q_des, qd_des = self.ik(robot, x, o, old_q_des)
                    old_q_des = q_des.clone()
                else:
                    q_des = self.q_0_lo
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

                # if sim_time > start_time and self.dt <= self.bezierAction.t_th:
                if self.dt <= self.bezierAction.t_th_total:

                    q_actual_traj.append(robot.data.joint_pos.clone().detach().cpu())
                    q_des_traj.append(q_des.detach().cpu())

                    qd_actual_traj.append(robot.data.joint_vel.clone().detach().cpu())
                    qd_des_traj.append(qd_des.detach().cpu())

                    tau_actual_traj.append(robot.data.applied_torque.clone().detach().cpu())
                    tau_des_traj.append(robot.data.computed_torque.clone().detach().cpu())

                    trunk_des_traj.append(trunk_des[..., 0:3].detach().cpu())

                    grf_traj.append(contact_sensor.data.net_forces_w.clone().detach().cpu())

                trunk_actual_traj.append(robot.data.root_state_w[..., 0:3].clone().detach().cpu())

            print("Simulation concluded :)")
            break
