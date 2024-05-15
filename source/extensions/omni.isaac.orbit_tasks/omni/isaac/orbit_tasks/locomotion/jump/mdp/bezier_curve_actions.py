
from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

import carb

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR, ISAAC_ORBIT_NUCLEUS_DIR
from omni.isaac.orbit.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.orbit.utils.math import subtract_frame_transforms, euler_xyz_from_quat, quat_from_euler_xyz
import omni.isaac.orbit.utils.string as string_utils
from omni.isaac.orbit.assets.articulation import Articulation
from omni.isaac.orbit.managers.action_manager import ActionTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.controllers import DifferentialIKController, DifferentialIKControllerCfg


if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv
    from . import actions_cfg


class BezierCurveAction(ActionTerm):
    r"""
    """

    cfg: actions_cfg.BezierCurveActionCfg
    """The configuration of the action term."""
    _asset: Articulation

    def __init__(self, cfg: actions_cfg.BezierCurveActionCfg, env: RLTaskEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)

        self.fl_entity_cfg = SceneEntityCfg(self.cfg.asset_name, joint_names=self.cfg.fl_joint_names, body_names=self.cfg.fl_body_names)
        self.fl_entity_cfg.resolve(self._env.scene)
        self.fl_body_idx = self.fl_entity_cfg.body_ids[0]

        self.fr_entity_cfg = SceneEntityCfg(self.cfg.asset_name, joint_names=self.cfg.fr_joint_names, body_names=self.cfg.fr_body_names)
        self.fr_entity_cfg.resolve(self._env.scene)
        self.fr_body_idx = self.fr_entity_cfg.body_ids[0]

        self.rl_entity_cfg = SceneEntityCfg(self.cfg.asset_name, joint_names=self.cfg.rl_joint_names, body_names=self.cfg.rl_body_names)
        self.rl_entity_cfg.resolve(self._env.scene)
        self.rl_body_idx = self.rl_entity_cfg.body_ids[0]

        self.rr_entity_cfg = SceneEntityCfg(self.cfg.asset_name, joint_names=self.cfg.rr_joint_names, body_names=self.cfg.rr_body_names)
        self.rr_entity_cfg.resolve(self._env.scene)
        self.rr_body_idx = self.rr_entity_cfg.body_ids[0]

        self.t_th_min = self.cfg.t_th_min
        self.t_th_max = self.cfg.t_th_max

        self.x_theta_min = self.cfg.x_theta_min
        self.x_theta_max = self.cfg.x_theta_max

        self.x_r_min = self.cfg.x_r_min
        self.x_r_max = self.cfg.x_r_max

        self.xd_theta_min = self.cfg.xd_theta_min
        self.xd_theta_max = self.cfg.xd_theta_max

        self.xd_r_min = self.cfg.xd_r_min
        self.xd_r_max = self.cfg.xd_r_max

        self.psi_min = self.cfg.psi_min
        self.psi_max = self.cfg.psi_max

        self.theta_min = self.cfg.theta_min
        self.theta_max = self.cfg.theta_max

        self.phi_min = self.cfg.phi_min
        self.phi_max = self.cfg.phi_max

        self.psid_min = self.cfg.psid_min
        self.psid_max = self.cfg.psid_max

        self.thetad_min = self.cfg.thetad_min
        self.thetad_max = self.cfg.thetad_max

        self.phid_min = self.cfg.phid_min
        self.phid_max = self.cfg.phid_max

        self.q_0 = self._asset.data.default_joint_pos.clone()

        diff_ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")

        self.fl_diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self._env.scene.num_envs, device=self._env.device)
        self.fr_diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self._env.scene.num_envs, device=self._env.device)
        self.rl_diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self._env.scene.num_envs, device=self._env.device)
        self.rr_diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self._env.scene.num_envs, device=self._env.device)

        # time passed from the start of the action
        self.dt = 0

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        self.trunk_lo_vis = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/trajectory",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.025,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.66)),
                    ),
                }))

        if self.cfg.debug_plot:
            self.figure = plt.figure(figsize=(10, 5))

            # Initialize lists to store trajectories
            self.desired_trajectory = []
            self.actual_trajectory = []

            # Initialize plot
            plt.ion()
            self.fig, self.ax = plt.subplots(3, 1, figsize=(8, 6))
            self.lines = []

        if self.cfg.debug_vis:
            self.trunk_traj_vis = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/trajectory",
                    markers={
                        "frame": sim_utils.UsdFileCfg(
                            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                            scale=(0.1, 0.1, 0.1),
                        ),
                    }))

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        # t_th, x_lo(sph), xd_lo(sph), o_lo, od_lo
        return 1 + 2 + 2 + 3 + 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def compute_bezier_w(self, start: torch.Tensor, start_v: torch.Tensor, end: torch.Tensor, end_v: torch.Tensor, t_th: torch.Tensor):

        w = torch.stack((
            start.unsqueeze(1),
            (start - (1. / 3.) * t_th * start_v).unsqueeze(1),
            (end - (1. / 3.) * t_th * end_v).unsqueeze(1),
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

    def ik(self, x, xd, o, od):

        # Add the origin to get the position for each robot environment
        x += self._env.scene.env_origins
        # x = self._asset.data.default_root_state[:, 0:3] + self._env.scene.env_origins
        # TODO: fix and enable orientation
        # o_quat = quat_from_euler_xyz(o[..., 0], o[..., 1], o[..., 2])
        o_quat = self._asset.data.default_root_state[:, 3:7]

        if self.cfg.debug_vis:
            self.trunk_traj_vis.visualize(x, o_quat)

        q_des = torch.zeros_like(self._asset.data.default_joint_pos)

        root_pose_w = self._asset.data.root_state_w[:, 0:7]

        fl_jacobian = self._asset.root_physx_view.get_jacobians()[:, self.fl_body_idx, :, np.array(self.fl_entity_cfg.joint_ids) + 6]
        fl_pose_w = self._asset.data.body_state_w[:, self.fl_entity_cfg.body_ids[0], 0:7]
        fl_joint_pos = self._asset.data.joint_pos[:, self.fl_entity_cfg.joint_ids]
        fl_foot_pos_b, fl_foot_orient_b = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], fl_pose_w[:, 0:3], fl_pose_w[:, 3:7])
        fl_foot_pos_des_b, fl_foot_orient_des_b = subtract_frame_transforms(x, o_quat, self.fl_foot_pos_w_0)

        fr_jacobian = self._asset.root_physx_view.get_jacobians()[:, self.fr_body_idx, :, np.array(self.fr_entity_cfg.joint_ids) + 6]
        fr_pose_w = self._asset.data.body_state_w[:, self.fr_entity_cfg.body_ids[0], 0:7]
        fr_joint_pos = self._asset.data.joint_pos[:, self.fr_entity_cfg.joint_ids]
        fr_foot_pos_b, fr_foot_orient_b = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], fr_pose_w[:, 0:3], fr_pose_w[:, 3:7])
        fr_foot_pos_des_b, fr_foot_orient_des_b = subtract_frame_transforms(x, o_quat, self.fr_foot_pos_w_0)

        rl_jacobian = self._asset.root_physx_view.get_jacobians()[:, self.rl_body_idx, :, np.array(self.rl_entity_cfg.joint_ids) + 6]
        rl_pose_w = self._asset.data.body_state_w[:, self.rl_entity_cfg.body_ids[0], 0:7]
        rl_joint_pos = self._asset.data.joint_pos[:, self.rl_entity_cfg.joint_ids]
        rl_foot_pos_b, rl_foot_orient_b = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], rl_pose_w[:, 0:3], rl_pose_w[:, 3:7])
        rl_foot_pos_des_b, rl_foot_orient_des_b = subtract_frame_transforms(x, o_quat, self.rl_foot_pos_w_0)

        rr_jacobian = self._asset.root_physx_view.get_jacobians()[:, self.rr_body_idx, :, np.array(self.rr_entity_cfg.joint_ids) + 6]
        rr_pose_w = self._asset.data.body_state_w[:, self.rr_entity_cfg.body_ids[0], 0:7]
        rr_joint_pos = self._asset.data.joint_pos[:, self.rr_entity_cfg.joint_ids]
        rr_foot_pos_b, rr_foot_orient_b = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], rr_pose_w[:, 0:3], rr_pose_w[:, 3:7])
        rr_foot_pos_des_b, rr_foot_orient_des_b = subtract_frame_transforms(x, o_quat, self.rr_foot_pos_w_0)

        # Orientation is just for visualization (ignore)
        self.fl_diff_ik_controller.set_command(fl_foot_pos_des_b, ee_quat=fl_foot_orient_des_b)
        self.fr_diff_ik_controller.set_command(fr_foot_pos_des_b, ee_quat=fr_foot_orient_des_b)
        self.rl_diff_ik_controller.set_command(rl_foot_pos_des_b, ee_quat=rl_foot_orient_des_b)
        self.rr_diff_ik_controller.set_command(rr_foot_pos_des_b, ee_quat=rr_foot_orient_des_b)

        q_des[:, self.fl_entity_cfg.joint_ids] = self.fl_diff_ik_controller.compute(fl_foot_pos_b, fl_foot_orient_b, fl_jacobian, fl_joint_pos)
        q_des[:, self.fr_entity_cfg.joint_ids] = self.fr_diff_ik_controller.compute(fr_foot_pos_b, fr_foot_orient_b, fr_jacobian, fr_joint_pos)
        q_des[:, self.rl_entity_cfg.joint_ids] = self.rl_diff_ik_controller.compute(rl_foot_pos_b, rl_foot_orient_b, rl_jacobian, rl_joint_pos)
        q_des[:, self.rr_entity_cfg.joint_ids] = self.rr_diff_ik_controller.compute(rr_foot_pos_b, rr_foot_orient_b, rr_jacobian, rr_joint_pos)

        if self.cfg.debug_plot:
            self.desired_trajectory.append(x[0].cpu().numpy())
            self.actual_trajectory.append(self._asset.data.root_state_w[0, 0:3].cpu().numpy())

        # TODO: fix computation of joint velocity targets
        qd_des = (q_des - self._asset.data.joint_pos) / self.cfg.time_step

        after_t_th = torch.where(self.dt > self.t_th)[0]

        if after_t_th.numel() > 0:
            q_des[after_t_th] = self._asset.data.default_joint_pos[0]
            qd_des[after_t_th] = self._asset.data.default_joint_vel[0]

        return q_des, qd_des

    def process_actions(self, actions: torch.Tensor):

        if self.cfg.debug_plot:
            self.desired_trajectory = []
            self.actual_trajectory = []

        # store the raw actions
        self._raw_actions[:] = actions
        # WARNING: !!!!!!!!!!!!!!!!!!!!!!!
        # TODO: discuss this part, could destroy the learning algorithm
        actions = torch.clip(actions, -1, 1)

        # reset time counter
        self.dt = 0

        # Reset IK controller
        self.fl_diff_ik_controller.reset()
        self.fr_diff_ik_controller.reset()
        self.rl_diff_ik_controller.reset()
        self.rr_diff_ik_controller.reset()

        # Fixed foot positions in wf
        self.fl_foot_pos_w_0 = torch.stack([torch.tensor([0.176, 0.178, 0.0]) for i in range(self.num_envs)]).to(self.device) + self._env.scene.env_origins
        self.fr_foot_pos_w_0 = torch.stack([torch.tensor([0.176, -0.178, 0.0]) for i in range(self.num_envs)]).to(self.device) + self._env.scene.env_origins
        self.rl_foot_pos_w_0 = torch.stack([torch.tensor([-0.260, 0.178, 0.0]) for i in range(self.num_envs)]).to(self.device) + self._env.scene.env_origins
        self.rr_foot_pos_w_0 = torch.stack([torch.tensor([-0.260, -0.178, 0.0]) for i in range(self.num_envs)]).to(self.device) + self._env.scene.env_origins

        # TODO: this hold for a robot that is in real world withoud capture sys?
        trunk_x_0 = self._asset.data.root_state_w[:, 0:3] - self._env.scene.env_origins
        trunk_xd_0 = self._asset.data.root_lin_vel_b.clone()
        trunk_o_0 = torch.stack(euler_xyz_from_quat(self._asset.data.root_state_w[:, 3:7]), dim=1)
        trunk_od_0 = self._asset.data.root_ang_vel_b.clone()

        trunk_tg = self._env.command_manager.get_command("trunk_target")[:, 0:3]

        self.t_th = (self.t_th_max - self.t_th_min) * 0.5 * (actions[..., 0] + 1) + self.t_th_min
        self.t_th = self.t_th.reshape(-1, 1)

        # Phi is the same for x and xd
        # TODO: check this!
        x_xd_phi = self.torch_cart2sph(trunk_tg)[..., 0]
        # Calculate X_lo
        x_theta = (self.x_theta_max - self.x_theta_min) * 0.5 * (actions[..., 1] + 1) + self.x_theta_min
        x_r = (self.x_r_max - self.x_r_min) * 0.5 * (actions[..., 2] + 1) + self.x_r_min

        trunk_x_lo = self.torch_sph2cart(torch.stack((x_xd_phi, x_theta, x_r), dim=1))

        # Calculate Xd_lo

        xd_theta = (self.xd_theta_max - self.xd_theta_min) * 0.5 * (actions[..., 3] + 1) + self.xd_theta_min
        xd_r = (self.xd_r_max - self.xd_r_min) * 0.5 * (actions[..., 4] + 1) + self.xd_r_min

        trunk_xd_lo = self.torch_sph2cart(torch.stack((x_xd_phi, xd_theta, xd_r), dim=1))

        # Calculate Phi_lo

        psi = (self.psi_max - self.psi_min) * 0.5 * (actions[..., 5] + 1) + self.psi_min
        theta = (self.theta_max - self.theta_min) * 0.5 * (actions[..., 6] + 1) + self.theta_min
        phi = (self.phi_max - self.phi_min) * 0.5 * (actions[..., 7] + 1) + self.phi_min

        trunk_o_lo = torch.stack((psi, theta, phi), dim=1)

        # Calculate Phid_lo

        psid = (self.psid_max - self.psid_min) * 0.5 * (actions[..., 8] + 1) + self.psid_min
        thetad = (self.thetad_max - self.thetad_min) * 0.5 * (actions[..., 9] + 1) + self.thetad_min
        phid = (self.phid_max - self.phid_min) * 0.5 * (actions[..., 10] + 1) + self.phid_min

        trunk_od_lo = torch.stack((psid, thetad, phid), dim=1)

        # self.trunk_lo_vis.visualize(trunk_x_lo + self._env.scene.env_origins)
        print(trunk_x_lo.shape)

        # Compute the weights of bezier curve for position and orientation
        self.w_x, self.w_xd = self.compute_bezier_w(trunk_x_0, trunk_xd_0, trunk_x_lo, trunk_xd_lo, self.t_th)
        self.w_o, self.w_od = self.compute_bezier_w(trunk_o_0, trunk_od_0, trunk_o_lo, trunk_od_lo, self.t_th)

        # apply the affine transformations
        self._processed_actions = actions

    def plot_trajectory(self, x_desired, x_actual):
        time = np.arange(0, len(x_desired[..., 0])) * self.cfg.time_step
        if not self.lines:
            self.lines.append(self.ax[0].plot(time, x_desired[..., 0], color='red')[0])
            self.lines.append(self.ax[0].plot(time, x_actual[..., 0], color='blue')[0])
            self.lines.append(self.ax[1].plot(time, x_desired[..., 1], color='red')[0])
            self.lines.append(self.ax[1].plot(time, x_actual[..., 1], color='blue')[0])
            self.lines.append(self.ax[2].plot(time, x_desired[..., 2], color='red')[0])
            self.lines.append(self.ax[2].plot(time, x_actual[..., 2], color='blue')[0])
        else:
            self.lines[0].set_data(time, x_desired[..., 0])
            self.lines[1].set_data(time, x_actual[..., 0])
            self.lines[2].set_data(time, x_desired[..., 1])
            self.lines[3].set_data(time, x_actual[..., 1])
            self.lines[4].set_data(time, x_desired[..., 2])
            self.lines[5].set_data(time, x_actual[..., 2])
            self.ax[0].relim()
            self.ax[0].autoscale_view()
            self.ax[1].relim()
            self.ax[1].autoscale_view()
            self.ax[2].relim()
            self.ax[2].autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def apply_actions(self):

        x, xd = self.bezier_trajectory(self.w_x, self.w_xd, self.dt, self.t_th)
        o, od = self.bezier_trajectory(self.w_o, self.w_od, self.dt, self.t_th)

        q_des, qd_des = self.ik(x, xd, o, od)

        # self._asset.set_joint_position_target(self._asset.data.default_joint_pos)
        self._asset.set_joint_position_target(q_des)
        self._asset.set_joint_velocity_target(qd_des)
        # print(q_des)

        if self.cfg.debug_plot:
            # Draw until end of t_th
            if self.dt <= self.t_th[..., 0]:
                self.plot_trajectory(
                    np.stack(self.desired_trajectory),
                    np.stack(self.actual_trajectory)
                )

        self.dt += self.cfg.time_step
