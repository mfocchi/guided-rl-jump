
from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING
import time as tm

from multiprocessing import Process, Queue
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

        self.q_lo_threshold = self.cfg.q_lo_threshold

        self.min_action = -5
        self.max_action = 5

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

        self.q_0 = self._asset.data.default_joint_pos.clone()[0]
        self.q_0_lo = torch.tensor([0.2187, -0.2191, 0.2343, -0.2364, 1.3717, 1.3716, 1.6770, 1.6784, -2.4063, -2.4061, -2.2808, -2.2778], device=self.device)

        diff_ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")

        self.fl_diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self._env.scene.num_envs, device=self.device)
        self.fr_diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self._env.scene.num_envs, device=self.device)
        self.rl_diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self._env.scene.num_envs, device=self.device)
        self.rr_diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self._env.scene.num_envs, device=self.device)

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
                        radius=0.05,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.66), opacity=0.3),
                    ),
                }))

        self.trunk_tg_vis = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/trajectory",
                markers={
                    "frame": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                        scale=(0.1, 0.1, 0.1),
                    ),
                }))

        if self.cfg.debug_plot:
            self.queue = Queue()
            self.plot_process = Process(target=self.plot_trajectory, args=(self.queue,))
            self.plot_process.start()
            print("Plotting process has started")

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
        # return 1 + 2 + 2 + 3 + 3
        return 1 + 2 + 2

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

    def ik(self, x, xd, o, od, old_q_des):

        # Add the origin to get the position for each robot environment
        x += self._env.scene.env_origins
        # o_quat = quat_from_euler_xyz(o[..., 0], o[..., 1], o[..., 2])
        o_quat = self._asset.data.default_root_state[:, 3:7].clone()

        if self.cfg.debug_vis:
            self.trunk_traj_vis.visualize(x, o_quat)

        q_des = torch.zeros_like(self._asset.data.default_joint_pos)

        root_pose_w = self._asset.data.root_state_w[:, 0:7].clone()

        fl_jacobian = self._asset.root_physx_view.get_jacobians()[:, self.fl_body_idx, :, np.array(self.fl_entity_cfg.joint_ids) + 6].clone()
        fl_pose_w = self._asset.data.body_state_w[:, self.fl_entity_cfg.body_ids[0], 0:7].clone()
        fl_joint_pos = self._asset.data.joint_pos[:, self.fl_entity_cfg.joint_ids].clone()
        fl_foot_pos_b, fl_foot_orient_b = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], fl_pose_w[:, 0:3], fl_pose_w[:, 3:7])
        fl_foot_pos_des_b, fl_foot_orient_des_b = subtract_frame_transforms(x, o_quat, fl_pose_w[:, 0:3])

        fr_jacobian = self._asset.root_physx_view.get_jacobians()[:, self.fr_body_idx, :, np.array(self.fr_entity_cfg.joint_ids) + 6].clone()
        fr_pose_w = self._asset.data.body_state_w[:, self.fr_entity_cfg.body_ids[0], 0:7].clone()
        fr_joint_pos = self._asset.data.joint_pos[:, self.fr_entity_cfg.joint_ids].clone()
        fr_foot_pos_b, fr_foot_orient_b = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], fr_pose_w[:, 0:3], fr_pose_w[:, 3:7])
        fr_foot_pos_des_b, fr_foot_orient_des_b = subtract_frame_transforms(x, o_quat, fr_pose_w[:, 0:3])

        rl_jacobian = self._asset.root_physx_view.get_jacobians()[:, self.rl_body_idx, :, np.array(self.rl_entity_cfg.joint_ids) + 6].clone()
        rl_pose_w = self._asset.data.body_state_w[:, self.rl_entity_cfg.body_ids[0], 0:7].clone()
        rl_joint_pos = self._asset.data.joint_pos[:, self.rl_entity_cfg.joint_ids].clone()
        rl_foot_pos_b, rl_foot_orient_b = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], rl_pose_w[:, 0:3], rl_pose_w[:, 3:7])
        rl_foot_pos_des_b, rl_foot_orient_des_b = subtract_frame_transforms(x, o_quat, rl_pose_w[:, 0:3])

        rr_jacobian = self._asset.root_physx_view.get_jacobians()[:, self.rr_body_idx, :, np.array(self.rr_entity_cfg.joint_ids) + 6].clone()
        rr_pose_w = self._asset.data.body_state_w[:, self.rr_entity_cfg.body_ids[0], 0:7].clone()
        rr_joint_pos = self._asset.data.joint_pos[:, self.rr_entity_cfg.joint_ids].clone()
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

        qd_des = (q_des - old_q_des) / self.cfg.time_step

        after_t_th = torch.where(self.dt > self.t_th)[0]

        if after_t_th.numel() > 0:
            # # save the lo config of new env after_t_th
            # for after_t_th_env in after_t_th:
            #     after_t_th_env = after_t_th_env.item()
            #     if after_t_th_env not in self._env.extras['after_t_th']:
            #         self._env.extras['actual_lo_config'][after_t_th_env] = self._asset.data.root_state_w[after_t_th_env].clone()

            # update the list of env after_t_th
            self._env.extras['after_t_th'] = after_t_th
            target_distance = torch.norm(self._env.command_manager.get_command("trunk_target")[:, 0:3], dim=1)
            # enable retraction only if jump is grather than threshold
            q_des[torch.where(target_distance[after_t_th] >= self.q_lo_threshold)] = self.q_0_lo
            q_des[torch.where(target_distance[after_t_th] < self.q_lo_threshold)] = self.q_0
            qd_des[after_t_th] = torch.zeros_like(self._asset.data.default_joint_vel.clone())[0]

        apex_env_ids = torch.tensor(list(self._env.extras['apex'].keys()), device=self.device, dtype=torch.int)

        if len(apex_env_ids) > 0:
            q_des[after_t_th] = self.q_0

        return q_des, qd_des

    def plot_trajectory(self, queue):

        plt.ion()
        fig, ax = plt.subplots(3, 1, figsize=(10, 8))

        x_desired = []
        x_actual = []

        while True:
            if not queue.empty():
                data = queue.get()

                if data == "reset":
                    print("Resetting plot")
                    ax[0].cla()
                    ax[1].cla()
                    ax[2].cla()
                    x_desired = []
                    x_actual = []
                else:
                    x_d, x_a = data

                    x_desired.append(x_d)
                    x_actual.append(x_a)

                    x_desired_np = np.stack(x_desired)
                    x_actual_np = np.stack(x_actual)

                    time = np.arange(0, len(x_desired_np)) * 0.005

                    ax[0].clear()
                    ax[1].clear()
                    ax[2].clear()

                    ax[0].plot(time, x_actual_np[..., 0], color='red', label='Actual')
                    ax[0].plot(time, x_desired_np[..., 0], color='blue', label='Desired')
                    ax[0].set_ylabel("X")
                    ax[0].legend()

                    ax[1].plot(time, x_actual_np[..., 1], color='red')
                    ax[1].plot(time, x_desired_np[..., 1], color='blue')
                    ax[1].set_ylabel("Y")

                    ax[2].plot(time, x_actual_np[..., 2], color='red')
                    ax[2].plot(time, x_desired_np[..., 2], color='blue')
                    ax[2].set_ylabel("Z")

                    ax[2].set_xlabel("Time [s]")

                    plt.show()
                    plt.pause(0.01)

            # Check if the plot window is closed
            if not plt.get_fignums():
                print("Plot window closed. Terminating subprocess.")
                break

        # Add a small sleep to avoid busy waiting
        tm.sleep(0.1)

        # Close the plot window before terminating
        plt.close()

    def map_range(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def process_actions(self, actions: torch.Tensor):

        if self.cfg.debug_plot:
            # Reset data
            self.queue.put("reset")

        # clip current action
        actions = torch.clip(actions, self.min_action, self.max_action)
        # store the raw actions
        self._raw_actions[:] = actions

        # reset time counter
        self.dt = 0

        # Reset IK controller
        self.fl_diff_ik_controller.reset()
        self.fr_diff_ik_controller.reset()
        self.rl_diff_ik_controller.reset()
        self.rr_diff_ik_controller.reset()

        # Reset qd computation
        self.old_q_des = self._asset.data.default_joint_pos.clone()

        # TODO: this hold for a robot that is in real world withoud capture sys?
        trunk_x_0 = self._asset.data.root_state_w[:, 0:3].clone() - self._env.scene.env_origins
        trunk_xd_0 = self._asset.data.root_lin_vel_b.clone()
        trunk_o_0 = torch.stack(euler_xyz_from_quat(self._asset.data.root_state_w[:, 3:7].clone()), dim=1)
        trunk_od_0 = self._asset.data.root_ang_vel_b.clone()

        trunk_tg = trunk_x_0 + self._env.command_manager.get_command("trunk_target")[:, 0:3]

        self._env.extras["trunk_tg"] = trunk_tg

        # self.t_th = (self.t_th_max - self.t_th_min) * 0.5 * (actions[..., 0] + 1) + self.t_th_min
        self.t_th = self.map_range(actions[..., 0], self.min_action, self.max_action, self.t_th_min, self.t_th_max)
        self.t_th = self.t_th.reshape(-1, 1)

        # Phi is the same for x and xd
        x_xd_phi = self.torch_cart2sph(trunk_tg)[..., 0].clone()

        # Calculate X_lo
        # x_theta = (self.x_theta_max - self.x_theta_min) * 0.5 * (actions[..., 1] + 1) + self.x_theta_min
        x_theta = self.map_range(actions[..., 1], self.min_action, self.max_action, self.x_theta_min, self.x_theta_max)

        # x_r = (self.x_r_max - self.x_r_min) * 0.5 * (actions[..., 2] + 1) + self.x_r_min
        x_r = self.map_range(actions[..., 2], self.min_action, self.max_action, self.x_r_min, self.x_r_max)

        trunk_x_lo = self.torch_sph2cart(torch.stack((x_xd_phi, x_theta, x_r), dim=1))
        self._env.extras["trunk_x_lo"] = trunk_x_lo

        # Calculate Xd_lo

        # xd_theta = (self.xd_theta_max - self.xd_theta_min) * 0.5 * (actions[..., 3] + 1) + self.xd_theta_min
        xd_theta = self.map_range(actions[..., 3], self.min_action, self.max_action, self.xd_theta_min, self.xd_theta_max)

        # xd_r = (self.xd_r_max - self.xd_r_min) * 0.5 * (actions[..., 4] + 1) + self.xd_r_min
        xd_r = self.map_range(actions[..., 4], self.min_action, self.max_action, self.xd_r_min, self.xd_r_max)

        trunk_xd_lo = self.torch_sph2cart(torch.stack((x_xd_phi, xd_theta, xd_r), dim=1))
        self._env.extras["trunk_xd_lo"] = trunk_xd_lo

        # # Calculate Phi_lo

        # psi = (self.psi_max - self.psi_min) * 0.5 * (actions[..., 5] + 1) + self.psi_min
        # theta = (self.theta_max - self.theta_min) * 0.5 * (actions[..., 6] + 1) + self.theta_min
        # phi = (self.phi_max - self.phi_min) * 0.5 * (actions[..., 7] + 1) + self.phi_min

        # trunk_o_lo = torch.stack((psi, theta, phi), dim=1)

        # # Calculate Phid_lo

        # psid = (self.psid_max - self.psid_min) * 0.5 * (actions[..., 8] + 1) + self.psid_min
        # thetad = (self.thetad_max - self.thetad_min) * 0.5 * (actions[..., 9] + 1) + self.thetad_min
        # phid = (self.phid_max - self.phid_min) * 0.5 * (actions[..., 10] + 1) + self.phid_min

        # trunk_od_lo = torch.stack((psid, thetad, phid), dim=1)

        self.trunk_lo_vis.visualize(trunk_x_lo + self._env.scene.env_origins)
        self.trunk_tg_vis.visualize(trunk_x_0 + self._env.command_manager.get_command("trunk_target")[:, 0:3] + self._env.scene.env_origins,
                                    self._env.command_manager.get_command("trunk_target")[:, 3:7])

        # Compute the weights of bezier curve for position and orientation
        self.w_x, self.w_xd = self.compute_bezier_w(trunk_x_0, trunk_xd_0, trunk_x_lo, trunk_xd_lo, self.t_th)
        # self.w_o, self.w_od = self.compute_bezier_w(trunk_o_0, trunk_od_0, trunk_o_lo, trunk_od_lo, self.t_th)

        # apply the affine transformations
        self._processed_actions = torch.cat((self.t_th, trunk_x_lo, trunk_xd_lo), dim=1)

        if self.cfg.debug_vis:
            # print(f"Command: {self._env.command_manager.get_command('trunk_target')}")
            print(f"Action: {self._raw_actions}")
            print(f"Processed action: {self._processed_actions}")
            # print(f"x_theta, x_r: { x_theta} , {x_r}")
            # print(f"xd_theta, xd_r: { xd_theta} , {xd_r}")
            # print(f"Pos in jf: {self.torch_sph2cart(torch.stack((torch.zeros_like(x_xd_phi), x_theta, x_r), dim=1))}")
            # print(f"Vel in jf: {self.torch_sph2cart(torch.stack((torch.zeros_like(x_xd_phi), xd_theta, xd_r), dim=1))}")

    def apply_actions(self):

        x, xd = self.bezier_trajectory(self.w_x, self.w_xd, self.dt, self.t_th)
        # o, od = self.bezier_trajectory(self.w_o, self.w_od, self.dt, self.t_th)
        # TODO: debug orientation and enable it
        o = None
        od = None

        q_des, qd_des = self.ik(x, xd, o, od, self.old_q_des)
        self.old_q_des = q_des.clone()

        self._asset.set_joint_position_target(q_des)
        self._asset.set_joint_velocity_target(qd_des)

        if self.cfg.debug_plot:

            if self.dt < self.t_th[0]:
                # Collect the current desired and actual positions for plotting
                desired_pos = x[0].clone().cpu().numpy()
                actual_pos = self._asset.data.root_state_w[0, 0:3].clone().cpu().numpy()

                # Send the positions to the plotting process
                self.queue.put((desired_pos, actual_pos))

        self.dt += self.cfg.time_step

    def __del__(self):
        if self.cfg.debug_plot:
            if self.plot_process.is_alive():
                self.plot_process.terminate()
                self.plot_process.join()
