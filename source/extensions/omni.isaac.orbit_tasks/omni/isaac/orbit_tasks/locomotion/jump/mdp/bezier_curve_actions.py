
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
from omni.isaac.orbit.utils.math import subtract_frame_transforms, euler_xyz_from_quat, quat_from_euler_xyz, wrap_to_pi
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

        self.min_action = self.cfg.min_action
        self.max_action = self.cfg.max_action

        self.lerp_time = self.cfg.lerp_time

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

        self.xd_mult_min = self.cfg.xd_mult_min
        self.xd_mult_max = self.cfg.xd_mult_max

        self.l_expl_min = self.cfg.l_expl_min
        self.l_expl_max = self.cfg.l_expl_max

        self.q_0_td = self._asset.data.default_joint_pos.clone()[0]
        # self.q_0_lo = self._asset.data.default_joint_pos.clone()[0]
        # self.q_0_td = torch.tensor([0.1789, -0.1784, 0.1867, -0.1861, 1.2234, 1.2230, 1.4733, 1.4733, -2.2329, -2.2327, -2.1055, -2.1053], device=self.device)
        self.q_0_lo = torch.tensor([0.3430, -0.3425, 0.3433, -0.3424, 1.5495, 1.5490, 1.9171, 1.9173, -2.6620, -2.6618, -2.4902, -2.4901], device=self.device)

        self.default_stiffness = self._asset.actuators["base_legs"].stiffness[0, 0]

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

        self.trunk_tg_vis = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/trajectory",
                markers={
                    "frame": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                        scale=(0.1, 0.1, 0.1),
                    ),
                }))

        if self.cfg.debug_vis:

            self.trunk_lo_vis = VisualizationMarkers(
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
        # t_th, x_lo(sph), xd_lo(sph), o_lo, od_lo, xd_mult, l_expl
        return 1 + 2 + 2 + 3 + 3 + 1 + 1
        # return 1 + 2 + 2

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

        # w_d = torch.stack((
        #     ((3 / t_th) * (w[:, 1] - w[:, 0])).unsqueeze(1),
        #     ((3 / t_th) * (w[:, 2] - w[:, 1])).unsqueeze(1),
        #     ((3 / t_th) * (w[:, 3] - w[:, 2])).unsqueeze(1)), dim=2).squeeze()

        return w  # , w_d

    # def bezier_trajectory(self, w: torch.Tensor, w_d: torch.Tensor, t_ex: float, t_th: torch.Tensor):
    def bezier_trajectory(self, w: torch.Tensor, t_ex: float, t_th: torch.Tensor):

        t = t_ex / t_th

        bezier_curve_3 = torch.cat((
            (1.0) * (t**0) * (1 - t)**(3 - 0),
            (3.0) * (t**1) * (1 - t)**(3 - 1),
            (3.0) * (t**2) * (1 - t)**(3 - 2),
            (1.0) * (t**3) * (1 - t)**(3 - 3),
        ), dim=-1)

        # bezier_curve_2 = torch.cat((
        #     (1.0) * (t**0) * (1 - t)**(2 - 0),
        #     (2.0) * (t**1) * (1 - t)**(2 - 1),
        #     (1.0) * (t**2) * (1 - t)**(2 - 2),
        # ), dim=-1)

        bezier_position = torch.cat((
            (w[..., 0] * bezier_curve_3).sum(dim=-1).reshape(-1, 1),
            (w[..., 1] * bezier_curve_3).sum(dim=-1).reshape(-1, 1),
            (w[..., 2] * bezier_curve_3).sum(dim=-1).reshape(-1, 1)), dim=1)

        # bezier_velocity = torch.cat((
        #     (w_d[..., 0] * bezier_curve_2).sum(dim=-1).reshape(-1, 1),
        #     (w_d[..., 1] * bezier_curve_2).sum(dim=-1).reshape(-1, 1),
        #     (w_d[..., 2] * bezier_curve_2).sum(dim=-1).reshape(-1, 1)), dim=1)

        return bezier_position  # , bezier_velocity

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

    def cerp(self, start, end, weight, start_tangent: float = 1e-3, end_tangent: float = 1e-3):
        # Hermite basis functions
        h00 = (2 * weight**3) - (3 * weight**2) + 1
        h10 = weight**3 - 2 * weight**2 + weight
        h01 = (-2 * weight**3) + (3 * weight**2)
        h11 = weight**3 - weight**2

        # Interpolation
        return (h00 * start) + (h10 * start_tangent) + (h01 * end) + (h11 * end_tangent)

    def ik(self, x, o, old_q_des):

        # Add the origin to get the position for each robot environment
        x += self._env.scene.env_origins
        o_quat = quat_from_euler_xyz(o[..., 0], o[..., 1], o[..., 2])
        # o_quat = self._asset.data.default_root_state[:, 3:7].clone()

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

        after_t_th_total = torch.where(self.dt > self.t_th_total)[0]

        if after_t_th_total.numel() > 0:

            new_lo = after_t_th_total[~torch.isin(after_t_th_total, self._env.extras['after_t_th_total'])]
            if new_lo.numel() > 0:

                self._env.extras['actual_lo_config'][new_lo] = self._asset.data.root_state_w[new_lo].clone()
                self._env.extras['t_th_q'][new_lo] = self._asset.data.joint_pos[new_lo].clone()

            self._env.extras['after_t_th_total'] = after_t_th_total

            elapsed_time = (torch.full_like(self.t_th, self.dt) - self.t_th)[after_t_th_total]
            elapsed_ratio = torch.clip(elapsed_time / torch.full_like(elapsed_time, self.lerp_time), 0, 1)

            q_0_lo_lerp = self.cerp(self._env.extras['t_th_q'][after_t_th_total],
                                    torch.expand_copy(self.q_0_lo, (len(after_t_th_total), len(self.q_0_td))),
                                    elapsed_ratio)

            q_des[after_t_th_total] = q_0_lo_lerp
            qd_des[after_t_th_total] = torch.zeros_like(self._asset.data.default_joint_vel[0])

            # reduce the stiffness to reduce instabilities
            self._asset.actuators["base_legs"].stiffness[after_t_th_total] = torch.full((1, 12), self.default_stiffness / 4).to(self.device)

        apex_env_ids = torch.tensor(list(self._env.extras['apex'].keys()), device=self.device, dtype=torch.int)

        if len(apex_env_ids) > 0:
            apex_elapsed_time = self._env.extras['apex_dt'][apex_env_ids]
            apex_elapsed_ratio = torch.clip(apex_elapsed_time / torch.full_like(apex_elapsed_time, 2*self.lerp_time), 0, 1).reshape(-1, 1)

            q_0_extended = torch.expand_copy(self.q_0_td, (len(apex_env_ids), len(self.q_0_td)))

            q_0_lerp = self.cerp(self._env.extras['apex_q'][apex_env_ids],
                                 q_0_extended,
                                 apex_elapsed_ratio)

            q_des[apex_env_ids] = q_0_lerp

            self._env.extras['apex_dt'][apex_env_ids] += self.cfg.time_step

        return q_des, qd_des

    def map_range(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def process_actions(self, actions: torch.Tensor):
        # clip current action
        actions = torch.clip(actions, self.min_action, self.max_action)
        # store the raw actions
        self._raw_actions[:] = actions

        # reset stiffness
        self._asset.actuators["base_legs"].stiffness = torch.full_like(self._asset.actuators["base_legs"].stiffness, self.default_stiffness)

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
        x_xd_phi = self.torch_cart2sph(trunk_tg.clone())[..., 0]

        # Calculate X_lo
        # x_theta = (self.x_theta_max - self.x_theta_min) * 0.5 * (actions[..., 1] + 1) + self.x_theta_min
        x_theta = self.map_range(actions[..., 1], self.min_action, self.max_action, self.x_theta_min, self.x_theta_max)

        # x_r = (self.x_r_max - self.x_r_min) * 0.5 * (actions[..., 2] + 1) + self.x_r_min
        x_r = self.map_range(actions[..., 2], self.min_action, self.max_action, self.x_r_min, self.x_r_max)

        trunk_x_lo = self.torch_sph2cart(torch.stack((x_xd_phi, x_theta, x_r), dim=1))
        self.trunk_x_lo = trunk_x_lo

        self._env.extras["trunk_x_lo"] = trunk_x_lo

        # Calculate Xd_lo

        # xd_theta = (self.xd_theta_max - self.xd_theta_min) * 0.5 * (actions[..., 3] + 1) + self.xd_theta_min
        xd_theta = self.map_range(actions[..., 3], self.min_action, self.max_action, self.xd_theta_min, self.xd_theta_max)

        # xd_r = (self.xd_r_max - self.xd_r_min) * 0.5 * (actions[..., 4] + 1) + self.xd_r_min
        xd_r = self.map_range(actions[..., 4], self.min_action, self.max_action, self.xd_r_min, self.xd_r_max)

        trunk_xd_lo = self.torch_sph2cart(torch.stack((x_xd_phi, xd_theta, xd_r), dim=1))
        self._env.extras["trunk_xd_lo"] = trunk_xd_lo

        # Calculate Phi_lo

        psi = self.map_range(actions[..., 5], self.min_action, self.max_action, self.psi_min, self.psi_max)
        theta = self.map_range(actions[..., 6], self.min_action, self.max_action, self.theta_min, self.theta_max)
        phi = self.map_range(actions[..., 7], self.min_action, self.max_action, self.phi_min, self.phi_max)

        trunk_o_lo = torch.stack((psi, theta, phi), dim=1)
        self._env.extras["trunk_o_lo"] = quat_from_euler_xyz(trunk_o_lo[..., 0], trunk_o_lo[..., 1], trunk_o_lo[..., 2])

        # Calculate Phid_lo

        psid = self.map_range(actions[..., 8], self.min_action, self.max_action, self.psid_min, self.psid_max)
        thetad = self.map_range(actions[..., 9], self.min_action, self.max_action, self.thetad_min, self.thetad_max)
        phid = self.map_range(actions[..., 10], self.min_action, self.max_action, self.phid_min, self.phid_max)

        trunk_od_lo = torch.stack((psid, thetad, phid), dim=1)
        self._env.extras["trunk_od_lo"] = trunk_od_lo

        # Calculate explosive path

        xd_mult = self.map_range(actions[..., 11], self.min_action, self.max_action, self.xd_mult_min, self.xd_mult_max)
        l_expl = self.map_range(actions[..., 12], self.min_action, self.max_action, self.l_expl_min, self.l_expl_max)

        trunk_xd_lo_un = trunk_xd_lo / torch.norm(trunk_xd_lo, dim=1).reshape(-1, 1)
        self.trunk_x_exp = trunk_x_lo + (trunk_xd_lo_un * l_expl.reshape(-1, 1))

        self._env.extras["trunk_x_exp"] = self.trunk_x_exp

        self.trunk_xd_exp = trunk_xd_lo * xd_mult.reshape(-1, 1)

        self._env.extras["trunk_xd_exp"] = self.trunk_xd_exp

        vf_n = torch.norm(self.trunk_xd_exp, dim=1)
        v0_n = torch.norm(trunk_xd_lo, dim=1)
        sf_n = torch.norm(self.trunk_x_exp, dim=1)
        s0_n = torch.norm(trunk_x_lo, dim=1)

        a = 0.5 * ((torch.pow(vf_n, 2) - torch.pow(v0_n, 2)) / ((sf_n - s0_n) + 1e-15))

        self._env.extras["a"] = a

        self.t_exp = ((vf_n - v0_n) / (a + 1e-15)).reshape(-1, 1)
        self.t_th_total = self.t_th + self.t_exp
        self._env.extras["t_th_total"] = self.t_th_total

        self.trunk_tg_vis.visualize(trunk_x_0 + self._env.command_manager.get_command("trunk_target")[:, 0:3] + self._env.scene.env_origins,
                                    self._env.command_manager.get_command("trunk_target")[:, 3:7])

        # Compute the weights of bezier curve for position and orientation
        self.w_x = self.compute_bezier_w(trunk_x_0, trunk_xd_0, trunk_x_lo, trunk_xd_lo, self.t_th)
        # self.w_o = self.compute_bezier_w(trunk_o_0, trunk_od_0, trunk_o_lo, trunk_od_lo, self.t_th)
        self.w_o = self.compute_bezier_w(trunk_o_0, trunk_od_0, trunk_o_lo, trunk_od_lo, self.t_th_total)

        # apply the affine transformations
        # self._processed_actions = torch.cat((self.t_th, trunk_x_lo, trunk_xd_lo), dim=1)
        self._processed_actions = torch.cat((self.t_th, trunk_x_lo, trunk_xd_lo, trunk_o_lo, trunk_od_lo, xd_mult.reshape(-1, 1), l_expl.reshape(-1, 1), self.trunk_x_exp, self.trunk_xd_exp, self.t_exp), dim=1)

        # # Compute t_flight
        # arg = torch.clip(trunk_xd_lo[..., 2] * trunk_xd_lo[..., 2] - 2 * 9.81 * (trunk_tg[..., 2] - trunk_x_lo[..., 2]), 0, torch.inf)
        # self.t_fl = (trunk_xd_lo[..., 2] + torch.sqrt(arg)) / 9.81
        # print("t_fl", self.t_fl)

        if self.cfg.debug_vis:
            print(f"Command: {self._env.command_manager.get_command('trunk_target')}")
            print(f"Action: {self._raw_actions}")
            print(f"Processed action:\n\
                    t_th: {self._processed_actions[...,0]}\n\
                    trunk_x_lo: {self._processed_actions[...,1:4]}\n\
                    trunk_xd_lo: {self._processed_actions[...,4:7]}\n\
                    trunk_o_lo: {self._processed_actions[...,7:10]}\n\
                    trunk_od_lo: {self._processed_actions[...,10:13]}\n\
                    xd_mult: {self._processed_actions[...,13]}\n\
                    l_expl: {self._processed_actions[...,14]}\n\
                    trunk_x_exp: {self._processed_actions[...,15:18]}\n\
                    trunk_xd_exp: {self._processed_actions[...,18:21]}\n\
                    t_exp: {self._processed_actions[...,21]}\n\
                    a: {a}\n\
                    ")

            self.trunk_lo_vis.visualize(self.trunk_x_exp + self._env.scene.env_origins, quat_from_euler_xyz(trunk_o_lo[..., 0], trunk_o_lo[..., 1], trunk_o_lo[..., 2]))

    def apply_actions(self):
        after_t_th = torch.where(self.dt > self.t_th)[0]

        x = self.bezier_trajectory(self.w_x, self.dt, self.t_th)

        if after_t_th.numel() > 0:
            t = self.dt - self.t_th
            t = torch.clip(t / self.t_exp, 0, 1)
            x_expl = torch.lerp(self.trunk_x_lo, self.trunk_x_exp, t)

            x[after_t_th] = x_expl[after_t_th]

        o = self.bezier_trajectory(self.w_o, self.dt, self.t_th_total)

        q_des, qd_des = self.ik(x, o, self.old_q_des)
        self.old_q_des = q_des.clone()

        self._asset.set_joint_position_target(q_des)
        self._asset.set_joint_velocity_target(qd_des)

        self.dt += self.cfg.time_step
