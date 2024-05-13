
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import carb

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR, ISAAC_ORBIT_NUCLEUS_DIR
from omni.isaac.orbit.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.orbit.utils.math import euler_xyz_from_quat, quat_from_euler_xyz
import omni.isaac.orbit.utils.string as string_utils
from omni.isaac.orbit.assets.articulation import Articulation
from omni.isaac.orbit.managers.action_manager import ActionTerm

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

        # TODO: set these in the config
        # T_th --
        self.t_th_min = 0.2
        self.t_th_max = 1
        # Lift-off position --
        # X_theta
        self.x_theta_min = torch.pi / 4
        self.x_theta_max = torch.pi / 2
        # X_r
        self.x_r_min = 0.15
        self.x_r_max = 0.4
        # Lift-off linear velocity --
        # Xd_theta
        self.xd_theta_min = torch.pi / 6
        self.xd_theta_max = torch.pi / 2
        # Xd_r
        self.xd_r_min = 0.1
        self.xd_r_max = 4
        # Lift-off pose --
        # Psi
        self.psi_min = -2 * torch.pi
        self.psi_max = 2 * torch.pi
        # Theta
        self.theta_min = - 2 * torch.pi
        self.theta_max = 2 * torch.pi
        # Phi
        self.phi_min = - 2 * torch.pi
        self.phi_max = 2 * torch.pi

        # Lift-off angular velocity --
        # Psi
        self.psid_min = -4
        self.psid_max = 4
        # Theta
        self.thetad_min = -4
        self.thetad_max = 4
        # Phi
        self.phid_min = -4
        self.phid_max = 4

        self.q_0 = self._asset.data.default_joint_pos.clone()
        self.dt = 0
        # log the resolved joint names for debugging
        # carb.log_info(
        #     f"Resolved joint names for the action term {self.__class__.__name__}:"
        #     f" {self._joint_names} [{self._joint_ids}]"
        # )
        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        if self.cfg.debug_vis:
            self.com_lo_vis = VisualizationMarkers(
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
        # x_lo(sph), xd_lo(sph), o_lo, od_lo, T_th
        return 2 + 2 + 3 + 3 + 1

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

    def process_actions(self, actions: torch.Tensor):

        # store the raw actions
        self._raw_actions[:] = actions
        # WARNING: !!!!!!!!!!!!!!!!!!!!!!!
        # TODO: discuss this part, could destroy the learning algorithm
        self.actions = torch.clip(actions, -1, 1)
        self.dt = 0

        trunk_x_0 = self._asset.data.root_state_w[:, 0:3] - self._env.scene.env_origins
        trunk_xd_0 = self._asset.data.root_lin_vel_b.clone()
        trunk_o_0 = torch.stack(euler_xyz_from_quat(self._asset.data.root_state_w[:, 3:7]), dim=1)
        trunk_od_0 = self._asset.data.root_ang_vel_b.clone()

        trunk_tg = self._env.command_manager.get_command("trunk_target")[:, 0:3]

        self.t_th = (self.t_th_max - self.t_th_min) * 0.5 * (actions[..., 0] + 1) + self.t_th_min
        self.t_th = self.t_th.reshape(-1, 1)

        # Phi is the same for x and xd
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

        # Compute the weights of bezier curve for position and orientation
        self.w_x, self.w_xd = self.compute_bezier_w(trunk_x_0, trunk_xd_0, trunk_x_lo, trunk_xd_lo, self.t_th)
        self.w_o, self.w_od = self.compute_bezier_w(trunk_o_0, trunk_od_0, trunk_o_lo, trunk_od_lo, self.t_th)

        # apply the affine transformations
        self._processed_actions = self.actions

    def apply_actions(self):

        x, xd = self.bezier_trajectory(self.w_x, self.w_xd, self.dt, self.t_th)
        o, od = self.bezier_trajectory(self.w_o, self.w_od, self.dt, self.t_th)

        traj = torch.cat((x, quat_from_euler_xyz(o[..., 0], o[..., 1], o[..., 2])), dim=-1)

        if self.cfg.debug_vis:
            self.com_lo_vis.visualize(self._env.scene.env_origins + traj[..., 0:3], traj[..., 3:7])

        # set position targets
        self._asset.set_joint_position_target(self.q_0)
        # TODO: send velocity

        self.dt += 0.005
