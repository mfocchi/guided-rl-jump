
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

    def process_actions(self, actions: torch.Tensor):
        print(actions)
        # store the raw actions
        self._raw_actions[:] = actions
        self.dt = 0

        com_tg = self._env.command_manager.get_command("com_target")[:, 0:3]

        trunk_x_0 = self._asset.data.root_state_w[:, 0:3] - self._env.scene.env_origins
        trunk_xd_0 = self._asset.data.root_lin_vel_b.clone()
        trunk_o_0 = torch.stack(euler_xyz_from_quat(self._asset.data.root_state_w[:, 3:7]), dim=1)
        trunk_od_0 = self._asset.data.root_ang_vel_b.clone()

        # The lo conf is relative to the initial conf
        trunk_x_lo = trunk_x_0.clone()
        trunk_o_lo = trunk_o_0.clone()

        # TODO: just for development, change and take action instead
        # Add relative position to initial position
        trunk_x_lo[:, 2] += 0.05
        trunk_x_lo[:, 0] += 0.25

        trunk_xd_lo = torch.zeros_like(trunk_xd_0)
        trunk_xd_lo[:, 2] = 0.8
        trunk_xd_lo[:, 0] = 0.4

        # Add relative orientation to initial orientation
        trunk_o_lo[:, 2] += 0.2

        # TODO: just for development, change and take action instead
        trunk_od_lo = torch.zeros_like(trunk_xd_0)
        trunk_od_lo[:, 2] += 0.2

        # TODO: just for development, change and take action instead
        self.T_th = torch.rand((self.num_envs, 1), device=self.device)

        # TODO:compute the yaw angle btwn trunk_x_0 and trunk_x_lo for the action
        # TODO: convert back the values from spherical to cartesian

        # Compute the weights of bezier curve for position and orientation
        self.w_x, self.w_xd = self.compute_bezier_w(trunk_x_0, trunk_xd_0, trunk_x_lo, trunk_xd_lo, self.T_th)
        self.w_o, self.w_od = self.compute_bezier_w(trunk_o_0, trunk_od_0, trunk_o_lo, trunk_od_lo, self.T_th)
 

        # apply the affine transformations
        self._processed_actions = self._raw_actions

    def apply_actions(self):

        x, xd = self.bezier_trajectory(self.w_x, self.w_xd, self.dt, self.T_th)
        o, od = self.bezier_trajectory(self.w_o, self.w_od, self.dt, self.T_th)

        traj = torch.cat((x, quat_from_euler_xyz(o[..., 0], o[..., 1], o[..., 2])), dim=-1)

        if self.cfg.debug_vis:
            self.com_lo_vis.visualize(self._env.scene.env_origins + traj[..., 0:3], traj[..., 3:7])

        # set position targets
        self._asset.set_joint_position_target(self.q_0)
        # TODO: send velocity

        self.dt += 0.005
