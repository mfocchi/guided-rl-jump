
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import carb

import omni.isaac.orbit.utils.string as string_utils
from omni.isaac.orbit.assets.articulation import Articulation
from omni.isaac.orbit.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv
    from . import actions_cfg


class BezierCurveAction(ActionTerm):
    r"""
    """

    cfg: actions_cfg.BezierCurveActionCfg
    """The configuration of the action term."""
    _asset: Articulation

    def __init__(self, cfg: actions_cfg.BezierCurveActionCfg, env: BaseEnv) -> None:
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

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return len(self._joint_ids)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        self.dt = 0
        # apply the affine transformations
        self._processed_actions = self._raw_actions

    def apply_actions(self):

        # set position targets
        self._asset.set_joint_position_target(self.q_0)

        self.dt += 0.005
