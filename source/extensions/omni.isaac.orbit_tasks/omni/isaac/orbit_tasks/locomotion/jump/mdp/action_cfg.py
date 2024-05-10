from dataclasses import MISSING

from omni.isaac.orbit.controllers import DifferentialIKControllerCfg
from omni.isaac.orbit.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.orbit.utils import configclass

from . import bezier_curve_actions


@configclass
class BezierCurveActionCfg(ActionTermCfg):
    """Configuration for the base joint action term.

    See :class:`JointAction` for more details.
    """

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""

    class_type: type[ActionTerm] = bezier_curve_actions.BezierCurveAction

    debug_vis: bool = False
