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

    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""
    offset: float | dict[str, float] = 0.0
    """Offset factor for the action (float or dict of regex expressions). Defaults to 0.0."""

    use_default_offset: bool = True

    class_type: type[ActionTerm] = bezier_curve_actions.BezierCurveAction
