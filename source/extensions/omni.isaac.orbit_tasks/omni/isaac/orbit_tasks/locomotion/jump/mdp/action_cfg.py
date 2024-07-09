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
    time_step: float = MISSING

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""

    fl_joint_names: list[str] = MISSING
    fl_body_names: list[str] = MISSING

    fr_joint_names: list[str] = MISSING
    fr_body_names: list[str] = MISSING

    rl_joint_names: list[str] = MISSING
    rl_body_names: list[str] = MISSING

    rr_joint_names: list[str] = MISSING
    rr_body_names: list[str] = MISSING

    class_type: type[ActionTerm] = bezier_curve_actions.BezierCurveAction

    min_action: float = MISSING
    max_action: float = MISSING

    lerp_time: float = MISSING

    # T_th --
    t_th_min: float = MISSING
    t_th_max: float = MISSING

    # -- Lift-off position --
    x_theta_min: float = MISSING
    x_theta_max: float = MISSING

    x_r_min: float = MISSING
    x_r_max: float = MISSING

    # -- Lift-off linear velocity --
    xd_theta_min: float = MISSING
    xd_theta_max: float = MISSING

    xd_r_min: float = MISSING
    xd_r_max: float = MISSING

    # -- Lift-off pose --
    psi_min: float = MISSING
    psi_max: float = MISSING

    theta_min: float = MISSING
    theta_max: float = MISSING

    phi_min: float = MISSING
    phi_max: float = MISSING

    # -- Lift-off angular velocity --
    psid_min: float = MISSING
    psid_max: float = MISSING

    thetad_min: float = MISSING
    thetad_max: float = MISSING

    phid_min: float = MISSING
    phid_max: float = MISSING

    # --explosion phase --

    xd_mult_min: float = MISSING
    xd_mult_max: float = MISSING


    l_expl_min: float = MISSING
    l_expl_max: float = MISSING


    debug_vis: bool = False
