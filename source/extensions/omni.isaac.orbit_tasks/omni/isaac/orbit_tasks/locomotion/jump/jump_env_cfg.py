
from __future__ import annotations

import math
import numpy as np
import torch
from dataclasses import MISSING

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.orbit.envs import RLPlanningTaskEnvCfg
from omni.isaac.orbit.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.orbit.managers import EventTermCfg as EventTerm
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.sensors import ContactSensorCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.orbit.markers.config import CONTACT_SENSOR_JUMP_MARKER_CFG
import omni.isaac.orbit_tasks.locomotion.jump.mdp as mdp

# ============================
# Global terms definition
# ============================

# Simulation
time_step = 0.005

# Terrain

#
mu = 1.0
initial_z = 0.5

# Action config
min_action = -5
max_action = 5


# Target config
# pos_x = (-0.6, 1.2)
# pos_y = (-0.6, 0.6)
# pos_z = (-0.4, 0.4)
# roll = (0, np.pi / 12)
# pitch = (0, np.pi / 12)
# yaw = (-np.pi, np.pi)

# pos_x = (-0.6, 1.2)
# pos_y = (-0.6, 0.6)
# pos_z = (-0.4, 0.4)
# roll = (0, 0)
# pitch = (0, 0)
# yaw = (-np.pi / 2, np.pi / 2)

pos_x = (0., 1.2)
pos_y = (-0.3, 0.3)
pos_z = (0., 0.)
roll = (0, 0)
pitch = (0, 0)
yaw = (0, 0)

# enable extusion of roll or pitch (no uniform random)
# roll_yaw_shufle = True
activate_curriculum = False
roll_yaw_shufle = False
# activate_curriculum = False

# Robot params
trunk_name = ""
foot_name = ""
legs_name = ""
legs_name_calf = ""
thigh_name = ""

robot_height = 0.
foot_offset = 0.

fl_joint_names = []
fl_body_names = []
fr_joint_names = []
fr_body_names = []
rl_joint_names = []
rl_body_names = []
rr_joint_names = []
rr_body_names = []

x_limit = 0.
y_limit = 0.
z_limit = 0.

q_0_lo = torch.tensor([])
mass_range = 0
stiffness_division = 1

mode = ""


# ============================
# Scene definition
# ============================


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=mu, dynamic_friction=mu)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # robots
    robot: ArticulationCfg = MISSING

    # sensorsc
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*(?:foot)$",
                                      track_air_time=True,
                                      visualizer_cfg=CONTACT_SENSOR_JUMP_MARKER_CFG.replace(prim_path="/Visuals/ContactSensor"),
                                      debug_vis=False)

    nonfoot_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/^(?=.*hip)(?=.*trunk).*",
                                      track_air_time=False,
                                      visualizer_cfg=CONTACT_SENSOR_JUMP_MARKER_CFG.replace(prim_path="/Visuals/NonFootContactSensor"),
                                      debug_vis=False)

    # add landing_platform
    landing_platform: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/landing_platform",
        spawn=sim_utils.CuboidCfg(
            size=(2, 2, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            activate_contact_sensors=True,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=mu, dynamic_friction=mu),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.46), roughness=1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.05, 0.0, -0.025))
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )

# ============================
# MDP settings
# ============================


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    trunk_target = mdp.UniformTargetCommandCfgJump(
        asset_name="robot",
        body_name=trunk_name,
        roll_yaw_shufle=roll_yaw_shufle,
        # command is sampled on a new episodejoint_pos
        # resampling_time > episode_length_s = no target change during episode
        resampling_time_range=(5, 5),
        debug_vis=False,
        # Position relative to the current one
        ranges=mdp.UniformTargetCommandCfgJump.Ranges(
            pos_x=pos_x,
            pos_y=pos_y,
            pos_z=pos_z,
            roll=roll,
            pitch=pitch,
            yaw=yaw
        )
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    jump_traj = mdp.BezierCurveActionCfg(asset_name="robot",
                                         time_step=time_step,
                                         joint_names=[".*"],
                                         fl_joint_names=fl_joint_names,
                                         fl_body_names=fl_body_names,
                                         fr_joint_names=fr_joint_names,
                                         fr_body_names=fr_body_names,
                                         rl_joint_names=rl_joint_names,
                                         rl_body_names=rl_body_names,
                                         rr_joint_names=rr_joint_names,
                                         rr_body_names=rr_body_names,
                                         min_action=min_action,
                                         max_action=max_action,
                                         robot_height=robot_height,
                                         contact_threshold=5,
                                         sensor_cfg=SceneEntityCfg("contact_forces", body_names=foot_name),
                                         lerp_time=0.1,
                                         t_th_min=0.4,
                                         t_th_max=0.8,
                                         x_theta_min=np.pi / 4,
                                         x_theta_max=np.pi / 2,
                                         x_r_min=0.1,
                                         x_r_max=0.4,
                                         xd_theta_min=np.pi / 6,
                                         xd_theta_max=np.pi / 2,
                                         xd_r_min=0.5,
                                         xd_r_max=5,
                                         psi_min=-np.pi / 6,
                                         psi_max=np.pi / 6,
                                         theta_min=-np.pi / 6,
                                         theta_max=np.pi / 6,
                                         phi_min=-np.pi / 4,
                                         phi_max=np.pi / 4,
                                         psid_min=-1,
                                         psid_max=1,
                                         thetad_min=-1,
                                         thetad_max=1,
                                         phid_min=-4,
                                         phid_max=4,
                                         xd_mult_min=1,
                                         xd_mult_max=3,
                                         l_expl_min=0.0,
                                         l_expl_max=0.3,
                                         q_0_lo=q_0_lo,
                                         legs_name=legs_name,
                                         legs_name_calf=legs_name_calf,
                                         debug_vis=True,
                                         debug_plot=False,
                                         mode=mode,
                                         debug_control=False,
                                         stiffness_division=stiffness_division)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        target_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "trunk_target"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=foot_name),
            "static_friction_range": (0.7, 0.7),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=trunk_name), "mass_range": (-mass_range, mass_range), "operation": "add"},
    )

    reset_robot = EventTerm(
        func=mdp.reset_robot_state,
        mode="reset",
        params={"initial_z": initial_z}
    )

    reset_landing_platform = EventTerm(
        func=mdp.reset_landing_platform,
        mode="reset",
        params={"initial_z": initial_z}
    )

    detect_apex = EventTerm(
        func=mdp.detect_apex,
        mode="interval",
        interval_range_s=(0., 0.),
        params={"base_lin_vel_threshold": -0.2,
                "foot_height_offset": foot_offset,
                "offset": 0.05,
                "initial_z": initial_z,
                "foot_name": foot_name}
    )

    detect_touchdown = EventTerm(
        func=mdp.detect_touchdown,
        mode="interval",
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=foot_name),
                "asset_cfg": SceneEntityCfg("robot"),
                "contact_threshold": 5.0},
        interval_range_s=(0., 0.)
    )

    detect_fail = EventTerm(
        func=mdp.detect_fail,
        mode="interval",
        params={"sensor_cfg": SceneEntityCfg("nonfoot_forces", body_names=foot_name),
                "contact_threshold": 5.0},
        interval_range_s=(0., 0.)
    )


@configclass
class RunningRewardsCfg:
    """Running Reward terms for the MDP."""

    # -- Penalities
    #    Must use a negative weight value

    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.001
    )

    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-0.001,
        params={"soft_ratio": 1.0}
    )

    applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.0005
    )

    friction_constraint = RewTerm(
        func=mdp.friction_constraint,
        weight=-0.005,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=foot_name),
            "mu": 0.7
        }
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- Task

    target_position_error = RewTerm(
        func=mdp.target_position_error,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=trunk_name),
                "command_name": "trunk_target",
                "coeff": 5.,
                "dist_coeff": 1.,
                "err_coeff": 10.,
                "bias": 1,
                "foot_height_offset": foot_offset,
                "foot_name": foot_name,
                "mode": mode},
    )


@configclass
class NegativeRewardsCfg:

    target_orientation_error = RewTerm(
        func=mdp.target_orientation_error,
        weight=-2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=trunk_name), "command_name": "trunk_target"},
    )

    no_touchdown = RewTerm(
        func=mdp.no_touchdown,
        weight=-0.5,
    )

    liftoff_position_error = RewTerm(
        func=mdp.liftoff_position_error,
        weight=-5,
    )

    liftoff_orientation_error = RewTerm(
        func=mdp.liftoff_orientation_error,
        weight=-1,
    )

    liftoff_linear_velocity_error = RewTerm(
        func=mdp.liftoff_linear_velocity_error,
        weight=-0.1,
    )

    liftoff_angular_velocity_error = RewTerm(
        func=mdp.liftoff_angular_velocity_error,
        weight=-0.1,
    )

    singularity_penalty = RewTerm(
        func=mdp.singularity_penalty,
        params={"x_limit": x_limit, "y_limit": y_limit, "z_limit": z_limit, "initial_z": initial_z},
        weight=-10,
    )

    touchdown_bounce_penalization = RewTerm(
        func=mdp.touchdown_bounce_penalization,
        weight=-1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=trunk_name), }
    )

    touchdown_angular_velocity_penalization = RewTerm(
        func=mdp.touchdown_angular_velocity_penalization,
        weight=-0.01,
    )

    action_limit_penalization = RewTerm(
        func=mdp.action_limit_penalization,
        params={"min_action": min_action, "max_action": max_action},
        weight=-10,
    )


@ configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@ configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    jump_complexity = CurrTerm(
        func=mdp.jump_curriculum, params={"term_name": "trunk_target",
                                          "start": 1.0,
                                          "num_steps": 500,
                                          "num_steps_rp": 1000,
                                          "pos_x": pos_x,
                                          "pos_y": pos_y,
                                          "pos_z": pos_z,
                                          "roll": roll,
                                          "pitch": pitch,
                                          "yaw": yaw,
                                          "activate": activate_curriculum}
    )


@ configclass
class LocomotionJumpEnvCfg(RLPlanningTaskEnvCfg):
    """Configuration for the locomotion jump environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=3)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    running_rewards: RunningRewardsCfg = RunningRewardsCfg()
    rewards: RewardsCfg = RewardsCfg()
    negative_rewards: NegativeRewardsCfg = NegativeRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.episode_length_s = 2
        self.sim.dt = time_step
        self.decimation = 1
        # simulation settings
        self.sim.disable_contact_processing = True

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
