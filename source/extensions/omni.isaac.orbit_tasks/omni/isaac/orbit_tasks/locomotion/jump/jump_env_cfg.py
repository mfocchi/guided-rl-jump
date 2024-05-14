
from __future__ import annotations

import math
import numpy as np
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


# External variables definition

# friction (which one?)
mu = 0.8
time_step = 0.005

##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=mu)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # robots
    robot: ArticulationCfg = MISSING

    # sensorsc
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*",
                                      track_air_time=True,
                                      visualizer_cfg=CONTACT_SENSOR_JUMP_MARKER_CFG.replace(prim_path="/Visuals/ContactSensor"),
                                      debug_vis=False)
    # contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*(?:foot|trunk)$", track_air_time=True, visualizer_cfg=CONTACT_SENSOR_JUMP_MARKER_CFG.replace(prim_path="/Visuals/ContactSensor"), debug_vis=True)

    # add landing_platform
    landing_platform: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/landing_platform",
        spawn=sim_utils.CuboidCfg(
            size=(0.75, 0.75, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            activate_contact_sensors=True,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=mu),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.2, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.025))
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

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    trunk_target = mdp.UniformTargetCommandCfg(
        asset_name="robot",
        body_name="trunk",
        # command is sampled on a new episodejoint_pos
        # resampling_time > episode_length_s = no target change during episode
        resampling_time_range=(5, 5),
        debug_vis=True,
        ranges=mdp.UniformTargetCommandCfg.Ranges(
            # pos_x=(-1, 1),
            # pos_y=(-1, 1),
            pos_x=(0, 1),
            pos_y=(0, 1),
            pos_z=(0.4, 0.6),
            # TODO: change orientation
            roll=(-np.pi / 6, np.pi / 6),
            pitch=(0, 0),  # depends on end-effector axis
            yaw=(0, 0)
        )
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    jump_traj = mdp.BezierCurveActionCfg(asset_name="robot",
                                         time_step=time_step,
                                         joint_names=[".*"],
                                         fl_joint_names=["FL.*"],
                                         fl_body_names=["FL_foot"],
                                         fr_joint_names=["FR.*"],
                                         fr_body_names=["FR_foot"],
                                         rl_joint_names=["RL.*"],
                                         rl_body_names=["RL_foot"],
                                         rr_joint_names=["RR.*"],
                                         rr_body_names=["RR_foot"],
                                         t_th_min=0.2,
                                         t_th_max=1,
                                         x_theta_min=np.pi / 4,
                                         x_theta_max=np.pi / 2,
                                         x_r_min=0.15,
                                         x_r_max=0.4,
                                         xd_theta_min=np.pi / 6,
                                         xd_theta_max=np.pi / 2,
                                         xd_r_min=0.1,
                                         xd_r_max=4,
                                         psi_min=-2 * np.pi,
                                         psi_max=2 * np.pi,
                                         theta_min=- 2 * np.pi,
                                         theta_max=2 * np.pi,
                                         phi_min=- 2 * np.pi,
                                         phi_max=2 * np.pi,
                                         psid_min=-4,
                                         psid_max=4,
                                         thetad_min=-4,
                                         thetad_max=4,
                                         phid_min=-4,
                                         phid_max=4,
                                         debug_vis=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # TODO: add foot bosition in base f

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))

        target_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "trunk_target"})

        # TODO: add contact state

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot = EventTerm(
        func=mdp.reset_robot_state,
        mode="reset"
    )

    reset_landing_platform = EventTerm(
        func=mdp.reset_landing_platform,
        mode="reset"
    )

    # apex_detection
    move_landing_platform = EventTerm(
        func=mdp.move_landing_platform,
        mode="interval",
        interval_range_s=(0., 0.)
    )

    touchdown = EventTerm(
        func=mdp.touch_down,
        mode="interval",
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
                "asset_cfg": SceneEntityCfg("robot"),
                "air_time_threshold": 0.1,
                "contact_threshold": 20.0},
        interval_range_s=(0., 0.)
    )


@configclass
class RunningRewardsCfg:
    """Running Reward terms for the MDP."""

    # -- Penalities
    #    Must use a negative weight value

    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.1
    )

    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-0.1,
        params={"soft_ratio": 1.0}
    )

    applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.1
    )

    friction_constraint = RewTerm(
        func=mdp.friction_constraint,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
            "mu": mu
        }
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!.*foot).*$"), "threshold": 1.0},
    )

    # TODO: unilaterality, treshold

    # TODO: singularity


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- Task

    target_position_error = RewTerm(
        func=mdp.target_position_error,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="trunk"), "command_name": "trunk_target"},
    )

    target_orientation_error = RewTerm(
        func=mdp.target_orientation_error,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="trunk"), "command_name": "trunk_target"},
    )


@ configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@ configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass
    # TODO: aumentare difficolta


@ configclass
class LocomotionJumpEnvCfg(RLPlanningTaskEnvCfg):
    """Configuration for the locomotion jump environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    running_rewards: RunningRewardsCfg = RunningRewardsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.episode_length_s = 2.0
        self.sim.dt = time_step
        self.decimation = 1
        # simulation settings
        self.sim.disable_contact_processing = True
        # TODO: verify and improve this initialization
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=mu)

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
