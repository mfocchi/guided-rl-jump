import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.actuators import ActuatorNetMLPCfg, DCMotorCfg
from omni.isaac.orbit.assets.articulation import ArticulationCfg
import os
import numpy as np

SOLO_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*_HAA", ".*_HFE", ".*_KFE"],
    effort_limit=10.0,  # taken from spec sheet
    velocity_limit=30.0,  # taken from spec sheet
    saturation_effort=10.0,  # same as effort limit
    stiffness=15.0,
    damping=0.1,
    friction=0.0,
)

# SOLO_ACTUATOR_CFG = DCMotorCfg(
#     joint_names_expr=[".*_HAA", ".*_HFE", ".*_KFE"],
#     effort_limit=23.7,  # taken from spec sheet
#     velocity_limit=30.0,  # taken from spec sheet
#     saturation_effort=23.7,  # same as effort limit
#     stiffness=120.0,
#     damping=0.5,
#     friction=0.0,
# )

SOLO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/riccardo/Documents/orbit/source/extensions/omni.isaac.orbit_assets/data/Robots/OpenRobotics/solo.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.24),
        joint_pos={
            ".*HAA": 0.,
            'FL_HFE': np.pi / 4,
            'FL_KFE': -np.pi / 2,
            'HL_HFE': -np.pi / 4,
            'HL_KFE': np.pi / 2,
            'FR_HFE': np.pi / 4,
            'FR_KFE': -np.pi / 2,
            'HR_HFE': -np.pi / 4,
            'HR_KFE': np.pi / 2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": SOLO_ACTUATOR_CFG,
    },
)
"""Configuration of Unitree Go1 using MLP-based actuator model."""
