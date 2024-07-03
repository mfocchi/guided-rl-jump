import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import AssetBaseCfg, RigidObjectCfg
from omni.isaac.orbit.sensors import ContactSensorCfg
from omni.isaac.orbit.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.markers.config import CONTACT_SENSOR_JUMP_MARKER_CFG
from omni.isaac.orbit_assets.unitree import UNITREE_GO1_CFG

mu = 1.0


@configclass
class SceneCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=mu, dynamic_friction=mu)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0)),
    )

    # robots
    robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensorsc
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*(?:foot)$",
                                      track_air_time=True,
                                      visualizer_cfg=CONTACT_SENSOR_JUMP_MARKER_CFG.replace(prim_path="/Visuals/ContactSensor"),
                                      debug_vis=False)

    landing_platform: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/landing_platform",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            activate_contact_sensors=True,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=mu, dynamic_friction=mu),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.2, 0.0)),
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