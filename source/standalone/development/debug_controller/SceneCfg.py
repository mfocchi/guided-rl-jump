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
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=mu)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*(?:foot)$",
                                      track_air_time=True,
                                      visualizer_cfg=CONTACT_SENSOR_JUMP_MARKER_CFG.replace(prim_path="/Visuals/ContactSensor"),
                                      debug_vis=True)
