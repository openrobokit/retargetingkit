"""Right Vision-Pro-to-xArm7-Ability-Hand teleoperation preset."""

from retargetingkit.configs.teleop.config import ArmHandTeleopConfig
from retargetingkit.retargeting.arm.online import ArmRetargeterConfig
from retargetingkit.retargeting.hand.optimization import OptimizationHandRetargeterConfig
from retargetingkit.sources.vr.vision_pro import VisionProArmHandConfig
from retargetingkit.targets.arms.xarm7 import XARM7_EE_LINK, XARM7_HOME_QPOS, XArm7Config
from retargetingkit.targets.hands.ability_hand import AbilityHandConfig
from robokit.assets.robots.arms import xarm7
from robokit.assets.robots.assembly import xarm7_ability
from robokit.assets.robots.end_effectors import ability_hand
from robokit.helpers.hand_retargeting.presets import ability


RIGHT_VISIONPRO_XARM7_ABILITY_CONFIG = ArmHandTeleopConfig(
    source=VisionProArmHandConfig(ip="192.168.8.133", recording_fps=50),
    arms={"right": XArm7Config(ip="192.168.1.242")},
    hands={"right": AbilityHandConfig(port="/dev/ttyUSB0")},
    arm_retargeters={
        "right": ArmRetargeterConfig(xarm7.URDF_GLB_PATH, XARM7_EE_LINK, XARM7_HOME_QPOS, side="right"),
    },
    hand_retargeters={
        "right": OptimizationHandRetargeterConfig(
            ability_hand.URDF_PATH_RIGHT,
            ability.spec,
            ability.online,
            "right",
            collision_spheres=ability_hand.COLLISION_SPHERE_PATH_RIGHT,
        ),
    },
    assembly_urdf=xarm7_ability.URDF_GLB_PATH,
)
