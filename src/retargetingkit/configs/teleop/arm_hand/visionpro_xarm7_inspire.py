"""Right Vision-Pro-to-xArm7-Inspire-Hand teleoperation preset."""

from retargetingkit.configs.teleop.config import ArmHandTeleopConfig
from retargetingkit.retargeting.arm.online import ArmRetargeterConfig
from retargetingkit.retargeting.hand.optimization import OptimizationHandRetargeterConfig
from retargetingkit.sources.vr.vision_pro import VisionProArmHandConfig
from retargetingkit.targets.arms.xarm7 import XARM7_EE_LINK, XARM7_HOME_QPOS, XArm7Config
from retargetingkit.targets.hands.inspire import InspireHandConfig
from robokit.assets.robots.arms import xarm7
from robokit.assets.robots.assembly import xarm7_inspire
from robokit.assets.robots.end_effectors import inspire_hand
from robokit.helpers.hand_retargeting.presets import inspire


RIGHT_VISIONPRO_XARM7_INSPIRE_CONFIG = ArmHandTeleopConfig(
    source=VisionProArmHandConfig(ip="192.168.8.133", recording_fps=50),
    arms={"right": XArm7Config(ip="192.168.1.242")},
    hands={"right": InspireHandConfig(serial_port="/dev/ttyUSB1", hand_id=1, baudrate=115200)},
    arm_retargeters={
        "right": ArmRetargeterConfig(xarm7.URDF_GLB_PATH, XARM7_EE_LINK, XARM7_HOME_QPOS, side="right"),
    },
    hand_retargeters={
        "right": OptimizationHandRetargeterConfig(
            inspire_hand.URDF_PATH_RIGHT,
            inspire.spec,
            inspire.online,
            "right",
            collision_spheres=inspire_hand.COLLISION_SPHERE_PATH_RIGHT,
            collision_ignore=inspire_hand.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT,
        ),
    },
    assembly_urdf=xarm7_inspire.URDF_PATH,
)
