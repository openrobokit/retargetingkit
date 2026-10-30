"""Right Vision-Pro-to-xArm6 teleoperation preset."""

from retargetingkit.configs.teleop.config import ArmTeleopConfig
from retargetingkit.retargeting.arm.online import ArmRetargeterConfig
from retargetingkit.sources.vr.vision_pro import VisionProArmConfig
from retargetingkit.targets.arms.xarm6 import XARM6_EE_LINK, XARM6_HOME_QPOS, XArm6Config
from robokit.assets.robots.arms import xarm6


RIGHT_VISIONPRO_XARM6_CONFIG = ArmTeleopConfig(
    source=VisionProArmConfig(ip="192.168.8.133", recording_fps=50),
    arms={"right": XArm6Config(ip="192.168.1.242")},
    retargeters={
        "right": ArmRetargeterConfig(xarm6.URDF_PATH, XARM6_EE_LINK, XARM6_HOME_QPOS, side="right"),
    },
)
