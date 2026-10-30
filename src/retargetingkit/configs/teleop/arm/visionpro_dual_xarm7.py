"""Dual Vision-Pro-to-xArm7 teleoperation preset."""

from retargetingkit.configs.teleop.config import ArmTeleopConfig
from retargetingkit.retargeting.arm.online import ArmRetargeterConfig
from retargetingkit.sources.vr.vision_pro import VisionProArmConfig
from retargetingkit.targets.arms.xarm7 import XARM7_EE_LINK, XARM7_HOME_QPOS, XArm7Config
from robokit.assets.robots.arms import xarm7


DUAL_VISIONPRO_XARM7_CONFIG = ArmTeleopConfig(
    source=VisionProArmConfig(ip="192.168.8.133", recording_fps=50),
    arms={
        "left": XArm7Config(ip="192.168.1.209"),
        "right": XArm7Config(ip="192.168.1.242"),
    },
    retargeters={
        "left": ArmRetargeterConfig(xarm7.URDF_GLB_PATH, XARM7_EE_LINK, XARM7_HOME_QPOS, side="left"),
        "right": ArmRetargeterConfig(xarm7.URDF_GLB_PATH, XARM7_EE_LINK, XARM7_HOME_QPOS, side="right"),
    },
)
