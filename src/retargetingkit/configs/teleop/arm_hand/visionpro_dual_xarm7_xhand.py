"""Dual Vision-Pro-to-xArm7-XHand teleoperation preset."""

from retargetingkit.configs.teleop.config import ArmHandTeleopConfig
from retargetingkit.retargeting.arm.online import ArmRetargeterConfig
from retargetingkit.retargeting.hand.optimization import OptimizationHandRetargeterConfig
from retargetingkit.sources.vr.vision_pro import VisionProArmHandConfig
from retargetingkit.targets.arms.xarm7 import XARM7_EE_LINK, XARM7_HOME_QPOS, XArm7Config
from retargetingkit.targets.hands.xhand import XHandConfig
from robokit.assets.robots.arms import xarm7
from robokit.assets.robots.assembly import xarm7_xhand
from robokit.assets.robots.end_effectors import xhand
from robokit.helpers.hand_retargeting.presets import xhand as xhand_preset


DUAL_VISIONPRO_XARM7_XHAND_CONFIG = ArmHandTeleopConfig(
    source=VisionProArmHandConfig(ip="192.168.8.133", recording_fps=50),
    arms={
        "left": XArm7Config(ip="192.168.1.209"),
        "right": XArm7Config(ip="192.168.1.242"),
    },
    hands={
        "left": XHandConfig(serial_port="/dev/ttyUSB1"),
        "right": XHandConfig(serial_port="/dev/ttyUSB0"),
    },
    arm_retargeters={
        "left": ArmRetargeterConfig(xarm7.URDF_GLB_PATH, XARM7_EE_LINK, XARM7_HOME_QPOS, side="left"),
        "right": ArmRetargeterConfig(xarm7.URDF_GLB_PATH, XARM7_EE_LINK, XARM7_HOME_QPOS, side="right"),
    },
    hand_retargeters={
        "left": OptimizationHandRetargeterConfig(
            xhand.URDF_PATH_LEFT, xhand_preset.spec_left, xhand_preset.online_left, "left"
        ),
        "right": OptimizationHandRetargeterConfig(
            xhand.URDF_PATH_RIGHT,
            xhand_preset.spec,
            xhand_preset.online,
            "right",
            collision_spheres=xhand.COLLISION_SPHERE_PATH_RIGHT,
            collision_ignore=xhand.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT,
        ),
    },
    assembly_urdf=xarm7_xhand.URDF_GLB_PATH,
)
