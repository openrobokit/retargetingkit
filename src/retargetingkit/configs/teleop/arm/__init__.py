"""Arm-teleop config registry."""

from retargetingkit.configs.teleop.arm.visionpro_dual_xarm7 import DUAL_VISIONPRO_XARM7_CONFIG
from retargetingkit.configs.teleop.arm.visionpro_xarm6 import RIGHT_VISIONPRO_XARM6_CONFIG
from retargetingkit.configs.teleop.arm.visionpro_xarm7 import RIGHT_VISIONPRO_XARM7_CONFIG


_ARM_TELEOP_CONFIGS = {
    "visionpro_xarm7": RIGHT_VISIONPRO_XARM7_CONFIG,
    "visionpro_dual_xarm7": DUAL_VISIONPRO_XARM7_CONFIG,
    "visionpro_xarm6": RIGHT_VISIONPRO_XARM6_CONFIG,
}
