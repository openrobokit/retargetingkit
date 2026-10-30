"""Arm-and-hand teleop config registry."""

from retargetingkit.configs.teleop.arm_hand.visionpro_dual_xarm7_xhand import (
    DUAL_VISIONPRO_XARM7_XHAND_CONFIG,
)
from retargetingkit.configs.teleop.arm_hand.visionpro_xarm6_sharpa import RIGHT_VISIONPRO_XARM6_SHARPA_CONFIG
from retargetingkit.configs.teleop.arm_hand.visionpro_xarm7_ability import RIGHT_VISIONPRO_XARM7_ABILITY_CONFIG
from retargetingkit.configs.teleop.arm_hand.visionpro_xarm7_inspire import RIGHT_VISIONPRO_XARM7_INSPIRE_CONFIG
from retargetingkit.configs.teleop.arm_hand.visionpro_xarm7_sharpa import RIGHT_VISIONPRO_XARM7_SHARPA_CONFIG
from retargetingkit.configs.teleop.arm_hand.visionpro_xarm7_xhand import RIGHT_VISIONPRO_XARM7_XHAND_CONFIG


_ARM_HAND_TELEOP_CONFIGS = {
    "visionpro_xarm7_xhand": RIGHT_VISIONPRO_XARM7_XHAND_CONFIG,
    "visionpro_dual_xarm7_xhand": DUAL_VISIONPRO_XARM7_XHAND_CONFIG,
    "visionpro_xarm7_ability": RIGHT_VISIONPRO_XARM7_ABILITY_CONFIG,
    "visionpro_xarm7_inspire": RIGHT_VISIONPRO_XARM7_INSPIRE_CONFIG,
    "visionpro_xarm7_sharpa": RIGHT_VISIONPRO_XARM7_SHARPA_CONFIG,
    "visionpro_xarm6_sharpa": RIGHT_VISIONPRO_XARM6_SHARPA_CONFIG,
}
