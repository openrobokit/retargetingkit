"""Hand-teleop config registry."""

from retargetingkit.configs.teleop.hand.manus_sharpa import (
    RIGHT_MANUS_SHARPA_CONFIG,
    RIGHT_MANUS_SHARPA_NEURAL_CONFIG,
)
from retargetingkit.configs.teleop.hand.manus_xhand import RIGHT_MANUS_XHAND_CONFIG
from retargetingkit.configs.teleop.hand.visionpro_sharpa import RIGHT_VISIONPRO_SHARPA_CONFIG
from retargetingkit.configs.teleop.hand.visionpro_xhand import RIGHT_VISIONPRO_XHAND_CONFIG


_HAND_TELEOP_CONFIGS = {
    "manus_xhand": RIGHT_MANUS_XHAND_CONFIG,
    "manus_sharpa": RIGHT_MANUS_SHARPA_CONFIG,
    "manus_sharpa_neural": RIGHT_MANUS_SHARPA_NEURAL_CONFIG,
    "visionpro_sharpa": RIGHT_VISIONPRO_SHARPA_CONFIG,
    "visionpro_xhand": RIGHT_VISIONPRO_XHAND_CONFIG,
}
