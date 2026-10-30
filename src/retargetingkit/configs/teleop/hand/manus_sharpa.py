"""Right Manus-to-Sharpa teleoperation preset."""

from retargetingkit.configs.teleop.config import HandTeleopConfig
from retargetingkit.retargeting.hand.neural import NeuralHandAssetRetargeterConfig
from retargetingkit.retargeting.hand.optimization import OptimizationHandRetargeterConfig
from retargetingkit.retargeting.hand.presets import HANDS
from retargetingkit.sources.glove.manus_glove import ManusSourceConfig
from retargetingkit.targets.hands.sharpa_hand import SharpaHandConfig
from robokit.assets.robots.end_effectors import sharpa_hand
from robokit.helpers.hand_retargeting.presets import sharpa


RIGHT_MANUS_SHARPA_CONFIG = HandTeleopConfig(
    source=ManusSourceConfig(),
    target=SharpaHandConfig(speed_coeff=1.0, current_coeff=0.5, device_sn=None),
    retargeter=OptimizationHandRetargeterConfig(
        sharpa_hand.URDF_PATH_RIGHT,
        sharpa.spec,
        sharpa.online,
        "right",
        collision_spheres=sharpa_hand.COLLISION_SPHERE_PATH_RIGHT,
        collision_ignore=sharpa_hand.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT,
    ),
    rate=60.0,
)

RIGHT_MANUS_SHARPA_NEURAL_CONFIG = HandTeleopConfig(
    source=ManusSourceConfig(use_calibration=False),
    target=SharpaHandConfig(speed_coeff=1.0, current_coeff=0.5, device_sn=None),
    retargeter=NeuralHandAssetRetargeterConfig("sharpa_motion_alignment_contact_inhand", HANDS["sharpa"]),
    rate=60.0,
)
