"""Right Vision-Pro-to-Sharpa teleoperation preset."""

from retargetingkit.configs.teleop.config import HandTeleopConfig
from retargetingkit.retargeting.hand.optimization import OptimizationHandRetargeterConfig
from retargetingkit.sources.vr.vision_pro import VisionProHandConfig
from retargetingkit.targets.hands.sharpa_hand import SharpaHandConfig
from robokit.assets.robots.end_effectors import sharpa_hand
from robokit.helpers.hand_retargeting.presets import sharpa


RIGHT_VISIONPRO_SHARPA_CONFIG = HandTeleopConfig(
    source=VisionProHandConfig(ip="192.168.8.133", recording_fps=30),
    target=SharpaHandConfig(device_sn=None),
    retargeter=OptimizationHandRetargeterConfig(
        sharpa_hand.URDF_PATH_RIGHT,
        sharpa.spec,
        sharpa.online,
        "right",
        collision_spheres=sharpa_hand.COLLISION_SPHERE_PATH_RIGHT,
        collision_ignore=sharpa_hand.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT,
    ),
)
