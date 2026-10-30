"""Right Vision-Pro-to-XHand teleoperation preset."""

from retargetingkit.configs.teleop.config import HandTeleopConfig
from retargetingkit.retargeting.hand.optimization import OptimizationHandRetargeterConfig
from retargetingkit.sources.vr.vision_pro import VisionProHandConfig
from retargetingkit.targets.hands.xhand import XHandConfig
from robokit.assets.robots.end_effectors import xhand
from robokit.helpers.hand_retargeting.presets import xhand as xhand_preset


RIGHT_VISIONPRO_XHAND_CONFIG = HandTeleopConfig(
    source=VisionProHandConfig(ip="192.168.8.133", recording_fps=30),
    target=XHandConfig(serial_port="/dev/ttyUSB1"),
    retargeter=OptimizationHandRetargeterConfig(
        xhand.URDF_PATH_RIGHT,
        xhand_preset.spec,
        xhand_preset.online,
        "right",
        collision_spheres=xhand.COLLISION_SPHERE_PATH_RIGHT,
        collision_ignore=xhand.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT,
    ),
)
