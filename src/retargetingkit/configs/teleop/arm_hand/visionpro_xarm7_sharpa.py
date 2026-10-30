"""Right Vision-Pro-to-xArm7-Sharpa teleoperation preset."""

from retargetingkit.configs.teleop.config import ArmHandTeleopConfig
from retargetingkit.retargeting.arm.online import ArmRetargeterConfig
from retargetingkit.retargeting.hand.optimization import OptimizationHandRetargeterConfig
from retargetingkit.sources.vr.vision_pro import VisionProArmHandConfig
from retargetingkit.targets.arms.xarm7 import XARM7_EE_LINK, XARM7_HOME_QPOS, XArm7Config
from retargetingkit.targets.hands.sharpa_hand import SharpaHandConfig
from robokit.assets.robots.arms import xarm7
from robokit.assets.robots.assembly import xarm7_sharpa
from robokit.assets.robots.end_effectors import sharpa_hand
from robokit.helpers.hand_retargeting.presets import sharpa


RIGHT_VISIONPRO_XARM7_SHARPA_CONFIG = ArmHandTeleopConfig(
    source=VisionProArmHandConfig(ip="192.168.8.133", recording_fps=50),
    arms={"right": XArm7Config(ip="192.168.1.242")},
    hands={"right": SharpaHandConfig(speed_coeff=0.1, current_coeff=0.3, device_sn=None)},
    arm_retargeters={
        "right": ArmRetargeterConfig(xarm7.URDF_GLB_PATH, XARM7_EE_LINK, XARM7_HOME_QPOS, side="right"),
    },
    hand_retargeters={
        "right": OptimizationHandRetargeterConfig(
            sharpa_hand.URDF_PATH_RIGHT,
            sharpa.spec,
            sharpa.online,
            "right",
            collision_spheres=sharpa_hand.COLLISION_SPHERE_PATH_RIGHT,
            collision_ignore=sharpa_hand.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT,
        ),
    },
    assembly_urdf=xarm7_sharpa.URDF_PATH,
)
