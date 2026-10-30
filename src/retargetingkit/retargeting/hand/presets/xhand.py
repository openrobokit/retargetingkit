"""XHand preset."""

from retargetingkit.retargeting.hand.presets import build_hand_preset
from robokit.assets.robots.end_effectors import xhand


XHAND = build_hand_preset(
    "xhand",
    xhand.URDF_PATH_RIGHT,
    {
        "thumb_tip": "right_hand_thumb_rota_tip",
        "index_tip": "right_hand_index_rota_tip",
        "middle_tip": "right_hand_mid_tip",
        "ring_tip": "right_hand_ring_tip",
        "pinky_tip": "right_hand_pinky_tip",
    },
    xhand.COLLISION_SPHERE_PATH_RIGHT,
    xhand.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT,
    32,
)
