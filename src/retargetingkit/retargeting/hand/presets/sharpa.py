"""Sharpa hand preset."""

from retargetingkit.retargeting.hand.presets import build_hand_preset
from robokit.assets.robots.end_effectors import sharpa_hand


SHARPA = build_hand_preset(
    "sharpa",
    sharpa_hand.URDF_PATH_RIGHT,
    {
        "thumb_tip": "right_thumb_fingertip",
        "index_tip": "right_index_fingertip",
        "middle_tip": "right_middle_fingertip",
        "ring_tip": "right_ring_fingertip",
        "pinky_tip": "right_pinky_fingertip",
    },
    sharpa_hand.COLLISION_SPHERE_PATH_RIGHT,
    sharpa_hand.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT,
)
