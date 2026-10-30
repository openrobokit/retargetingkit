"""Allegro hand preset."""

from retargetingkit.retargeting.hand.presets import build_hand_preset
from robokit.assets.robots.end_effectors import allegro_hand


ALLEGRO = build_hand_preset(
    "allegro",
    allegro_hand.URDF_PATH_RIGHT,
    {
        "index_tip": "link_3.0_tip",
        "middle_tip": "link_7.0_tip",
        "ring_tip": "link_11.0_tip",
        "thumb_tip": "link_15.0_tip",
    },
    None,
    None,
)
