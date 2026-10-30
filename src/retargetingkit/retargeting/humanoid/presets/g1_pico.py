"""PICO (XRoboToolkit) to Unitree G1 preset.

- Solver weights and scale table come from our own G1 calibration.
- Link names and rotation offsets come from GMR's `xrobot_to_g1.json`, calibrated to PICO's local joint frames.
- Hence `*_hip_yaw_link` rather than robokit's `*_hip_roll_link`.
- Quaternions are scalar-first (w, x, y, z).
- Pair with `use_ankle_as_foot_target=False` so PICO's own foot joints drive the IK targets.
"""

import dataclasses
from copy import deepcopy

import numpy as np

from retargetingkit.retargeting.humanoid.presets.g1 import g1
from robokit.assets.robots.humanoids import unitree_g1
from robokit.helpers.humanoid_retargeting.config import HumanoidContext, LinkMapping
from robokit.terms.dense.frame_task import FrameTask
from robokit.terms.dense.position_limit import PositionLimit


_PELVIS_ROTATION = [-0.5, 0.5, -0.5, -0.5]
_ZERO = [0.0, 0.0, 0.0]
_TOE = [0.05, 0.0, 0.0]
_SQRT_HALF = 0.7071067811865475

# link: (human joint, position weight, orientation weight, rotation, position offset)
_LINKS = {
    "pelvis": ("pelvis", 2000, 600, _PELVIS_ROTATION, _ZERO),
    "left_hip_yaw_link": ("left_hip", 10, 15, _PELVIS_ROTATION, _ZERO),
    "left_knee_link": ("left_knee", 10, 15, _PELVIS_ROTATION, _ZERO),
    "left_toe_link": ("left_foot", 100, 50, _PELVIS_ROTATION, _TOE),
    "right_hip_yaw_link": ("right_hip", 10, 15, _PELVIS_ROTATION, _ZERO),
    "right_knee_link": ("right_knee", 10, 15, _PELVIS_ROTATION, _ZERO),
    "right_toe_link": ("right_foot", 100, 50, _PELVIS_ROTATION, _TOE),
    "torso_link": ("spine3", 0, 30, _PELVIS_ROTATION, _ZERO),
    "left_shoulder_yaw_link": ("left_shoulder", 30, 45, [_SQRT_HALF, 0.0, _SQRT_HALF, 0.0], _ZERO),
    "left_elbow_link": ("left_elbow", 30, 45, [0.0, 0.0, 1.0, 0.0], _ZERO),
    "left_wrist_yaw_link": ("left_wrist", 30, 45, [0.0, 0.0, 1.0, 0.0], _ZERO),
    "right_shoulder_yaw_link": ("right_shoulder", 30, 45, [0.0, _SQRT_HALF, 0.0, -_SQRT_HALF], _ZERO),
    "right_elbow_link": ("right_elbow", 30, 45, [0.0, 1.0, 0.0, 0.0], _ZERO),
    "right_wrist_yaw_link": ("right_wrist", 30, 45, [0.0, 1.0, 0.0, 0.0], _ZERO),
}


g1_pico = dataclasses.replace(
    g1,
    urdf_path=str(unitree_g1.URDF_PATH.with_name("g1_custom_collision_29dof.urdf")),
    human_height_assumption=1.66,
    ground_height=-1.3,
    scale_table={
        "pelvis": 1.0714,
        "spine3": 0.8526,
        "left_hip": 0.9269,
        "right_hip": 0.9265,
        "left_knee": 0.8395,
        "right_knee": 0.8389,
        "left_foot": 0.8178,
        "right_foot": 0.8087,
        "left_shoulder": 0.8143,
        "right_shoulder": 0.8161,
        "left_elbow": 0.7785,
        "right_elbow": 0.7761,
        "left_wrist": 0.7444,
        "right_wrist": 0.7444,
    },
    link_mapping={
        link: LinkMapping(joint, np.array(position, np.float32), np.array(rotation, np.float32))
        for link, (joint, _, _, rotation, position) in _LINKS.items()
    },
    terms=[
        FrameTask(
            frame_index=HumanoidContext.TARGET_LINKS,
            T_world_target=HumanoidContext.TARGETS,
            position_weight={link: entry[1] for link, entry in _LINKS.items()},
            orientation_weight={link: entry[2] for link, entry in _LINKS.items()},
            name="target",
        ),
        PositionLimit(weight=30.0, residual_mode="exp_barrier", frame=HumanoidContext.FRAME, name="position_limit"),
        *deepcopy(g1.terms[2:]),
    ],
)


__all__ = ["g1_pico"]
