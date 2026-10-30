"""Canonical human-motion frames."""

from abc import ABC
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal

import numpy as np
from jaxtyping import Bool, Float


Side = Literal["left", "right"]
BODY_JOINTS = (
    "pelvis",
    "left_hip",
    "left_knee",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_foot",
    "spine3",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)


class Frame(ABC):
    """Base type for canonical human-motion observations."""


@dataclass
class HandPose:
    """One hand pose: 21 MANO keypoints, wrist-local, meters; orientations default to identity (vector IK ignores them).

    >>> import numpy as np
    >>> f = HandPose(np.zeros((21, 3), dtype=np.float32))
    >>> bool(f.valid.all()) and f.joint_names[MANO21.INDEX_TIP] == "index_tip"
    True
    """

    positions: Float[np.ndarray, "21 3"]
    orientations: Float[np.ndarray, "21 4"] = field(
        default_factory=lambda: np.tile([1.0, 0.0, 0.0, 0.0], (21, 1)).astype(np.float32)
    )  # wxyz identity
    valid: Bool[np.ndarray, "21"] = field(default_factory=lambda: np.ones(21, dtype=bool))
    joint_names: tuple[str, ...] = field(default_factory=lambda: MANO21_JOINTS)


@dataclass
class HandFrame(Frame):
    """Latest left/right hand poses from one physical source.

    >>> frame = HandFrame({"right": HandPose(np.zeros((21, 3), dtype=np.float32))})
    >>> frame.hands["right"].positions.shape
    (21, 3)
    """

    hands: dict[Side, HandPose]


@dataclass
class ArmFrame(Frame):
    """World-frame wrist poses used for arm retargeting.

    >>> frame = ArmFrame({"right": np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)})
    >>> frame.wrists["right"].shape
    (7,)
    """

    wrists: dict[Side, Float[np.ndarray, "7"]]


@dataclass
class ArmHandFrame(Frame):
    """Synchronized arm and hand observations from one physical source.

    >>> arm = ArmFrame({"right": np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)})
    >>> hand = HandFrame({"right": HandPose(np.zeros((21, 3), dtype=np.float32))})
    >>> ArmHandFrame(arm, hand).arm.wrists["right"].shape
    (7,)
    """

    arm: ArmFrame
    hand: HandFrame


@dataclass
class BodyFrame(Frame):
    """Named world-frame body transforms and performer height.

    >>> frame = BodyFrame(("pelvis",), np.zeros((1, 7), dtype=np.float32), 1.75)
    >>> frame.transforms.shape
    (1, 7)
    """

    joint_names: tuple[str, ...]
    transforms: Float[np.ndarray, "J 7"]
    human_height: float


class MANO21(IntEnum):
    """Named indices into the 21-keypoint MANO / OpenPose hand layout.

    >>> MANO21.INDEX_TIP
    <MANO21.INDEX_TIP: 8>
    """

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


MANO21_JOINTS: tuple[str, ...] = tuple(joint.name.lower() for joint in MANO21)

MANO21_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)
