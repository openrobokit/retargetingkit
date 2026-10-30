"""Canonical robot commands produced by retargeters."""

from abc import ABC
from dataclasses import dataclass

import numpy as np


class Command(ABC):
    """Base type for canonical robot commands."""


@dataclass
class QPosCommand(Command):
    """Actuated joint positions in URDF order.

    >>> QPosCommand(np.zeros(7)).qpos.shape
    (7,)
    """

    qpos: np.ndarray


@dataclass
class BasePoseQPosCommand(QPosCommand):
    """Joint positions plus the floating-base world pose as `[xyz, wxyz]`.

    >>> BasePoseQPosCommand(np.zeros(29), np.array([0, 0, 0, 1, 0, 0, 0])).base_pose.shape
    (7,)
    """

    base_pose: np.ndarray
