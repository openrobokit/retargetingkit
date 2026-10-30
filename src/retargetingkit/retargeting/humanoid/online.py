"""Canonical body-frame to humanoid-state retargeting."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import tyro

from retargetingkit.retargeting.retargeter import OPAQUE, Retargeter, RetargeterConfig
from retargetingkit.sources.frames import BodyFrame
from retargetingkit.targets.commands import BasePoseQPosCommand
from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOnlineConfig


@dataclass
class HumanoidRetargeterConfig(RetargeterConfig[BodyFrame, BasePoseQPosCommand]):
    """One robokit humanoid preset plus output adjustment.

    >>> HumanoidRetargeterConfig(HumanoidRetargeterOnlineConfig()).z_offset
    0.0
    """

    preset: Annotated[HumanoidRetargeterOnlineConfig, OPAQUE, tyro.conf.Suppress]
    z_offset: float = 0.0

    def build(self, device: str) -> "HumanoidRetargeter":
        """Build and warm up the online retargeter.

        >>> HumanoidRetargeterConfig(g1_aligned).build("cpu")  # doctest: +SKIP
        """
        return HumanoidRetargeter(self, device)


class HumanoidRetargeter(Retargeter[BodyFrame, BasePoseQPosCommand]):
    """Named human body transforms to a floating-base robot command.

    >>> HumanoidRetargeter(config, "cpu")  # doctest: +SKIP
    """

    def __init__(self, config: HumanoidRetargeterConfig, device: str) -> None:
        from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOnline

        assert config.preset.urdf_path is not None
        self.robot = config.preset.load_robot()
        self._retargeter = HumanoidRetargeterOnline(config.preset, device=device, robot=self.robot)
        self._retargeter.warmup(1)
        self.human_joint_names = tuple(self._retargeter.human_joint_names)
        self.urdf_path = Path(config.preset.urdf_path)
        self._z_offset = config.z_offset

    def solve(self, frame: BodyFrame) -> BasePoseQPosCommand:
        indices = [frame.joint_names.index(name) for name in self.human_joint_names]
        height = np.array([frame.human_height], dtype=np.float32)
        robot_pose = self._retargeter.solve_numpy(frame.transforms[indices][None], height)[0]
        robot_pose[2] += self._z_offset
        return BasePoseQPosCommand(robot_pose[7:].copy(), robot_pose[:7].copy())

    def reset(self) -> None:
        """Reset the solver.

        >>> retargeter.reset()  # doctest: +SKIP
        """
        self._retargeter.reset()
