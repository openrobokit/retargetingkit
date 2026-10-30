"""Wrapper around RoboKit online-optimization hand retargeting shared by the teleop runtimes."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated

import numpy as np
import tyro

from retargetingkit.retargeting.retargeter import OPAQUE, Retargeter, RetargeterConfig
from retargetingkit.sources.frames import HandFrame, Side
from retargetingkit.targets.commands import QPosCommand
from robokit.helpers.hand_retargeting import HandRetargeterOnline
from robokit.helpers.hand_retargeting.config import HandRetargeterOnlineConfig, HandSpec
from robokit.robo import Robot


@dataclass
class OptimizationHandRetargeterConfig(RetargeterConfig[HandFrame, QPosCommand]):
    """Online-optimization hand retargeting; `spec`/`online` come from `robokit.helpers.hand_retargeting.presets`.

    >>> OptimizationHandRetargeterConfig(Path("hand.urdf"), spec, online, "right").side  # doctest: +SKIP
    'right'
    """

    urdf: Path
    spec: Annotated[HandSpec, tyro.conf.Suppress]
    online: Annotated[HandRetargeterOnlineConfig, OPAQUE, tyro.conf.Suppress]
    side: Side
    collision_spheres: Path | None = None
    collision_ignore: Path | None = None

    def build(self, device: str) -> "OptimizationHandRetargeter":
        """Build and warm up the optimization retargeter.

        >>> config.build("cpu")  # doctest: +SKIP
        """
        return OptimizationHandRetargeter(self, device)


class OptimizationHandRetargeter(Retargeter[HandFrame, QPosCommand]):
    """Canonical hand frame to robot-hand joint positions.

    >>> retargeter = config.build("cpu")  # doctest: +SKIP
    """

    def __init__(self, config: OptimizationHandRetargeterConfig, device: str) -> None:
        self.side: Side = config.side
        self.urdf_path = config.urdf
        robot = Robot.load(
            config.urdf,
            load_collision_spheres=config.collision_spheres is not None,
            collision_spheres_path=config.collision_spheres,
            self_collision_ignore_path=config.collision_ignore,
        )
        self._retargeter = HandRetargeterOnline(
            config=config.online,
            robot=robot,
            spec=replace(config.spec, floating_base=False),
            device=device,
        )
        self._retargeter.warmup(1)
        self.robot = robot

    def solve(self, frame: HandFrame) -> QPosCommand:
        hand = frame.hands[self.side]
        qpos = self._retargeter.solve_numpy(hand.positions)[0, 7:].astype(np.float32)
        return QPosCommand(qpos)
