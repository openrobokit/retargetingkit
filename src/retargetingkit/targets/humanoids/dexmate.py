"""Dexmate Vega-1 whole-body controller (torso+head+arms, 20-DOF) via the dexcontrol SDK."""

from dataclasses import dataclass

import numpy as np

from retargetingkit.targets.target import Target, TargetConfig


_COMPONENTS = (("torso", 3), ("head", 3), ("left_arm", 7), ("right_arm", 7))

JOINT_NAMES: list[str] = (
    [f"torso_j{i}" for i in (1, 2, 3)]
    + [f"head_j{i}" for i in (1, 2, 3)]
    + [f"L_arm_j{i}" for i in range(1, 8)]
    + [f"R_arm_j{i}" for i in range(1, 8)]
)
"""Order of the qpos array this driver accepts and returns; matches the Vega URDF joint names."""


@dataclass
class DexmateConfig(TargetConfig):
    """Dexmate Vega-1 whole-body controller configuration."""

    auto_shutdown: bool = True
    """Register the SDK's atexit shutdown hook. Disable when the caller owns shutdown via `close`."""

    def build(self) -> "Dexmate":
        """Connect to the robot.

        >>> DexmateConfig().build()  # doctest: +SKIP
        """
        return Dexmate(self)


class Dexmate(Target):
    """Dexmate Vega-1 whole-body controller. qpos layout: `JOINT_NAMES` (torso, head, left, right).

    >>> # Dexmate(DexmateConfig())  # doctest: +SKIP
    """

    joint_names = JOINT_NAMES

    def __init__(self, config: DexmateConfig = DexmateConfig()):
        from dexcontrol.robot import Robot

        self._config = config
        self._robot = Robot(auto_shutdown=config.auto_shutdown)

    def set_qpos(self, qpos: np.ndarray, wait_time: float = 0.0, max_vel: float | None = None) -> None:
        """Set targets in `JOINT_NAMES` order; `wait_time>0` blocks while interpolating at up to `max_vel` rad/s."""
        targets, start = {}, 0
        for name, dof in _COMPONENTS:
            targets[name] = qpos[start : start + dof]
            start += dof
        self._robot.set_joint_pos(
            targets, wait_time=wait_time, wait_kwargs={"max_vel": max_vel} if max_vel is not None else None
        )

    def get_qpos(self) -> np.ndarray:
        """Return current joint positions in `JOINT_NAMES` order."""
        return np.concatenate(
            [np.asarray(getattr(self._robot, name).get_joint_pos()) for name, _ in _COMPONENTS]
        ).astype(np.float32)

    def close(self) -> None:
        """Shut down the robot connection."""
        self._robot.shutdown()
