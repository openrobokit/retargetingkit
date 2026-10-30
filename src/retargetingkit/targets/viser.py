"""Passive Viser target: display one or more URDFs, optionally beside a source keypoint skeleton."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from retargetingkit.targets.commands import BasePoseQPosCommand, Command
from retargetingkit.targets.target import Target, TargetConfig
from robokit.robo import Robot


def assembly_qpos_indices(assembly_urdf: Path, component_urdfs: tuple[Path, ...]) -> np.ndarray:
    """Map concatenated component qpos into assembly URDF joint order.

    >>> assembly_qpos_indices(Path("assembly.urdf"), (Path("arm.urdf"), Path("hand.urdf")))  # doctest: +SKIP
    """
    assembly_names = [
        name.removeprefix("left_").removeprefix("right_")
        for name in Robot.load(str(assembly_urdf)).spec.actuated_joint_names
    ]
    component_names = [
        name.removeprefix("left_").removeprefix("right_")
        for urdf in component_urdfs
        for name in Robot.load(str(urdf)).spec.actuated_joint_names
    ]
    assert len(assembly_names) == len(set(assembly_names))
    assert len(component_names) == len(set(component_names))
    assert set(assembly_names) == set(component_names)
    return np.asarray([component_names.index(name) for name in assembly_names], dtype=np.int64)


@dataclass
class ViserConfig(TargetConfig):
    """URDFs to display, in the order callers index them.

    >>> ViserConfig((Path("robot.urdf"),)).urdfs[0].name
    'robot.urdf'
    """

    urdfs: tuple[Path, ...]
    keypoint_edges: tuple[tuple[int, int], ...] = ()

    def build(self) -> "Viser":
        """Start the Viser server.

        >>> ViserConfig((Path("robot.urdf"),)).build()  # doctest: +SKIP
        """
        return Viser(self.urdfs, self.keypoint_edges)


class Viser(Target):
    """Display robot configurations in a browser; `set_qpos` maps robokit's actuated-joint order to the URDF parser's.

    >>> Viser((Path("robot.urdf"),))  # doctest: +SKIP
    """

    def __init__(self, urdfs: tuple[Path, ...], keypoint_edges: tuple[tuple[int, int], ...] = ()) -> None:
        import viser
        from viser.extras import ViserUrdf

        self.server = viser.ViserServer()
        self.server.scene.add_grid("/grid", 2.0, 2.0)
        self._roots = []
        self._urdfs: list[Any] = []
        self._orders: list[list[int]] = []
        for index, urdf in enumerate(urdfs):
            offset = (0.0, 0.35 - 0.7 * index if len(urdfs) > 1 else 0.0, 0.0)
            self._roots.append(self.server.scene.add_frame(f"/robot{index}", position=offset, show_axes=False))
            model = ViserUrdf(self.server, Path(urdf), root_node_name=f"/robot{index}")
            names = Robot.load(str(urdf)).spec.actuated_joint_names
            self._orders.append([names.index(name) for name in model.get_actuated_joint_names()])
            self._urdfs.append(model)
        self._keypoint_edges = np.asarray(keypoint_edges, dtype=np.int64).reshape(-1, 2)
        self.server.scene.add_frame("/keypoints", show_axes=False, position=(0.0, 0.35, 0.0))
        self._keypoints = self.server.scene.add_point_cloud(
            "/keypoints/joints",
            np.zeros((1, 3), dtype=np.float32),
            colors=(255, 180, 50),
            point_size=0.005,
            point_shape="circle",
        )
        self._bones = self.server.scene.add_line_segments(
            "/keypoints/bones",
            np.zeros((len(self._keypoint_edges), 2, 3), dtype=np.float32),
            colors=(255, 180, 50),
            thickness=2.0,
            thickness_units="screen",
        )

    @property
    def port(self) -> int:
        """Actual HTTP port selected by Viser."""
        return self.server.get_port()

    def set_qpos(self, qpos: np.ndarray, index: int = 0) -> None:
        """Display one robot configuration."""
        self._urdfs[index].update_cfg(qpos[self._orders[index]])

    def set_command(self, command: Command) -> None:
        """Display one floating-base configuration."""
        assert isinstance(command, BasePoseQPosCommand)
        with self.server.atomic():
            self._roots[0].position = command.base_pose[:3]
            self._roots[0].wxyz = command.base_pose[3:7]
            self.set_qpos(command.qpos)

    def set_points(self, positions: np.ndarray) -> None:
        """Display the source keypoint skeleton beside the robots."""
        with self.server.atomic():
            self._keypoints.points = positions
            self._bones.points = positions[self._keypoint_edges]

    def close(self) -> None:
        """Stop the Viser server."""
        self.server.stop()


__all__ = ["Viser", "ViserConfig", "assembly_qpos_indices"]
