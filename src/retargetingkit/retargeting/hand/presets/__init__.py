"""Neural-retargeting hand presets: only fingertip links and collision assets are per hand; the URDF gives the rest."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from retargetingkit.sources.frames import MANO21
from robokit.robo.robot_spec import RobotSpec


@dataclass(frozen=True)
class HandPreset:
    """URDF-validated robot-hand geometry and its MANO-21 correspondence."""

    name: str
    urdf_path: Path
    joint_names: tuple[str, ...]
    finger_joint_names: tuple[tuple[str, ...], ...]
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    human_keypoint_indices: tuple[int, ...]
    fingertip_links: tuple[str, ...]
    collision_spheres_path: Path | None
    collision_ignore_path: Path | None
    collision_max_active_pairs: int

    @property
    def finger_joint_sizes(self) -> tuple[int, ...]:
        """Return each finger's number of actuated joints.

        >>> HANDS["xhand"].finger_joint_sizes
        (3, 3, 2, 2, 2)
        """
        return tuple(len(names) for names in self.finger_joint_names)

    @property
    def joint_count(self) -> int:
        """Return the number of actuated joints.

        >>> HANDS["sharpa"].joint_count
        22
        """
        return len(self.joint_names)


def build_hand_preset(
    name: str,
    urdf_path: Path,
    fingertip_links: dict[str, str],
    collision_spheres_path: Path | None,
    collision_ignore_path: Path | None,
    collision_max_active_pairs: int = 50,
) -> HandPreset:
    """Derive one hand preset from its URDF and a MANO-name to tip-link map.

    >>> build_hand_preset("x", Path("x.urdf"), {}, None, None)  # doctest: +SKIP
    """
    robot = RobotSpec.parse(urdf_path, load_meshes=False, mesh_dir=None)
    actuated = set(robot.actuated_joint_names)
    tip_links = tuple(fingertip_links.values())
    assert all(link in robot.link_names for link in tip_links)
    finger_joint_names = tuple(
        tuple(
            robot.joint_names[index]
            for index in np.flatnonzero(robot.link_ancestor_joints_mask[robot.link_names.index(link)])
            if robot.joint_names[index] in actuated
        )
        for link in tip_links
    )
    joint_names = tuple(robot.actuated_joint_names)
    assert tuple(joint for finger in finger_joint_names for joint in finger) == joint_names
    limits = robot.actuated_joint_limits.astype(np.float32)
    return HandPreset(
        name=name,
        urdf_path=urdf_path,
        joint_names=joint_names,
        finger_joint_names=finger_joint_names,
        joint_lower=limits[:, 0],
        joint_upper=limits[:, 1],
        human_keypoint_indices=tuple(MANO21[mano.upper()] for mano in fingertip_links),
        fingertip_links=tip_links,
        collision_spheres_path=collision_spheres_path,
        collision_ignore_path=collision_ignore_path,
        collision_max_active_pairs=collision_max_active_pairs,
    )


from retargetingkit.retargeting.hand.presets.allegro import ALLEGRO  # noqa: E402
from retargetingkit.retargeting.hand.presets.sharpa import SHARPA  # noqa: E402
from retargetingkit.retargeting.hand.presets.xhand import XHAND  # noqa: E402


HANDS = {"sharpa": SHARPA, "allegro": ALLEGRO, "xhand": XHAND}

__all__ = ["ALLEGRO", "HANDS", "SHARPA", "XHAND", "HandPreset", "build_hand_preset"]
