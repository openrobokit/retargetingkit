"""Viser drawing helpers shared by visualization scripts."""

from typing import Any

import numpy as np
from viser import ViserServer
from viser.extras import ViserUrdf

from retargetingkit.retargeting.hand.presets import HandPreset
from retargetingkit.sources.glove.manus_conversion import FINGER_JOINT_RANGES, MANUS_SKELETON_EDGES


CONTACT_COLOR = (40, 90, 230)
_FINGER_COLORS = ((255, 100, 100), (255, 180, 50), (255, 255, 80), (200, 255, 80), (100, 200, 100))
_WRIST_COLOR = (200, 200, 200)


def draw_manus_skeleton(server: ViserServer, positions: np.ndarray, parent_path: str) -> None:
    """Draw a Manus 25-joint skeleton colored by finger.

    >>> draw_manus_skeleton(server, np.zeros((25, 3)), "/hand")  # doctest: +SKIP
    """
    colors = np.full((25, 3), _WRIST_COLOR, dtype=np.uint8)
    for joint_range, color in zip(FINGER_JOINT_RANGES, _FINGER_COLORS):
        colors[list(joint_range)] = color
    # every bone ends on a finger joint, so the child joint picks the bone color
    for edge_index, (start, end) in enumerate(MANUS_SKELETON_EDGES):
        server.scene.add_spline_catmull_rom(
            f"{parent_path}/bone_{edge_index}",
            points=positions[[start, end]],
            color=colors[end],
            thickness=3.0,
            thickness_units="screen",
        )
    server.scene.add_point_cloud(
        f"{parent_path}/joints", positions.astype(np.float32), colors=colors, point_size=0.002, point_shape="circle"
    )


def add_hand_urdf(server: ViserServer, preset: HandPreset, root_node_name: str) -> tuple[ViserUrdf, tuple[int, ...]]:
    """Add a preset hand to Viser and return its joint-order mapping.

    >>> add_hand_urdf(server, HANDS["xhand"], "/robot")  # doctest: +SKIP
    """
    urdf = ViserUrdf(server, preset.urdf_path, root_node_name=root_node_name)
    return urdf, tuple(preset.joint_names.index(name) for name in urdf.get_actuated_joint_names())


def add_object_nodes(
    server: ViserServer, prefix: str, meshes: dict[str, tuple[np.ndarray, np.ndarray]]
) -> dict[str, Any]:
    """Add one Viser mesh node per object.

    >>> add_object_nodes(server, "/robot", {"mug": (vertices, faces)})  # doctest: +SKIP
    """
    return {
        name: server.scene.add_mesh_simple(
            f"{prefix}/object_{name}", vertices=vertices, faces=faces, color=(70, 130, 180), opacity=0.55
        )
        for name, (vertices, faces) in meshes.items()
    }


def add_contact_spheres(server: ViserServer, prefix: str) -> list[Any]:
    """Add one contact-point sphere per finger.

    >>> add_contact_spheres(server, "/robot")  # doctest: +SKIP
    """
    return [
        server.scene.add_icosphere(f"{prefix}/contact_{finger}", radius=0.006, color=CONTACT_COLOR)
        for finger in range(5)
    ]
