"""Visual-only offline hand-object retargeting."""

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import trimesh
import warp as wp

from retargetingkit.sources.frames import Side
from robokit.geom import MeshGeom, WarpScene
from robokit.helpers.hand_retargeting import HandRetargeterOffline, HandRetargeterOfflineConfig, HandSpec
from robokit.helpers.hand_retargeting.config import refine_config
from robokit.robo import Robot
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.numpy import inverse_tf_mat, pose7_to_tf_mat, quaternion_apply, tf_mat_to_pose7


MANO16_LINKS = (
    "palm",
    "thumb1",
    "thumb2",
    "thumb3",
    "index1",
    "index2",
    "index3",
    "middle1",
    "middle2",
    "middle3",
    "ring1",
    "ring2",
    "ring3",
    "pinky1",
    "pinky2",
    "pinky3",
)


@dataclass(frozen=True)
class ObjectPart:
    """One rigid part of an articulated object."""

    name: str
    vertices: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class HandTrack:
    """One wrist-local MANO-21 hand track and its object-part contacts."""

    keypoints_wrist: np.ndarray
    wrist_pose_world: np.ndarray
    contact_points_part: np.ndarray
    contact_normals_part: np.ndarray
    contact_part: np.ndarray
    contact_mask: np.ndarray


@dataclass(frozen=True)
class ArticulatedObjectTrack:
    """Object root, articulation, rigid parts, and each part's world pose."""

    name: str
    root_pose_world: np.ndarray
    qpos: np.ndarray
    parts: tuple[ObjectPart, ...]
    part_pose_world: np.ndarray


@dataclass(frozen=True)
class HandObjectClip:
    """Robot-independent bimanual interaction clip; poses are `[x, y, z, qw, qx, qy, qz]` in meters."""

    name: str
    fps: float
    hands: dict[Side, HandTrack]
    object: ArticulatedObjectTrack

    @property
    def num_frames(self) -> int:
        """Number of synchronized frames.

        >>> obj = ArticulatedObjectTrack("o", np.zeros((3, 7)), np.zeros((3, 1)), (), np.zeros((3, 0, 7)))
        >>> HandObjectClip("x", 30.0, {}, obj).num_frames
        3
        """
        return int(self.object.root_pose_world.shape[0])

    def keypoints_world(self, side: Side) -> np.ndarray:
        """Return one hand's MANO-21 keypoints in world coordinates.

        >>> pose = np.array([[1, 2, 3, 1, 0, 0, 0]], dtype=np.float32)
        >>> hand = HandTrack(np.zeros((1, 21, 3)), pose, np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), np.zeros((1, 1), np.int8), np.zeros((1, 1), bool))
        >>> obj = ArticulatedObjectTrack("o", pose, np.zeros((1, 1)), (), np.zeros((1, 0, 7)))
        >>> HandObjectClip("x", 30.0, {"right": hand}, obj).keypoints_world("right")[0, 0].tolist()
        [1.0, 2.0, 3.0]
        """
        hand = self.hands[side]
        pose = hand.wrist_pose_world
        return (pose[:, None, :3] + quaternion_apply(pose[:, None, 3:], hand.keypoints_wrist)).astype(np.float32)

    def contacts_world(self, side: Side) -> np.ndarray:
        """Return one hand's semantic contacts in world coordinates.

        >>> pose = np.array([[[1, 2, 3, 1, 0, 0, 0]]], dtype=np.float32)
        >>> hand = HandTrack(np.zeros((1, 21, 3)), np.zeros((1, 7)), np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), np.zeros((1, 1), np.int8), np.ones((1, 1), bool))
        >>> obj = ArticulatedObjectTrack("o", pose[:, 0], np.zeros((1, 1)), (), pose)
        >>> HandObjectClip("x", 30.0, {"right": hand}, obj).contacts_world("right")[0, 0].tolist()
        [1.0, 2.0, 3.0]
        """
        hand = self.hands[side]
        frames = np.arange(self.num_frames)[:, None]
        pose = self.object.part_pose_world[frames, hand.contact_part]
        return (pose[..., :3] + quaternion_apply(pose[..., 3:], hand.contact_points_part)).astype(np.float32)

    def contact_normals_world(self, side: Side) -> np.ndarray:
        """Return one hand's outward object contact normals in world coordinates.

        >>> pose = np.array([[[0, 0, 0, 1, 0, 0, 0]]], dtype=np.float32)
        >>> hand = HandTrack(np.zeros((1, 21, 3)), np.zeros((1, 7)), np.zeros((1, 1, 3)), np.array([[[0, 0, 1]]], np.float32), np.zeros((1, 1), np.int8), np.ones((1, 1), bool))
        >>> obj = ArticulatedObjectTrack("o", pose[:, 0], np.zeros((1, 1)), (), pose)
        >>> HandObjectClip("x", 30.0, {"right": hand}, obj).contact_normals_world("right")[0, 0].tolist()
        [0.0, 0.0, 1.0]
        """
        hand = self.hands[side]
        frames = np.arange(self.num_frames)[:, None]
        pose = self.object.part_pose_world[frames, hand.contact_part]
        return quaternion_apply(pose[..., 3:], hand.contact_normals_part).astype(np.float32)

    def select(self, frames: np.ndarray | slice) -> "HandObjectClip":
        """Select frames while sharing immutable object meshes.

        >>> obj = ArticulatedObjectTrack("o", np.zeros((3, 7)), np.zeros((3, 1)), (), np.zeros((3, 0, 7)))
        >>> HandObjectClip("x", 30.0, {}, obj).select(slice(1, 3)).num_frames
        2
        """
        hands: dict[Side, HandTrack] = {
            side: HandTrack(
                hand.keypoints_wrist[frames],
                hand.wrist_pose_world[frames],
                hand.contact_points_part[frames],
                hand.contact_normals_part[frames],
                hand.contact_part[frames],
                hand.contact_mask[frames],
            )
            for side, hand in self.hands.items()
        }
        obj = self.object
        return HandObjectClip(
            self.name,
            self.fps,
            hands,
            ArticulatedObjectTrack(
                obj.name,
                obj.root_pose_world[frames],
                obj.qpos[frames],
                obj.parts,
                obj.part_pose_world[frames],
            ),
        )


def load_hand_object_clip(path: str | Path) -> HandObjectClip:
    """Load a numeric-only canonical hand-object NPZ.

    >>> load_hand_object_clip("clip.npz").num_frames  # doctest: +SKIP
    """
    with np.load(path) as data:
        parts = tuple(
            ObjectPart(
                str(name),
                data[f"object_part_{index}_vertices"].copy(),
                data[f"object_part_{index}_faces"].copy(),
            )
            for index, name in enumerate(data["object_part_names"].tolist())
        )
        hands: dict[Side, HandTrack] = {
            side: HandTrack(
                data[f"{side}_keypoints_wrist"].copy(),
                data[f"{side}_wrist_pose_world"].copy(),
                data[f"{side}_contact_points_part"].copy(),
                data[f"{side}_contact_normals_part"].copy(),
                data[f"{side}_contact_part"].copy(),
                data[f"{side}_contact_mask"].copy(),
            )
            for side in data["sides"].tolist()
        }
        obj = ArticulatedObjectTrack(
            str(data["object_name"]),
            data["object_root_pose_world"].copy(),
            data["object_qpos"].copy(),
            parts,
            data["object_part_pose_world"].copy(),
        )
        return HandObjectClip(str(data["name"]), float(data["fps"]), hands, obj)


_TIP_CONTACT_INDICES = np.array([3, 6, 9, 12, 15], dtype=np.int32)


def derive_fingertip_contacts(
    parts: tuple[ObjectPart, ...],
    part_pose_world: np.ndarray,
    tips_world: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pick each fingertip's closest object part and threshold contact; return MANO-16 points, normals, parts, masks.

    >>> box = trimesh.creation.box((0.2, 0.2, 0.2))
    >>> part = ObjectPart("box", box.vertices.astype(np.float32), box.faces.astype(np.int32))
    >>> pose = np.array([[[0, 0, 0, 1, 0, 0, 0]]], dtype=np.float32)
    >>> tips = np.array([[[0.101, 0, 0]] * 5], dtype=np.float32)
    >>> points, normals, part_index, mask = derive_fingertip_contacts((part,), pose, tips, 0.003)
    >>> bool(mask[0, 3]), [round(value, 3) for value in points[0, 3].tolist()]
    (True, [0.1, 0.0, 0.0])
    """
    num_frames = tips_world.shape[0]
    part_pose_inverse = inverse_tf_mat(pose7_to_tf_mat(part_pose_world))
    closest_by_part = []
    normal_by_part = []
    distance_by_part = []
    for part_index, part in enumerate(parts):
        mesh = trimesh.Trimesh(part.vertices, part.faces, process=False)
        transform = part_pose_inverse[:, part_index]
        tips_part = np.einsum("fij,fpj->fpi", transform[:, :3, :3], tips_world) + transform[:, None, :3, 3]
        closest, distance, triangle = trimesh.proximity.closest_point(mesh, tips_part.reshape(-1, 3))
        closest_by_part.append(closest.reshape(num_frames, 5, 3).astype(np.float32))
        normal_by_part.append(mesh.face_normals[triangle].reshape(num_frames, 5, 3).astype(np.float32))
        distance_by_part.append(distance.reshape(num_frames, 5).astype(np.float32))
    distances = np.stack(distance_by_part)
    contact_part_tip = np.argmin(distances, axis=0).astype(np.int8)
    frame_index = np.arange(num_frames)[:, None]
    finger_index = np.arange(5)[None]
    contact_points = np.zeros((num_frames, 16, 3), dtype=np.float32)
    contact_normals = np.zeros_like(contact_points)
    contact_part = np.zeros((num_frames, 16), dtype=np.int8)
    contact_mask = np.zeros((num_frames, 16), dtype=np.bool_)
    contact_points[:, _TIP_CONTACT_INDICES] = np.stack(closest_by_part)[contact_part_tip, frame_index, finger_index]
    contact_normals[:, _TIP_CONTACT_INDICES] = np.stack(normal_by_part)[contact_part_tip, frame_index, finger_index]
    contact_part[:, _TIP_CONTACT_INDICES] = contact_part_tip
    contact_mask[:, _TIP_CONTACT_INDICES] = distances[contact_part_tip, frame_index, finger_index] < threshold
    return contact_points, contact_normals, contact_part, contact_mask


CONTACT_LINK_TARGETS = (
    ("palm", "wrist"),
    ("thumb2", "thumb_mcp"),
    ("thumb3", "thumb_tip"),
    ("index2", "index_pip"),
    ("index3", "index_tip"),
    ("middle2", "middle_pip"),
    ("middle3", "middle_tip"),
    ("ring2", "ring_pip"),
    ("ring3", "ring_tip"),
    ("pinky2", "pinky_pip"),
    ("pinky3", "pinky_tip"),
)
FINGERTIP_CONTACTS = tuple(pair for pair in CONTACT_LINK_TARGETS if pair[1].endswith("_tip"))
CONTACT_CLEARANCE = 0.01


def _compute_inward_contact_points(
    robot: Robot, link_indices: np.ndarray, link_poses: np.ndarray, normals: np.ndarray
) -> np.ndarray:
    """Return each link mesh point facing furthest into the object."""
    inward_local = np.einsum("fsji,fsj->fsi", link_poses[..., :3, :3], -normals)
    points = np.zeros(inward_local.shape, dtype=np.float32)
    for slot, link in enumerate(link_indices):
        geom = robot.spec.link_visual_geometries.get(robot.spec.link_names[link])
        vertices = None if geom is None else np.asarray(geom.to_mesh().vertices, np.float32)
        if vertices is not None and vertices.shape[0]:
            points[:, slot] = vertices[np.argmax(inward_local[:, slot] @ vertices.T, axis=1)]
    return points


@dataclass(frozen=True)
class RobotHandTrack:
    """Retargeted joint trajectory and floating wrist base.

    >>> RobotHandTrack(np.zeros((2, 6)), np.zeros((2, 7))).qpos.shape
    (2, 6)
    """

    qpos: np.ndarray
    base_pose_world: np.ndarray


@dataclass(frozen=True)
class HandObjectMotion:
    """Retargeted hands paired with the source object's exact kinematic track."""

    clip: HandObjectClip
    hands: dict[Side, RobotHandTrack]


def hand_object_configs(
    offline: HandRetargeterOfflineConfig, surface_contact: bool = False
) -> tuple[HandRetargeterOfflineConfig, HandRetargeterOfflineConfig]:
    """Derive smooth hand tracking and contact refinement from an offline preset; only refinement keeps `optimize_base`.

    >>> track, refine = hand_object_configs(HandRetargeterOfflineConfig())
    >>> track.task("velocity").weight, refine.task("contact").weight
    (30.0, 300.0)
    """
    track = replace(offline, optimize_base=True, terms=deepcopy(offline.terms))
    track.task("velocity").weight = 30.0
    track.task("acceleration").weight = 10.0
    track.task("root_smoothness").base_weight[3:6] = 50.0  # pyright: ignore[reportAttributeAccessIssue]
    refine = refine_config(replace(track, optimize_base=offline.optimize_base))
    refine.task("acceleration").weight = 100.0
    refine.task("contact").contact_margin = 0.0  # pyright: ignore[reportAttributeAccessIssue]
    if surface_contact:
        refine.task("collision").weight = 100.0
        refine.task("collision").margin = 0.0  # pyright: ignore[reportAttributeAccessIssue]
        refine.task("contact").weight = 100.0
        refine.task("surface_contact").weight = 300.0
        refine.task("anchor").dense_task.weight = 20.0
        refine.task("base_anchor").dense_task.base_weight = 300.0  # pyright: ignore[reportAttributeAccessIssue]
    else:
        refine.task("collision").weight = 0.0
        refine.task("contact").weight = 300.0
        refine.task("anchor").dense_task.weight = 10.0
    return track, refine


class HandObjectRetargeter:
    """Two-stage kinematic solve: `track` follows the hand keypoints, then `refine` anchors that motion to contacts."""

    def __init__(
        self,
        robots: dict[Side, Robot],
        specs: dict[Side, HandSpec],
        track: HandRetargeterOfflineConfig,
        refine: HandRetargeterOfflineConfig,
        device: str,
        surface_contact: bool = False,
    ) -> None:
        self.robots = robots
        self.specs = specs
        self.track = track
        self.refine = refine
        self.device = device
        self.surface_contact = surface_contact

    def solve(self, clip: HandObjectClip) -> HandObjectMotion:
        """Retarget the available hands without dynamics or object motion optimization.

        >>> retargeter.solve(clip)  # doctest: +SKIP
        """
        contact_indices = [MANO16_LINKS.index(name) for name, _ in CONTACT_LINK_TARGETS]
        target_names = tuple(target for _, target in CONTACT_LINK_TARGETS)
        fingertip_indices = [MANO16_LINKS.index(name) for name, _ in FINGERTIP_CONTACTS]
        fingertip_targets = tuple(target for _, target in FINGERTIP_CONTACTS)
        outputs: dict[Side, RobotHandTrack] = {}
        T_world_bottom = pose7_to_tf_mat(clip.object.part_pose_world[:, 0])
        T_bottom_world = inverse_tf_mat(T_world_bottom)
        R_bottom_world = T_bottom_world[:, :3, :3]
        scene = None
        scene_indices = None
        if self.surface_contact:
            part_meshes = [
                wp.Mesh(
                    points=wp.from_numpy(part.vertices, dtype=wp.vec3, device=self.device),
                    indices=wp.from_numpy(part.faces.reshape(-1), dtype=wp.int32, device=self.device),
                )
                for part in clip.object.parts
            ]
            num_parts = len(part_meshes)
            scene = WarpScene(clip.num_frames, self.device).add(
                MeshGeom(
                    [mesh for _ in range(clip.num_frames) for mesh in part_meshes],
                    np.arange(0, (clip.num_frames + 1) * num_parts, num_parts, dtype=np.int32),
                    poses=wp.from_numpy(
                        np.ascontiguousarray(
                            (T_bottom_world[:, None] @ pose7_to_tf_mat(clip.object.part_pose_world)).reshape(-1, 4, 4)
                        ),
                        dtype=wp.mat44,
                        device=self.device,
                    ),
                )
            )
            scene_indices = wp.from_numpy(
                np.arange(clip.num_frames, dtype=np.int32)[None], dtype=wp.int32, device=self.device
            )
        for side in clip.hands:
            robot = self.robots[side]
            spec = replace(self.specs[side], contact_target_names=target_names)
            points_world = clip.keypoints_world(side)
            points_bottom = (
                np.einsum("fij,ftj->fti", R_bottom_world, points_world) + T_bottom_world[:, None, :3, 3]
            ).astype(np.float32)
            wrist_bottom = tf_mat_to_pose7(T_bottom_world @ pose7_to_tf_mat(clip.hands[side].wrist_pose_world)).astype(
                np.float32
            )
            points = points_bottom if self.surface_contact else points_world
            root_quat = wrist_bottom[:, 3:] if self.surface_contact else clip.hands[side].wrist_pose_world[:, 3:]
            direct = HandRetargeterOffline(self.track, robot, spec, device=self.device)
            packed = direct.solve_numpy(points, root_quat_wxyz=root_quat)
            if not self.surface_contact:
                link_indices = np.asarray(
                    [robot.spec.link_names.index(spec.target_link_names[name]) for name in target_names],
                    dtype=np.int32,
                )
                normals = clip.contact_normals_world(side)[:, contact_indices]
                init_state = robot.state(
                    q=wp.from_numpy(packed[None, :, 7:].copy(), dtype=wp.float32, device=self.device),
                    T_world_base=wp.from_numpy(packed[None, :, :7].copy(), dtype=wp_vec7, device=self.device),
                )
                robot.forward_kinematics(init_state)
                link_poses = pose7_to_tf_mat(init_state.T_world_link.numpy()[0][:, link_indices])
                local_contacts = _compute_inward_contact_points(robot, link_indices, link_poses, normals)
                refine = HandRetargeterOffline(self.refine, robot, spec, device=self.device)
                packed = refine.solve_with_contact_numpy(
                    points,
                    clip.contacts_world(side)[:, contact_indices] + CONTACT_CLEARANCE * normals,
                    clip.hands[side].contact_mask[:, contact_indices],
                    init_state=init_state,
                    local_contact_points=local_contacts,
                )
                outputs[side] = RobotHandTrack(packed[:, 7:], packed[:, :7])
                continue

            fingertip_links = np.asarray(
                [robot.spec.link_names.index(spec.target_link_names[name]) for name in fingertip_targets],
                dtype=np.int32,
            )
            sphere_indices = []
            for link_index in fingertip_links:
                sphere_link = robot.spec.link_names[link_index].replace("_fingertip", "_elastomer")
                candidates = np.flatnonzero(
                    robot.spec.collision_spheres_link_indices == robot.spec.link_names.index(sphere_link)
                )
                surface_y = (
                    robot.spec.local_collision_sphere_centers[candidates, 1]
                    - robot.spec.collision_sphere_radii[candidates]
                )
                sphere_indices.append(int(candidates[np.argmin(surface_y)]))
            refine_spec = replace(
                self.specs[side],
                contact_target_names=fingertip_targets,
                contact_sphere_indices=tuple(sphere_indices),
            )

            contacts_world = clip.contacts_world(side)[:, fingertip_indices]
            normals_world = clip.contact_normals_world(side)[:, fingertip_indices]
            contact_mask = clip.hands[side].contact_mask[:, fingertip_indices]
            contacts_bottom = (
                np.einsum("fij,fsj->fsi", R_bottom_world, contacts_world) + T_bottom_world[:, None, :3, 3]
            ).astype(np.float32)
            normals_bottom = np.einsum("fij,fsj->fsi", R_bottom_world, normals_world).astype(np.float32)
            init_state = robot.state(
                q=wp.from_numpy(packed[None, :, 7:].copy(), dtype=wp.float32, device=self.device),
                T_world_base=wp.from_numpy(packed[None, :, :7].copy(), dtype=wp_vec7, device=self.device),
            )
            robot.forward_kinematics(init_state)
            link_poses = pose7_to_tf_mat(init_state.T_world_link.numpy()[0][:, fingertip_links])
            local_contacts = _compute_inward_contact_points(robot, fingertip_links, link_poses, normals_bottom)
            refine = HandRetargeterOffline(
                self.refine,
                robot,
                refine_spec,
                scene=scene,
                device=self.device,
                scene_indices=scene_indices,
            )
            packed = refine.solve_with_contact_numpy(
                points_bottom,
                contacts_bottom,
                contact_mask,
                init_state=init_state,
                local_contact_points=local_contacts,
                surface_contact_mask=contact_mask,
            )
            packed[:, :7] = tf_mat_to_pose7(T_world_bottom @ pose7_to_tf_mat(packed[:, :7]))
            outputs[side] = RobotHandTrack(packed[:, 7:], packed[:, :7])
        return HandObjectMotion(clip, outputs)


__all__ = [
    "ArticulatedObjectTrack",
    "CONTACT_LINK_TARGETS",
    "HandObjectClip",
    "HandObjectMotion",
    "HandObjectRetargeter",
    "HandTrack",
    "MANO16_LINKS",
    "ObjectPart",
    "RobotHandTrack",
    "derive_fingertip_contacts",
    "load_hand_object_clip",
]
