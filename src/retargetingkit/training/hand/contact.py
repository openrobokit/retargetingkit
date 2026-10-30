"""YCB grasp generation and capture-time geometry for neural retargeting."""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from retargetingkit.assets import calib
from retargetingkit.assets.ycb import ycb
from retargetingkit.retargeting.hand.neural import denormalize_joints, normalize_joints, select_mano_keypoints
from retargetingkit.sources.glove.manus_conversion import manus_to_mano_keypoints
from retargetingkit.training.hand.data import GraspSpecData
from retargetingkit.training.hand.kinematics import make_collision_residual
from robokit.geom import MeshGeom, WarpScene
from robokit.robo import Robot
from robokit.xform.numpy import quaternion_to_axis_angle, random_quaternions
from robokit.xform.warp.torch_wrappers import axis_angle_to_quaternion, quaternion_to_matrix


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
N_FINGERS = len(FINGER_NAMES)


@dataclass(frozen=True)
class YcbObjectModel:
    """Centroid-centered YCB mesh and its Robokit SDF scene."""

    mesh: trimesh.Trimesh
    scene: WarpScene

    def signed_distance(
        self,
        points_world: torch.Tensor,
        translation: torch.Tensor,
        rotation: torch.Tensor,
    ) -> torch.Tensor:
        """Query signed distances at batched world points."""
        local = torch.einsum(
            "bij,bnj->bni",
            rotation.transpose(-1, -2),
            points_world - translation[:, None, :],
        )
        offsets = torch.tensor([0, local.shape[0] * local.shape[1]], dtype=torch.int32, device=local.device)
        signed_distance, _, _ = self.scene.query_sdf_torch(local.reshape(-1, 3).contiguous(), offsets)
        return signed_distance.reshape(local.shape[:2])


def load_ycb_mesh(name: str) -> trimesh.Trimesh:
    """Load a YCB mesh and center its vertices at the mesh centroid.

    >>> load_ycb_mesh("006_mustard_bottle")  # doctest: +SKIP
    """
    mesh = cast(
        trimesh.Trimesh,
        trimesh.load(ycb(name) / "google_16k" / "nontextured.ply", process=False, force="mesh"),
    )
    mesh.vertices -= mesh.centroid
    return mesh


def load_ycb_object(
    name: str,
    device: torch.device,
    sdf_voxel_size: float,
    sdf_padding: float,
) -> YcbObjectModel:
    """Load a centered YCB mesh into a cached Robokit SDF scene.

    >>> load_ycb_object("006_mustard_bottle", torch.device("cuda"), 0.002, 0.03)  # doctest: +SKIP
    """
    mesh = load_ycb_mesh(name)
    geometry = MeshGeom(
        [mesh],
        np.array([0, 1], dtype=np.int32),
        enable_sdf=True,
        sdf_voxel_size=sdf_voxel_size,
        sdf_padding=sdf_padding,
    )
    return YcbObjectModel(mesh, WarpScene(num_scenes=1, device=device).add(geometry))


@dataclass(frozen=True)
class HandContactModel:
    """Hand kinematics plus its Robokit collision-sphere metadata."""

    robot: Robot
    joint_limits: torch.Tensor
    collision_max_active_pairs: int
    sphere_centers_local: torch.Tensor
    sphere_radii: torch.Tensor
    sphere_link_indices: torch.Tensor
    pad_indices: tuple[torch.Tensor, ...]

    @classmethod
    def build(
        cls,
        robot: Robot,
        joint_limits: torch.Tensor,
        pad_link_names: tuple[str, ...],
        collision_max_active_pairs: int,
    ) -> "HandContactModel":
        """Build collision-sphere metadata on the joint-limit device.

        >>> HandContactModel.build(robot, limits, pad_links, 50)  # doctest: +SKIP
        """
        device = joint_limits.device
        robot_spec = robot.spec
        link_names = np.asarray(
            [robot_spec.link_names[int(index)] for index in robot_spec.collision_spheres_link_indices]
        )
        pad_indices = tuple(
            torch.from_numpy(np.flatnonzero(link_names == pad_link_name)).to(device) for pad_link_name in pad_link_names
        )
        assert all(index.numel() > 0 for index in pad_indices)
        return cls(
            robot=robot,
            joint_limits=joint_limits,
            collision_max_active_pairs=collision_max_active_pairs,
            sphere_centers_local=torch.from_numpy(robot_spec.local_collision_sphere_centers).to(device),
            sphere_radii=torch.from_numpy(robot_spec.local_collision_sphere_radii).to(device),
            sphere_link_indices=torch.from_numpy(robot_spec.collision_spheres_link_indices).to(device),
            pad_indices=pad_indices,
        )

    def spheres(self, normalized_joints: torch.Tensor) -> torch.Tensor:
        """Run exact FK and return world collision-sphere centers."""
        joints = denormalize_joints(normalized_joints, self.joint_limits[:, 0], self.joint_limits[:, 1])
        transforms = self.robot.forward_kinematics_via_matrix_torch(joints)
        return self.robot.transform_link_points_torch(
            transforms,
            self.sphere_centers_local,
            self.sphere_link_indices,
        )


def nearest_pad_gaps(
    spheres: torch.Tensor,
    pad_indices: tuple[torch.Tensor, ...],
    sphere_radii: torch.Tensor,
    obj: YcbObjectModel,
    translation: torch.Tensor,
    rotation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return all sphere gaps and each finger's nearest pad gap and center.

    >>> nearest_pad_gaps(spheres, pads, radii, obj, translation, rotation)  # doctest: +SKIP
    """
    gap_all = obj.signed_distance(spheres, translation, rotation) - sphere_radii[None, :]
    batch_indices = torch.arange(spheres.shape[0], device=spheres.device)
    min_gap = gap_all.new_zeros((spheres.shape[0], N_FINGERS))
    contact_points = spheres.new_zeros((spheres.shape[0], N_FINGERS, 3))
    for finger, indices in enumerate(pad_indices):
        min_gap[:, finger], nearest = gap_all[:, indices].min(dim=1)
        contact_points[:, finger] = spheres[batch_indices, indices[nearest]]
    return gap_all, min_gap, contact_points


def fit_object_translation(
    obj: YcbObjectModel,
    fingertips: torch.Tensor,
    contact_offset: torch.Tensor,
    mask: torch.Tensor,
    rotation: torch.Tensor,
    translation_init: torch.Tensor,
    keepout_points: torch.Tensor,
    keepout_radii: torch.Tensor | float,
    grasp_penetration: float,
    iterations: int,
    learning_rate: float,
) -> tuple[torch.Tensor, float, np.ndarray]:
    """Fit object translation to live fingertips while keeping the hand outside.

    >>> fit_object_translation(obj, tips, offsets, mask, rotation, initial, points, 0.0, 0.0, 10, 0.01)  # doctest: +SKIP
    """
    translation = translation_init.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([translation], lr=learning_rate)
    for _ in range(iterations):
        optimizer.zero_grad()
        targets = translation + contact_offset
        contact = (mask[:, None] * (fingertips - targets).square()).sum()
        penetration = (
            F.relu(keepout_radii - obj.signed_distance(keepout_points, translation, rotation) - grasp_penetration)
            .square()
            .sum()
        )
        (100.0 * contact + 200.0 * penetration).backward()
        optimizer.step()
    with torch.no_grad():
        targets = translation + contact_offset
        gap = float(((fingertips - targets).norm(dim=1) * mask).sum() / mask.sum())
    return translation.detach(), gap, targets.cpu().numpy().astype(np.float32)


def fabrik_reach(chain: np.ndarray, target: np.ndarray, iterations: int) -> np.ndarray:
    """Bend a fixed-length joint chain until its tip reaches `target`.

    >>> chain = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])
    >>> out = fabrik_reach(chain, np.array([1.0, 1.0, 0.0]), 20)
    >>> bool(np.allclose(out[0], [0, 0, 0])) and float(np.linalg.norm(out[-1] - [1, 1, 0])) < 1e-4
    True
    """
    base = chain[0].copy()
    lengths = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    positions = chain.astype(np.float64)
    for _ in range(iterations):
        positions[-1] = target
        for index in range(len(positions) - 2, -1, -1):
            direction = positions[index] - positions[index + 1]
            positions[index] = positions[index + 1] + direction / np.linalg.norm(direction) * lengths[index]
        positions[0] = base
        for index in range(1, len(positions)):
            direction = positions[index] - positions[index - 1]
            positions[index] = positions[index - 1] + direction / np.linalg.norm(direction) * lengths[index - 1]
    return positions.astype(chain.dtype)


def generate_grasps(
    config: dict[str, Any],
    contact_model: HandContactModel,
    human_keypoint_indices: tuple[int, ...],
    device: torch.device,
) -> GraspSpecData:
    """Optimize and filter YCB grasp specs from one generation config.

    >>> generate_grasps(config, contact_model, keypoint_indices, torch.device("cuda"))  # doctest: +SKIP
    """
    losses = config["losses"]
    filters = config["filter"]
    combos = config["combos"]
    grasps_per_combo = int(config["grasps_per_combo"])
    max_keep_per_object = int(config["max_keep_per_object"])
    iterations = int(config["iterations"])
    learning_rate = float(config["learning_rate"])
    grasp_penetration = float(config["grasp_penetration"])
    self_collision = make_collision_residual(
        contact_model.robot,
        contact_model.joint_limits,
        contact_model.collision_max_active_pairs,
        float(config["self_collision_margin"]),
    )
    self_overlap_residual = make_collision_residual(
        contact_model.robot,
        contact_model.joint_limits,
        contact_model.collision_max_active_pairs,
        0.0,
    )

    with np.load(calib.get(config["init_pose_asset"])) as raw:
        initial_pool = normalize_joints(
            torch.from_numpy(raw["grasp_q"].astype(np.float32)).to(device),
            contact_model.joint_limits[:, 0],
            contact_model.joint_limits[:, 1],
        )
    with np.load(calib.get(config["human_reference_asset"])) as raw:
        reference_tips = torch.from_numpy(
            np.stack(
                [
                    select_mano_keypoints(
                        manus_to_mano_keypoints(positions, rotations),
                        human_keypoint_indices,
                    )
                    for positions, rotations in zip(raw["right_positions"], raw["right_rotations"], strict=True)
                ]
            ).astype(np.float32)
        ).to(device)

    candidate_count = len(combos) * grasps_per_combo
    mask = torch.zeros((candidate_count, N_FINGERS), device=device)
    for combo_index, combo in enumerate(combos):
        mask[combo_index * grasps_per_combo : (combo_index + 1) * grasps_per_combo, combo] = 1.0
    nonthumb = mask.clone()
    nonthumb[:, 0] = 0.0

    parts: list[GraspSpecData] = []
    for name in config["objects"]:
        obj = load_ycb_object(
            name,
            device,
            float(config["sdf_voxel_size"]),
            float(config["sdf_padding"]),
        )
        initial_joints = initial_pool[torch.randint(initial_pool.shape[0], (candidate_count,), device=device)].clone()
        object_rotvec = torch.from_numpy(
            quaternion_to_axis_angle(random_quaternions(candidate_count)).astype(np.float32)
        ).to(device)
        with torch.no_grad():
            initial_spheres = contact_model.spheres(initial_joints)
            pad_centers = torch.stack(
                [initial_spheres[:, indices].mean(dim=1) for indices in contact_model.pad_indices],
                dim=1,
            )
            initial_translation = (pad_centers * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True)

        normalized_joints = initial_joints.clone().requires_grad_(True)
        object_translation = initial_translation.clone().requires_grad_(True)
        object_rotvec = object_rotvec.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([normalized_joints, object_translation, object_rotvec], lr=learning_rate)
        for _ in range(iterations):
            optimizer.zero_grad()
            object_rotation = quaternion_to_matrix(axis_angle_to_quaternion(object_rotvec))
            spheres = contact_model.spheres(normalized_joints)
            gap_all, min_gap, contact_points = nearest_pad_gaps(
                spheres,
                contact_model.pad_indices,
                contact_model.sphere_radii,
                obj,
                object_translation,
                object_rotation,
            )
            contact_loss = (mask * (min_gap + grasp_penetration).square()).sum()
            penetration_loss = F.relu(-gap_all - grasp_penetration).square().sum()
            contact_normal = torch.autograd.grad(
                obj.signed_distance(contact_points, object_translation, object_rotation).sum(),
                contact_points,
                create_graph=True,
            )[0]
            contact_normal = F.normalize(contact_normal, dim=-1)
            arm = contact_points - object_translation[:, None]
            net_force = (mask[..., None] * contact_normal).sum(dim=1)
            net_torque = (mask[..., None] * torch.cross(arm, contact_normal, dim=-1)).sum(dim=1)
            force_closure_loss = net_force.square().sum() + net_torque.square().sum()
            collision = self_collision(normalized_joints).sum(dim=-1).mean()
            regularization = (normalized_joints - initial_joints).square().sum()
            joint_limit = F.relu(normalized_joints.abs() - 1.0).square().sum()
            loss = (
                float(losses["contact"]) * contact_loss
                + float(losses["penetration"]) * penetration_loss
                + float(losses["self_collision"]) * collision
                + float(losses["force_closure"]) * force_closure_loss
                + float(losses["regularization"]) * regularization
                + float(losses["joint_limit"]) * joint_limit
            )
            loss.backward()
            optimizer.step()

        object_rotation = quaternion_to_matrix(axis_angle_to_quaternion(object_rotvec))
        spheres = contact_model.spheres(normalized_joints)
        gap_all, min_gap, contact_points = nearest_pad_gaps(
            spheres,
            contact_model.pad_indices,
            contact_model.sphere_radii,
            obj,
            object_translation,
            object_rotation,
        )
        directions = F.normalize(contact_points - object_translation[:, None], dim=-1)
        opposing_cosine = (nonthumb * (directions * directions[:, 0:1]).sum(dim=-1)).sum(dim=1) / nonthumb.sum(dim=1)
        max_penetration = F.relu(-gap_all).max(dim=1).values
        grip_gap = torch.where(mask > 0, min_gap, min_gap.new_full((), -1e9)).max(dim=1).values
        self_overlap = self_overlap_residual(normalized_joints).amax(dim=1)

        surface_query = contact_points.detach().requires_grad_(True)
        surface_distance = obj.signed_distance(
            surface_query,
            object_translation.detach(),
            object_rotation.detach(),
        )
        surface_normal = torch.autograd.grad(surface_distance.sum(), surface_query)[0]
        contact_surface = (surface_query - surface_distance[..., None] * F.normalize(surface_normal, dim=-1)).detach()

        contact_count = mask.sum(dim=1)
        centered_contact = (
            contact_surface
            - (mask[..., None] * contact_surface).sum(dim=1, keepdim=True) / contact_count[:, None, None]
        )
        centered_human = (
            reference_tips[None]
            - (mask[:, None, :, None] * reference_tips[None]).sum(dim=2, keepdim=True)
            / contact_count[:, None, None, None]
        )
        squared_residual = (mask[:, None, :, None] * (centered_contact[:, None] - centered_human).square()).sum(
            dim=(2, 3)
        ) / contact_count[:, None]
        reachability = squared_residual.sqrt().min(dim=1).values
        filter_results = torch.stack(
            [
                self_overlap < float(filters["max_self_overlap"]),
                grip_gap < float(filters["max_gap"]),
                max_penetration < float(filters["max_penetration"]),
                opposing_cosine < float(filters["max_opposing_cosine"]),
                reachability < float(filters["max_reachability_residual"]),
            ],
            dim=1,
        )
        passed = filter_results.all(dim=1).cpu().numpy()
        keep_per_combo = int(np.ceil(max_keep_per_object / len(combos)))
        keep = np.zeros_like(passed)
        reachability_np = reachability.detach().cpu().numpy()
        for combo_index in range(len(combos)):
            first = combo_index * grasps_per_combo
            survivors = np.flatnonzero(passed[first : first + grasps_per_combo]) + first
            keep[survivors[np.argsort(reachability_np[survivors])[:keep_per_combo]]] = True

        qpos = (
            denormalize_joints(
                normalized_joints.clamp(-1.0, 1.0),
                contact_model.joint_limits[:, 0],
                contact_model.joint_limits[:, 1],
            )
            .detach()
            .cpu()
            .numpy()
        )
        parts.append(
            GraspSpecData(
                qpos=qpos[keep].astype(np.float32),
                object_translation=object_translation.detach().cpu().numpy()[keep].astype(np.float32),
                object_rotvec=object_rotvec.detach().cpu().numpy()[keep].astype(np.float32),
                contact_mask=mask.cpu().numpy()[keep].astype(bool),
                contact_points=contact_surface.cpu().numpy()[keep].astype(np.float32),
                object_names=np.array([name] * int(keep.sum())),
            )
        )
        per_combo = [
            int(keep[index * grasps_per_combo : (index + 1) * grasps_per_combo].sum()) for index in range(len(combos))
        ]
        print(
            f"{name}: kept={int(keep.sum())}/{candidate_count} per_combo={per_combo} "
            f"filters={filter_results.sum(dim=0).tolist()} "
            f"gap={float(grip_gap[keep].detach().mean()) * 1000.0 if keep.any() else 0.0:.2f}mm "
            f"penetration={float(max_penetration[keep].detach().max()) * 1000.0 if keep.any() else 0.0:.2f}mm "
            f"reachability={float(reachability[keep].detach().mean()) * 1000.0 if keep.any() else 0.0:.2f}mm",
            flush=True,
        )
    return GraspSpecData(
        qpos=np.concatenate([part.qpos for part in parts]),
        object_translation=np.concatenate([part.object_translation for part in parts]),
        object_rotvec=np.concatenate([part.object_rotvec for part in parts]),
        contact_mask=np.concatenate([part.contact_mask for part in parts]),
        contact_points=np.concatenate([part.contact_points for part in parts]),
        object_names=np.concatenate([part.object_names for part in parts]),
    )


__all__ = [
    "FINGER_NAMES",
    "HandContactModel",
    "N_FINGERS",
    "YcbObjectModel",
    "fabrik_reach",
    "fit_object_translation",
    "generate_grasps",
    "load_ycb_mesh",
    "load_ycb_object",
    "nearest_pad_gaps",
]
