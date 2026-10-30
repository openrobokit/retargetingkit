"""Warp-backed exact FK, self-collision, and robot workspace sampling."""

from collections.abc import Callable

import torch
import warp as wp

from retargetingkit.retargeting.hand.neural import denormalize_joints, normalize_joints
from retargetingkit.retargeting.hand.presets import HandPreset
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo import Robot
from robokit.terms import SelfCollisionTask


def load_hand_robot(spec: HandPreset) -> Robot:
    """Load the Robokit robot described by `spec`.

    >>> load_hand_robot(HANDS["sharpa"])  # doctest: +SKIP
    """
    wp.init()
    spheres_path = spec.collision_spheres_path
    robot = Robot.load(
        spec.urdf_path,
        load_meshes=False,
        mesh_dir=None,
        load_collision_spheres=spheres_path is not None,
        collision_spheres_path=None if spheres_path is None else str(spheres_path),
        self_collision_ignore_path=spec.collision_ignore_path,
    )
    assert tuple(robot.spec.actuated_joint_names) == spec.joint_names
    return robot


def robot_keypoints(
    robot: Robot,
    fingertip_link_indices: tuple[int, ...],
    joint_limits: torch.Tensor,
    normalized_joints: torch.Tensor,
) -> torch.Tensor:
    """Return exact fingertip positions for normalized joints.

    >>> robot_keypoints(robot, tip_indices, limits, joints)  # doctest: +SKIP
    """
    joints = denormalize_joints(normalized_joints, joint_limits[:, 0], joint_limits[:, 1])
    transforms = robot.forward_kinematics_via_matrix_torch(joints)
    return transforms[..., fingertip_link_indices, :3, 3]


def sample_robot_workspace(
    robot: Robot,
    fingertip_link_indices: tuple[int, ...],
    joint_limits: torch.Tensor,
    sample_count: int,
) -> torch.Tensor:
    """Uniformly sample joint limits and return an `[N, K, 3]` CPU cloud.

    >>> sample_robot_workspace(robot, tip_indices, limits, 100).shape  # doctest: +SKIP
    """
    batches: list[torch.Tensor] = []
    joint_lower = joint_limits[:, 0]
    joint_upper = joint_limits[:, 1]
    with torch.no_grad():
        for start in range(0, sample_count, 100_000):
            size = min(100_000, sample_count - start)
            joints = torch.rand((size, len(joint_limits)), device=joint_limits.device)
            joints = joints * (joint_upper - joint_lower) + joint_lower
            normalized = normalize_joints(joints, joint_lower, joint_upper)
            batches.append(robot_keypoints(robot, fingertip_link_indices, joint_limits, normalized).cpu())
    return torch.cat(batches)


def make_collision_residual(
    robot: Robot,
    joint_limits: torch.Tensor,
    max_active_pairs: int,
    margin_m: float,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a differentiable per-sample self-collision residual.

    >>> make_collision_residual(robot, limits, 50, 0.0)  # doctest: +SKIP
    """
    task = SelfCollisionTask(
        robot=robot,
        weight=1.0,
        margin=margin_m,
        representation="sphere",
        max_active_pairs=max_active_pairs,
    )

    def residual(normalized_joints: torch.Tensor) -> torch.Tensor:
        joints = denormalize_joints(normalized_joints, joint_limits[:, 0], joint_limits[:, 1])
        return _SelfCollisionResidual.apply(joints, task, robot)

    return residual


class _SelfCollisionResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, joints: torch.Tensor, task: SelfCollisionTask, robot: Robot) -> torch.Tensor:
        state = robot.state(wp.from_torch(joints.detach().contiguous(), dtype=wp.float32))
        values = VarValues({VarKey.ROBOT: state})
        residual = wp.to_torch(task.compute_weighted_residual(values)).clone()
        ctx.jacobian = wp.to_torch(task.compute_weighted_jacobian_analytic(values)).clone()
        return residual

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (grad_output,) = grad_outputs
        return torch.einsum("bk,bkd->bd", grad_output, ctx.jacobian), None, None


__all__ = [
    "load_hand_robot",
    "make_collision_residual",
    "robot_keypoints",
    "sample_robot_workspace",
]
