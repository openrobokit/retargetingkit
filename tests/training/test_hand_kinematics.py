import torch

from retargetingkit.retargeting.hand.presets import HANDS
from retargetingkit.training.hand.kinematics import (
    load_hand_robot,
    make_collision_residual,
    robot_keypoints,
    sample_robot_workspace,
)
from retargetingkit.training.hand.metrics import collision_loss


class TestHandKinematics:
    def test_exact_fk_shapes_and_gradients(self) -> None:
        device = torch.device("cpu")

        for name, joint_count, keypoint_count in (("sharpa", 22, 5), ("allegro", 16, 4)):
            spec = HANDS[name]
            robot = load_hand_robot(spec)
            joint_limits = torch.from_numpy(robot.spec.actuated_joint_limits).float().to(device)
            fingertip_link_indices = tuple(robot.spec.link_names.index(link) for link in spec.fingertip_links)
            joints = torch.zeros((2, joint_count), device=device, requires_grad=True)
            keypoints = robot_keypoints(robot, fingertip_link_indices, joint_limits, joints)
            keypoints.square().mean().backward()

            assert keypoints.shape == (2, keypoint_count, 3)
            assert joints.grad is not None
            assert joints.grad.abs().max() > 0

    def test_workspace_cloud_shape_and_chunk_boundaries(self) -> None:
        hand = HANDS["allegro"]
        robot = load_hand_robot(hand)
        limits = torch.from_numpy(robot.spec.actuated_joint_limits).float()
        tips = tuple(robot.spec.link_names.index(link) for link in hand.fingertip_links)

        cloud = sample_robot_workspace(robot, tips, limits, 37)

        assert cloud.shape == (37, len(tips), 3)
        assert bool(torch.isfinite(cloud).all())

    def test_workspace_sample_count_is_exact_across_chunks(self) -> None:
        hand = HANDS["allegro"]
        robot = load_hand_robot(hand)
        limits = torch.from_numpy(robot.spec.actuated_joint_limits).float()
        tips = tuple(robot.spec.link_names.index(link) for link in hand.fingertip_links)

        assert sample_robot_workspace(robot, tips, limits, 1).shape[0] == 1
        assert sample_robot_workspace(robot, tips, limits, 100).shape[0] == 100

    def test_collision_residual_and_gradient(self) -> None:
        device = torch.device("cpu")
        torch.manual_seed(0)
        spec = HANDS["sharpa"]
        robot = load_hand_robot(spec)
        joint_limits = torch.from_numpy(robot.spec.actuated_joint_limits).float().to(device)
        residual_fn = make_collision_residual(
            robot,
            joint_limits,
            spec.collision_max_active_pairs,
            0.0,
        )

        zero = torch.zeros((2, 22), device=device)
        residual = residual_fn(zero)
        assert residual.shape == (2, 50)
        torch.testing.assert_close(collision_loss(residual), residual.sum(dim=-1).mean())

        q = next(
            candidate
            for candidate in (torch.rand(1, 22, device=device) * 2 - 1 for _ in range(50))
            if float(collision_loss(residual_fn(candidate))) > 1e-4
        )
        q = q.clone().requires_grad_(True)
        loss = collision_loss(residual_fn(q))
        loss.backward()
        assert q.grad is not None
        gradient = q.grad.clone()

        epsilon = 1e-4
        finite_difference = torch.zeros_like(q)
        for index in range(22):
            delta = torch.zeros_like(q)
            delta[0, index] = epsilon
            finite_difference[0, index] = (
                collision_loss(residual_fn(q + delta)) - collision_loss(residual_fn(q - delta))
            ) / (2 * epsilon)
        torch.testing.assert_close(gradient, finite_difference, atol=1e-3, rtol=1e-2)
