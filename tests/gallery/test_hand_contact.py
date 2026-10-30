"""Real trajectories covering arm initialization and contact refinement."""

import os

import numpy as np
import pytest
import torch
import warp as wp

from robokit.helpers.hand_retargeting import HandRetargeterOffline
from robokit.robo import Robot, RobotState


if not wp.is_cuda_available():
    pytest.skip("Offline hand-contact regressions require CUDA", allow_module_level=True)

from retargetingkit.gallery.hand_contact import BASE_POSE, ROBOTS, _load_config, _solve  # noqa: E402


pytestmark = pytest.mark.skipif(
    os.environ.get("RETARGETINGKIT_ASSET_TESTS") != "1",
    reason="set RETARGETINGKIT_ASSET_TESTS=1 with locally prepared gallery assets",
)


class TestOfflineHandContact:
    @pytest.mark.parametrize("robot_name", ROBOTS)
    @pytest.mark.parametrize("capture", (0, 1, 10))
    @pytest.mark.parametrize("contact", (False, True))
    def test_smooth_trajectory_and_fixed_arm_during_refinement(
        self, robot_name: str, capture: int, contact: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = HandRetargeterOffline.solve_with_contact_numpy
        refinements = []

        def refine(
            retargeter: HandRetargeterOffline,
            target_points: np.ndarray,
            contact_points: np.ndarray,
            contact_mask: np.ndarray,
            init_state: RobotState,
            local_contact_points: np.ndarray,
        ) -> np.ndarray:
            before_q = init_state.q.numpy()[0].copy()
            before_base = init_state.T_world_base.numpy()[0].copy()
            result = original(retargeter, target_points, contact_points, contact_mask, init_state, local_contact_points)
            spec = retargeter.robot.spec
            arm = [
                i
                for i, name in enumerate(spec.actuated_joint_names)
                if name not in retargeter.config.active_joint_names
            ]
            torch.testing.assert_close(
                torch.from_numpy(result[:, 7:][:, arm]), torch.from_numpy(before_q[:, arm]), rtol=0, atol=1e-6
            )
            torch.testing.assert_close(
                torch.from_numpy(result[:, :7]), torch.from_numpy(before_base), rtol=0, atol=1e-6
            )
            torch.testing.assert_close(
                torch.from_numpy(result[:2, 7:]), torch.from_numpy(before_q[:2]), rtol=0, atol=1e-6
            )
            refinements.append(True)
            return result

        monkeypatch.setattr(HandRetargeterOffline, "solve_with_contact_numpy", refine)
        result = _solve(robot_name, capture, contact, "cuda:0", 0)
        assert len(refinements) == int(contact)
        q = result.joints
        steps = torch.from_numpy(np.rad2deg(np.diff(q, axis=0)))
        print(f"{robot_name} capture={capture} contact={contact}: max_step_deg={steps.abs().max().item():.3f}")
        torch.testing.assert_close(steps, torch.zeros_like(steps), rtol=0, atol=10.0)
        torch.testing.assert_close(
            torch.from_numpy(result.T_world_base), torch.from_numpy(np.tile(BASE_POSE, (len(q), 1))), rtol=0, atol=1e-6
        )
        asset, _, _ = _load_config(robot_name)
        lower, upper = Robot.load(asset.URDF_PATH).spec.actuated_joint_limits.astype(np.float32).T
        torch.testing.assert_close(torch.from_numpy(q), torch.from_numpy(np.clip(q, lower, upper)), rtol=0, atol=1e-6)
