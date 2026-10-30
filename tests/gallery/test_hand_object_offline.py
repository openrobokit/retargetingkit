"""GigaHands contact refinement must preserve the tracked wrist trajectory."""

import os
import sys

import numpy as np
import pytest
import torch
import warp as wp

from robokit.helpers.hand_retargeting import HandRetargeterOffline
from robokit.robo import RobotState


if not wp.is_cuda_available():
    pytest.skip("Offline hand-object regressions require CUDA", allow_module_level=True)

from retargetingkit.assets import gigahands_hoi  # noqa: E402
from retargetingkit.gallery import hand_object_gigahands as gallery  # noqa: E402
from retargetingkit.retargeting.hand_object.offline import load_hand_object_clip  # noqa: E402


pytestmark = pytest.mark.skipif(
    os.environ.get("RETARGETINGKIT_ASSET_TESTS") != "1",
    reason="set RETARGETINGKIT_ASSET_TESTS=1 with locally prepared gallery assets",
)


class TestOfflineHandObject:
    @pytest.mark.parametrize(
        "clip_name", ("gigahands_p52_instrument_0034", "gigahands_p051_fastfood_0052_fastfood_burger")
    )
    def test_gigahands_refinement_preserves_wrist(self, clip_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
        clip = load_hand_object_clip(gigahands_hoi.root() / f"{clip_name}.npz")
        monkeypatch.setattr(gallery, "load_hand_object_clip", lambda path: clip)
        original = HandRetargeterOffline.solve_with_contact_numpy
        checked = []

        def refine(
            retargeter: HandRetargeterOffline,
            target_points: np.ndarray,
            contact_points: np.ndarray,
            contact_mask: np.ndarray,
            init_state: RobotState,
            local_contact_points: np.ndarray,
        ) -> np.ndarray:
            before = init_state.T_world_base.numpy()[0].copy()
            initial_steps = torch.from_numpy(np.rad2deg(np.diff(init_state.q.numpy()[0], axis=0)))
            torch.testing.assert_close(initial_steps, torch.zeros_like(initial_steps), rtol=0, atol=6.0)
            result = original(retargeter, target_points, contact_points, contact_mask, init_state, local_contact_points)
            torch.testing.assert_close(torch.from_numpy(result[:, :7]), torch.from_numpy(before), rtol=0, atol=1e-6)
            steps = torch.from_numpy(np.rad2deg(np.diff(result[:, 7:], axis=0)))
            torch.testing.assert_close(steps, torch.zeros_like(steps), rtol=0, atol=6.0)
            print(f"{clip_name}: max_joint_step_deg={steps.abs().max().item():.3f}")
            checked.append(True)
            return result

        monkeypatch.setattr(HandRetargeterOffline, "solve_with_contact_numpy", refine)
        monkeypatch.setattr(sys, "argv", ["gallery", "--no-viewer"])
        gallery.main()
        assert len(checked) == len(clip.hands)
