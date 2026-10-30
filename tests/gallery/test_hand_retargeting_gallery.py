"""The DexYCB example shares the gallery's preset and streaming behavior."""

import argparse
import importlib
import os
import sys

import numpy as np
import pytest
import torch
import warp as wp

from robokit.gallery.hand_retargeting import FLOATING, PRESETS, WITH_WRIST, Gallery
from robokit.helpers.hand_retargeting import HandRetargeterOnline
from robokit.robo import RobotState


pytestmark = pytest.mark.skipif(
    os.environ.get("RETARGETINGKIT_ASSET_TESTS") != "1",
    reason="set RETARGETINGKIT_ASSET_TESTS=1 with locally prepared gallery assets",
)


class TestHandRetargetingGallery:
    @pytest.mark.skipif(not wp.is_cuda_available(), reason="DexYCB example requires CUDA")
    @pytest.mark.parametrize("ee_name", tuple(PRESETS))
    def test_example_matches_gallery(
        self, ee_name: str, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
    ) -> None:
        preset = PRESETS[ee_name]
        assert preset.online is not None
        assert preset.online_wrist is not None
        results = []
        solve = HandRetargeterOnline.solve_numpy

        def record(
            solver: HandRetargeterOnline, target: np.ndarray, init_state: RobotState | None = None
        ) -> np.ndarray:
            packed = solve(solver, target, init_state)
            results.append(packed.copy())
            return packed

        monkeypatch.setattr(HandRetargeterOnline, "solve_numpy", record)
        example = importlib.import_module("retargetingkit.examples.hand_optimization.06_dexycb_online")
        monkeypatch.setattr(sys, "argv", ["example", "--ee", ee_name, "--port", "0", "--loops", "3", "--fps", "100000"])
        example.main()
        example_results = np.concatenate(results)
        results.clear()
        gallery = Gallery(
            argparse.Namespace(
                mode="online",
                ee=ee_name,
                base=FLOATING,
                motion=WITH_WRIST,
                capture_index=0,
                mano_root="",
                port=0,
            )
        )
        request.addfinalizer(gallery.server.stop)
        gallery.playing.value = False
        for index in range(1, gallery.num_frames * 3):
            gallery.update(index % gallery.num_frames)
        gallery_results = np.concatenate(results)
        expected, actual = torch.from_numpy(example_results), torch.from_numpy(gallery_results)
        # Collision-pair selection runs in parallel on the GPU; compare each quantity in its own units.
        torch.testing.assert_close(
            expected[:, :3],
            actual[:, :3],
            atol=1e-4 if preset.online_wrist.task("self_collision").weight else 0,
            rtol=0,
        )
        torch.testing.assert_close(
            expected[:, 3:7],
            actual[:, 3:7],
            atol=1e-3 if preset.online_wrist.task("self_collision").weight else 0,
            rtol=0,
        )
        torch.testing.assert_close(
            expected[:, 7:],
            actual[:, 7:],
            atol=float(np.deg2rad(0.25)) if preset.online_wrist.task("self_collision").weight else 0,
            rtol=0,
        )
        print(
            f"{ee_name} example/gallery: solves={len(example_results)} "
            f"max_base_difference_mm={np.linalg.norm(example_results[:, :3] - gallery_results[:, :3], axis=-1).max() * 1000:.5f} "
            f"max_joint_difference_deg={np.rad2deg(np.abs(example_results[:, 7:] - gallery_results[:, 7:])).max():.5f}"
        )
