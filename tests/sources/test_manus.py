from pathlib import Path

import numpy as np
import torch

from retargetingkit.sources.glove._manus_sdk import SIDE_LEFT, SIDE_RIGHT, ManusSdkFrame
from retargetingkit.sources.glove.manus_conversion import (
    MANUS_BENCHMARK_REORIENT_ROTATION,
    hand_pose_from_glove,
    manus_to_mano_keypoints,
)
from retargetingkit.sources.glove.manus_glove import DualManusGloveAggregator, parse_glove_id


class TestManusGlovePipeline:
    def test_real_recording_replay(self) -> None:
        recording = Path(__file__).parents[1] / "fixtures/retargetingkit/data/test_fixtures/sources/manus/real.npz"
        with np.load(recording) as data:
            assert data.files == [
                "right_timestamps",
                "right_publish_time",
                "right_recv_ns",
                "right_glove_id",
                "right_side_code",
                "right_positions",
                "right_rotations",
            ]
            positions = data["right_positions"]
            rotations = data["right_rotations"]
            assert positions.shape == (3, 25, 3)
            assert rotations.shape == (3, 25, 4)
            assert positions.dtype == rotations.dtype == np.float32
            assert np.isfinite(positions).all() and np.isfinite(rotations).all()
            assert np.all(np.diff(data["right_recv_ns"]) > 0)

        replay = np.stack(
            [
                manus_to_mano_keypoints(position, rotation)
                for position, rotation in zip(positions, rotations, strict=True)
            ]
        )
        assert replay.shape == (3, 21, 3)
        assert replay.dtype == np.float32
        assert np.isfinite(replay).all()
        torch.testing.assert_close(torch.from_numpy(replay[:, 0]), torch.zeros(3, 3))

    def test_typed_sdk_frame_is_classified_and_stored(self) -> None:
        positions = np.zeros((25, 3), dtype=np.float32)
        rotations = np.zeros((25, 4), dtype=np.float32)
        rotations[:, 3] = 1.0
        aggregator = DualManusGloveAggregator()
        aggregator.update(
            ManusSdkFrame(
                glove_id=7,
                publish_time=11,
                recv_ns=13,
                side_code=SIDE_RIGHT,
                positions=positions,
                rotations=rotations,
            )
        )

        left, right = aggregator.latest()
        assert left is None
        assert right is not None
        assert right.glove_id == 7
        assert right.side == "right"
        assert right.side_code == SIDE_RIGHT
        torch.testing.assert_close(torch.from_numpy(right.positions), torch.from_numpy(positions))

    def test_explicit_glove_side_overrides_sdk_side(self) -> None:
        aggregator = DualManusGloveAggregator(left_override=42)
        aggregator.update(
            ManusSdkFrame(
                glove_id=42,
                publish_time=0,
                recv_ns=0,
                side_code=SIDE_RIGHT,
                positions=np.zeros((25, 3), dtype=np.float32),
                rotations=np.zeros((25, 4), dtype=np.float32),
            )
        )

        left, right = aggregator.latest()
        assert left is not None
        assert left.side_code == SIDE_LEFT
        assert right is None

    def test_manus_to_mano_preserves_coordinate_mapping(self) -> None:
        positions = np.zeros((25, 3), dtype=np.float32)
        positions[1] = [1.0, 0.0, 0.0]
        rotations = np.zeros((25, 4), dtype=np.float32)
        rotations[:, 3] = 1.0

        mano = manus_to_mano_keypoints(positions, rotations)

        assert mano.shape == (21, 3)
        torch.testing.assert_close(
            torch.from_numpy(mano[1]),
            torch.tensor([0.0, 0.0, 1.0]),
        )

    def test_calibration_applies_benchmark_reorientation(self) -> None:
        positions = np.zeros((25, 3), dtype=np.float32)
        positions[6] = [1.0, 0.0, 0.0]
        rotations = np.zeros((25, 4), dtype=np.float32)
        rotations[:, 3] = 1.0
        calibration = {
            "thumb_rotation": np.zeros(3, dtype=np.float32),
            "thumb_translation": np.zeros(3, dtype=np.float32),
            "finger_scales": np.ones(5, dtype=np.float32),
        }

        original = hand_pose_from_glove(positions, rotations).positions
        reoriented = hand_pose_from_glove(positions, rotations, calibration).positions

        torch.testing.assert_close(
            torch.from_numpy(reoriented),
            torch.from_numpy(original @ MANUS_BENCHMARK_REORIENT_ROTATION.T),
        )

    def test_parse_glove_id(self) -> None:
        assert parse_glove_id("42") == 42
        assert parse_glove_id("0x2A") == 42
        assert parse_glove_id(None) is None
