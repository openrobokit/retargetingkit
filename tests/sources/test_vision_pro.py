import pickle
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from retargetingkit.sources.vr.vision_pro import (
    VISION_PRO_25_TO_MANO_21,
    VISION_PRO_JOINT_NAMES,
    VisionPro,
    VisionProArmConfig,
    VisionProArmHandConfig,
    VisionProHandConfig,
    vision_pro_to_mano_keypoints,
    vision_pro_wrist_matrix,
)


@pytest.fixture(scope="module")
def avp_pkl():
    path = Path(__file__).parents[1] / "fixtures/retargetingkit/data/test_fixtures/sources/vision_pro/real.pkl"
    with path.open("rb") as file:
        frames = pickle.load(file)
    return str(path), frames


class TestVisionPro:
    def test_get_head_shape(self, avp_pkl):
        path, _ = avp_pkl
        vp = VisionPro(recording_path=path)
        assert vp.get_head().shape == (1, 4, 4)

    def test_get_wrist_shape(self, avp_pkl):
        path, _ = avp_pkl
        vp = VisionPro(recording_path=path)
        assert vp.get_left_wrist().shape == (1, 4, 4)
        assert vp.get_right_wrist().shape == (1, 4, 4)

    def test_get_fingers_shape(self, avp_pkl):
        path, _ = avp_pkl
        vp = VisionPro(recording_path=path)
        assert vp.get_left_fingers().shape == (25, 4, 4)
        assert vp.get_right_fingers().shape == (25, 4, 4)

    def test_get_pinch_distance(self, avp_pkl):
        path, frames = avp_pkl
        vp = VisionPro(recording_path=path)
        assert isinstance(vp.get_left_pinch_distance(), float)
        assert isinstance(vp.get_right_pinch_distance(), float)

    def test_get_wrist_roll(self, avp_pkl):
        path, _ = avp_pkl
        vp = VisionPro(recording_path=path)
        assert isinstance(vp.get_left_wrist_roll(), float)
        assert isinstance(vp.get_right_wrist_roll(), float)

    def test_get_all_keys(self, avp_pkl):
        path, _ = avp_pkl
        vp = VisionPro(recording_path=path)
        data = vp.get_all()
        expected = {
            "head",
            "left_wrist",
            "right_wrist",
            "left_fingers",
            "right_fingers",
            "left_arm",
            "right_arm",
            "left_pinch_distance",
            "right_pinch_distance",
            "left_wrist_roll",
            "right_wrist_roll",
        }
        assert expected <= set(data.keys())
        assert data["right_fingers"].dtype == np.float64
        assert np.isfinite(data["right_fingers"]).all()
        assert len(VISION_PRO_JOINT_NAMES) == data["right_fingers"].shape[0]

    def test_playback_clamps_to_last(self, avp_pkl):
        path, frames = avp_pkl
        vp = VisionPro(recording_path=path, recording_fps=100000)
        time.sleep(0.01)
        torch.testing.assert_close(torch.from_numpy(vp.get_head()), torch.from_numpy(frames[-1]["head"]))


class TestVisionProSource:
    def test_playback_hand_frame(self, avp_pkl) -> None:
        path, _ = avp_pkl
        source = VisionProHandConfig(recording_path=path).build()
        frame = source.get_frame()
        assert frame is not None
        assert set(frame.hands) == {"left", "right"}
        assert frame.hands["right"].positions.shape == (21, 3)
        source.close()

    def test_playback_arm_frame(self, avp_pkl) -> None:
        path, _ = avp_pkl
        source = VisionProArmConfig(recording_path=path).build()
        frame = source.get_frame()
        assert frame is not None
        assert set(frame.wrists) == {"left", "right"}
        assert frame.wrists["right"].shape == (7,)
        source.close()

    def test_playback_arm_hand_frame(self, avp_pkl) -> None:
        path, _ = avp_pkl
        source = VisionProArmHandConfig(recording_path=path).build()
        frame = source.get_frame()
        assert frame is not None
        assert set(frame.arm.wrists) == set(frame.hand.hands) == {"left", "right"}
        source.close()


class TestVisionProConversion:
    def test_finger_poses_to_mano21(self) -> None:
        poses = np.tile(np.eye(4, dtype=np.float32), (25, 1, 1))
        poses[:, :3, 3] = np.arange(75, dtype=np.float32).reshape(25, 3)
        rotation = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float32)
        expected = poses[list(VISION_PRO_25_TO_MANO_21), :3, 3] @ rotation
        expected -= expected[:1]

        actual = vision_pro_to_mano_keypoints(poses, "right")

        torch.testing.assert_close(torch.from_numpy(actual), torch.from_numpy(expected))

    def test_wrist_axes_and_translation(self) -> None:
        wrist = np.eye(4, dtype=np.float32)
        wrist[:3, 3] = (1.0, 2.0, 3.0)
        expected = np.array(
            [
                [1.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, -1.0, -1.0],
                [0.0, 1.0, 0.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        actual = vision_pro_wrist_matrix(wrist, "left")

        torch.testing.assert_close(torch.from_numpy(actual), torch.from_numpy(expected))
