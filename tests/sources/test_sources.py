from pathlib import Path

import numpy as np
import pytest
import torch

from retargetingkit.assets import sample_bvh
from retargetingkit.sources.misc.human_frame_file import BodyFileSource
from retargetingkit.sources.vr.pico import PICO_JOINT_NAMES, pico_pose_array_to_transforms


SAMPLE_BVH = sample_bvh.EXAMPLE_BVH


@pytest.fixture(scope="module")
def bvh_source() -> BodyFileSource:
    return BodyFileSource(SAMPLE_BVH, BVH_GOLDEN_JOINTS)


@pytest.fixture(scope="module")
def sample_pico_npz() -> Path:
    return Path(__file__).parents[1] / "fixtures/retargetingkit/data/test_fixtures/sources/pico/real.npz"


class TestBvhFileSource:
    def test_loads_frames(self, bvh_source: BodyFileSource) -> None:
        assert bvh_source.num_frames > 0
        assert bvh_source.fps > 0

    def test_get_frame_returns_dense_transforms(self, bvh_source: BodyFileSource) -> None:
        frame = bvh_source.get_frame()
        assert frame is not None
        assert frame.transforms.shape == (len(BVH_GOLDEN_JOINTS), 7)
        assert frame.transforms.dtype == np.float32
        assert frame.joint_names == BVH_GOLDEN_JOINTS

    def test_hold_returns_frame_zero(self, bvh_source: BodyFileSource) -> None:
        f1 = bvh_source.get_frame()
        f2 = bvh_source.get_frame()
        assert f1 is not None and f2 is not None
        torch.testing.assert_close(torch.from_numpy(f1.transforms), torch.from_numpy(f2.transforms), rtol=0, atol=0)

    def test_reset(self, bvh_source: BodyFileSource) -> None:
        bvh_source.reset()
        assert bvh_source.frame_idx == 0

    def test_ankle_as_foot_target(self) -> None:
        names = ("left_ankle", "left_foot")
        src = BodyFileSource(SAMPLE_BVH, names, use_ankle_as_foot_target=True)
        frame = src.get_frame()
        assert frame is not None
        torch.testing.assert_close(
            torch.from_numpy(frame.transforms[names.index("left_foot")]),
            torch.from_numpy(frame.transforms[names.index("left_ankle")]),
            rtol=1e-7,
            atol=0,
        )

    def test_human_height(self) -> None:
        src = BodyFileSource(SAMPLE_BVH, ("pelvis",), human_height=1.80)
        assert src.human_height == 1.80

    def test_parsed_frames_match_golden(self, bvh_source: BodyFileSource) -> None:
        # Snapshot of parsed positions for 4 frames spanning the take; tripwire for the dense loader.
        frames = bvh_source._frames
        indices = np.linspace(0, len(frames) - 1, 4).astype(int)
        out = np.stack([frames[i, :, :3].reshape(-1) for i in indices])
        torch.testing.assert_close(torch.from_numpy(out), torch.from_numpy(BVH_FRAME_GOLDEN), rtol=1e-4, atol=1e-5)


BVH_GOLDEN_JOINTS = ("pelvis", "head", "left_wrist", "right_wrist", "left_ankle", "right_ankle")

# fmt: off
BVH_FRAME_GOLDEN = np.array(
    [
        (0.000000, 0.000000, 0.971200, 0.000000, 0.000000, 1.636200, 0.730000, 0.000000, 1.493200, -0.730000, 0.000000, 1.493200, 0.110000, 0.000000, 0.100000, -0.110000, 0.000000, 0.100000),  # noqa: E501
        (-0.068441, -0.148428, 0.969829, -0.086105, -0.254180, 1.621386, 0.130063, -0.251731, 0.932266, -0.351318, -0.250503, 0.946766, 0.079069, -0.111491, 0.102973, -0.184573, -0.089326, 0.101221),  # noqa: E501
        (-0.050862, 0.078751, 0.932286, -0.252104, -0.480639, 1.216969, -0.127689, -0.814741, 0.834849, -0.309819, -0.936846, 0.827612, -0.014007, -0.186823, 0.149543, -0.100727, -0.105157, 0.052649),  # noqa: E501
        (-0.032725, -0.211242, 0.975082, -0.132708, -0.654121, 1.452342, 0.127211, -0.381636, 0.936157, -0.270198, -0.920807, 0.869986, 0.115951, -0.291946, 0.146306, -0.171349, -0.228177, 0.089493),  # noqa: E501
    ],
    dtype=np.float32,
)
# fmt: on


class TestPicoFrameConversion:
    def _make_identity_poses(self) -> np.ndarray:
        poses = np.zeros((24, 7), dtype=np.float32)
        poses[:, 6] = 1.0
        return poses

    def test_output_has_24_joints(self) -> None:
        frame = pico_pose_array_to_transforms(self._make_identity_poses(), PICO_JOINT_NAMES)
        assert frame.shape == (24, 7)

    def test_pelvis_exists(self) -> None:
        frame = pico_pose_array_to_transforms(self._make_identity_poses(), ("pelvis",))
        assert frame.shape == (1, 7)

    def test_output_is_float32(self) -> None:
        frame = pico_pose_array_to_transforms(self._make_identity_poses(), PICO_JOINT_NAMES)
        assert frame.dtype == np.float32

    def test_quaternions_are_normalized(self) -> None:
        poses = self._make_identity_poses()
        poses[0, 3:] = [0.1, 0.2, 0.3, 0.9]
        frame = pico_pose_array_to_transforms(poses, PICO_JOINT_NAMES)
        for transform in frame:
            norm = float(np.linalg.norm(transform[3:]))
            assert abs(norm - 1.0) < 1e-5, f"quat norm={norm}"

    def test_y_up_to_z_up_conversion(self) -> None:
        poses = self._make_identity_poses()
        poses[0, :3] = [0.0, 1.0, 0.0]
        frame = pico_pose_array_to_transforms(poses, ("pelvis",))
        pos = frame[0, :3]
        assert abs(pos[2] - 1.0) < 1e-5, f"expected z≈1.0, got {pos[2]}"
        assert abs(pos[0]) < 1e-5
        assert abs(pos[1]) < 1e-5

    def test_ankle_as_foot_target(self) -> None:
        poses = self._make_identity_poses()
        poses[7, :3] = [0.1, 0.0, 0.0]  # left_ankle
        poses[10, :3] = [0.5, 0.0, 0.0]  # left_foot
        names = ("left_ankle", "left_foot")
        frame = pico_pose_array_to_transforms(poses, names, use_ankle_as_foot_target=True)
        torch.testing.assert_close(
            torch.from_numpy(frame[names.index("left_foot")]),
            torch.from_numpy(frame[names.index("left_ankle")]),
            rtol=1e-7,
            atol=0,
        )

    def test_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected.*24.*7"):
            pico_pose_array_to_transforms(np.zeros((10, 7), dtype=np.float32), PICO_JOINT_NAMES)

    def test_known_joint_names(self) -> None:
        expected = {
            "pelvis",
            "left_hip",
            "right_hip",
            "spine1",
            "left_knee",
            "right_knee",
            "spine2",
            "left_ankle",
            "right_ankle",
            "spine3",
            "left_foot",
            "right_foot",
            "neck",
            "left_collar",
            "right_collar",
            "head",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hand",
            "right_hand",
        }
        assert set(PICO_JOINT_NAMES) == expected


class TestPicoFileSource:
    def test_real_recording_contract(self, sample_pico_npz: Path) -> None:
        data = np.load(sample_pico_npz)
        poses = data["poses"]
        timestamps = data["timestamps_s"]
        assert poses.shape == (100, 24, 7)
        assert poses.dtype == np.float32
        assert np.isfinite(poses).all()
        assert timestamps.shape == (100,)
        assert timestamps.dtype == np.float64
        assert np.diff(timestamps).min() > 0
        quaternion_norms = torch.linalg.vector_norm(torch.from_numpy(poses[..., 3:]), dim=-1)
        torch.testing.assert_close(quaternion_norms, torch.ones_like(quaternion_norms), rtol=1e-5, atol=1e-5)

    def test_loads_recording(self, sample_pico_npz: Path) -> None:
        src = BodyFileSource(sample_pico_npz, PICO_JOINT_NAMES)
        assert src.num_frames > 0
        assert src.fps > 0

    def test_get_frame_returns_dense_transforms(self, sample_pico_npz: Path) -> None:
        src = BodyFileSource(sample_pico_npz, PICO_JOINT_NAMES)
        frame = src.get_frame()
        assert frame is not None
        assert frame.transforms.shape == (24, 7)

    def test_hold_and_release(self, sample_pico_npz: Path) -> None:
        src = BodyFileSource(sample_pico_npz, PICO_JOINT_NAMES)
        f1 = src.get_frame()
        f2 = src.get_frame()
        assert f1 is not None and f2 is not None
        torch.testing.assert_close(torch.from_numpy(f1.transforms), torch.from_numpy(f2.transforms), rtol=0, atol=0)

        src.release()
        f3 = src.get_frame()
        assert f3 is not None

    def test_human_height(self, sample_pico_npz: Path) -> None:
        src = BodyFileSource(sample_pico_npz, PICO_JOINT_NAMES, human_height=1.65)
        assert src.human_height == 1.65
