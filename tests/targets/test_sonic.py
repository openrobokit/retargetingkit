import numpy as np
import torch

from retargetingkit.targets.humanoids.sonic import (
    HEADER_SIZE,
    NUM_JOINTS,
    PROTOCOL_JOINT_NAMES,
    PROTOCOL_TO_URDF,
    TOPIC,
    URDF_JOINT_NAMES,
    URDF_TO_PROTOCOL,
    build_command_message,
    build_header,
    pack_v1_message,
    reorder_protocol_to_urdf,
    reorder_urdf_to_protocol,
    unpack_v1_message,
)


class TestJointMapping:
    def test_mapping_lengths(self) -> None:
        assert len(URDF_TO_PROTOCOL) == NUM_JOINTS
        assert len(PROTOCOL_TO_URDF) == NUM_JOINTS
        assert len(URDF_JOINT_NAMES) == NUM_JOINTS
        assert len(PROTOCOL_JOINT_NAMES) == NUM_JOINTS

    def test_urdf_to_protocol_is_permutation(self) -> None:
        assert sorted(URDF_TO_PROTOCOL) == list(range(NUM_JOINTS))

    def test_protocol_to_urdf_is_permutation(self) -> None:
        assert sorted(PROTOCOL_TO_URDF) == list(range(NUM_JOINTS))

    def test_roundtrip_identity(self) -> None:
        x = np.arange(NUM_JOINTS, dtype=np.float32)
        actual = reorder_protocol_to_urdf(reorder_urdf_to_protocol(x))
        torch.testing.assert_close(torch.from_numpy(actual), torch.from_numpy(x), rtol=0, atol=0)

    def test_roundtrip_inverse(self) -> None:
        x = np.arange(NUM_JOINTS, dtype=np.float32)
        actual = reorder_urdf_to_protocol(reorder_protocol_to_urdf(x))
        torch.testing.assert_close(torch.from_numpy(actual), torch.from_numpy(x), rtol=0, atol=0)

    def test_batched(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.standard_normal((5, NUM_JOINTS)).astype(np.float32)
        rt = reorder_protocol_to_urdf(reorder_urdf_to_protocol(x))
        torch.testing.assert_close(torch.from_numpy(rt), torch.from_numpy(x), rtol=0, atol=0)

    def test_not_self_inverse(self) -> None:
        x = np.arange(NUM_JOINTS, dtype=np.float32)
        once = reorder_urdf_to_protocol(x)
        twice = reorder_urdf_to_protocol(once)
        assert not np.array_equal(twice, x)


class TestSonicProtocol:
    def test_header_size(self) -> None:
        h = build_header([{"name": "x", "dtype": "f32", "shape": [1]}])
        assert len(h) == HEADER_SIZE

    def test_pack_unpack_roundtrip(self) -> None:
        rng = np.random.default_rng(42)
        n = 3
        joint_pos = rng.standard_normal((n, NUM_JOINTS)).astype(np.float32)
        joint_vel = rng.standard_normal((n, NUM_JOINTS)).astype(np.float32)
        body_quat = rng.standard_normal((n, 4)).astype(np.float64)
        frame_idx = np.arange(n, dtype=np.int64)

        msg = pack_v1_message(joint_pos, joint_vel, body_quat, frame_idx)
        assert msg[: len(TOPIC)] == TOPIC

        fields = unpack_v1_message(msg)
        for name, expected in (
            ("joint_pos", joint_pos),
            ("joint_vel", joint_vel),
            ("body_quat", body_quat),
            ("frame_index", frame_idx),
        ):
            torch.testing.assert_close(torch.from_numpy(fields[name]), torch.from_numpy(expected), rtol=0, atol=0)

    def test_pack_with_hand_joints(self) -> None:
        n = 1
        msg = pack_v1_message(
            joint_pos=np.zeros((n, NUM_JOINTS), dtype=np.float32),
            joint_vel=np.zeros((n, NUM_JOINTS), dtype=np.float32),
            body_quat=np.array([[1, 0, 0, 0]], dtype=np.float64),
            frame_index=np.array([0], dtype=np.int64),
            left_hand_joints=np.zeros(7, dtype=np.float32),
            right_hand_joints=np.ones(7, dtype=np.float32),
        )
        fields = unpack_v1_message(msg)
        assert "left_hand_joints" in fields
        assert "right_hand_joints" in fields
        torch.testing.assert_close(torch.from_numpy(fields["right_hand_joints"]), torch.ones(7), rtol=0, atol=0)

    def test_catch_up_flag(self) -> None:
        n = 1
        msg = pack_v1_message(
            joint_pos=np.zeros((n, NUM_JOINTS), dtype=np.float32),
            joint_vel=np.zeros((n, NUM_JOINTS), dtype=np.float32),
            body_quat=np.array([[1, 0, 0, 0]], dtype=np.float64),
            frame_index=np.array([0], dtype=np.int64),
            catch_up=False,
        )
        fields = unpack_v1_message(msg)
        assert fields["catch_up"][0] == 0

    def test_command_message(self) -> None:
        msg = build_command_message(start=True, planner=False)
        assert msg[:7] == b"command"
