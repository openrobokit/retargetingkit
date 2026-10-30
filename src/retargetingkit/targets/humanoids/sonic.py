"""SONIC v1 wire format and G1 joint order mapping; payload fields follow header order, C-contiguous and little-endian.

Wire format: `[topic_bytes][1280-byte JSON header (null-padded)][binary payload]`

Header JSON schema::

    {"v": 1, "endian": "le", "count": N,
     "fields": [{"name": ..., "dtype": ..., "shape": [...]}, ...]}

Joint ordering:
  - **URDF order**: sequential per body part (robokit output).
    left_leg(6), right_leg(6), waist(3), left_arm(7), right_arm(7).
  - **IsaacLab/protocol order**: interleaved by DOF type (SONIC wire).
  - Mapping arrays from `policy_parameters.hpp` in gear_sonic_deploy.
  - These are NOT self-inverse — applying twice does NOT return the original.
"""

import json
import time
from dataclasses import dataclass

import numpy as np
import zmq

from retargetingkit.targets.commands import BasePoseQPosCommand, Command
from retargetingkit.targets.target import Target, TargetConfig


# ── Wire format constants ────────────────────────────────────────────────────

HEADER_SIZE = 1280
TOPIC = b"pose"
COMMAND_TOPIC = b"command"

_NP_TO_WIRE: dict[str, str] = {
    "float32": "f32",
    "float64": "f64",
    "int32": "i32",
    "int64": "i64",
    "uint8": "u8",
    "bool": "bool",
}

_WIRE_TO_NP: dict[str, np.dtype] = {
    "f32": np.dtype(np.float32),
    "f64": np.dtype(np.float64),
    "i32": np.dtype(np.int32),
    "i64": np.dtype(np.int64),
    "u8": np.dtype(np.uint8),
    "bool": np.dtype(np.bool_),
}

# ── G1 29-DOF joint names ────────────────────────────────────────────────────

NUM_JOINTS = 29

URDF_JOINT_NAMES: list[str] = [
    "left_hip_pitch",  # 0
    "left_hip_roll",  # 1
    "left_hip_yaw",  # 2
    "left_knee",  # 3
    "left_ankle_pitch",  # 4
    "left_ankle_roll",  # 5
    "right_hip_pitch",  # 6
    "right_hip_roll",  # 7
    "right_hip_yaw",  # 8
    "right_knee",  # 9
    "right_ankle_pitch",  # 10
    "right_ankle_roll",  # 11
    "waist_yaw",  # 12
    "waist_roll",  # 13
    "waist_pitch",  # 14
    "left_shoulder_pitch",  # 15
    "left_shoulder_roll",  # 16
    "left_shoulder_yaw",  # 17
    "left_elbow",  # 18
    "left_wrist_roll",  # 19
    "left_wrist_pitch",  # 20
    "left_wrist_yaw",  # 21
    "right_shoulder_pitch",  # 22
    "right_shoulder_roll",  # 23
    "right_shoulder_yaw",  # 24
    "right_elbow",  # 25
    "right_wrist_roll",  # 26
    "right_wrist_pitch",  # 27
    "right_wrist_yaw",  # 28
]

# isaaclab[i] = mujoco[URDF_TO_PROTOCOL[i]]
URDF_TO_PROTOCOL: list[int] = [
    0,
    6,
    12,  # hip_pitch: left, right, waist_yaw
    1,
    7,
    13,  # hip_roll:  left, right, waist_roll
    2,
    8,
    14,  # hip_yaw:   left, right, waist_pitch
    3,
    9,  # knee:      left, right
    15,
    22,  # shoulder_pitch: left, right
    4,
    10,  # ankle_pitch: left, right
    16,
    23,  # shoulder_roll: left, right
    5,
    11,  # ankle_roll: left, right
    17,
    24,  # shoulder_yaw: left, right
    18,
    25,  # elbow: left, right
    19,
    26,  # wrist_roll: left, right
    20,
    27,  # wrist_pitch: left, right
    21,
    28,  # wrist_yaw: left, right
]

# mujoco[i] = isaaclab[PROTOCOL_TO_URDF[i]]
PROTOCOL_TO_URDF: list[int] = [
    0,
    3,
    6,
    9,
    13,
    17,  # left leg
    1,
    4,
    7,
    10,
    14,
    18,  # right leg
    2,
    5,
    8,  # waist
    11,
    15,
    19,
    21,
    23,
    25,
    27,  # left arm
    12,
    16,
    20,
    22,
    24,
    26,
    28,  # right arm
]

PROTOCOL_JOINT_NAMES: list[str] = [URDF_JOINT_NAMES[i] for i in URDF_TO_PROTOCOL]


def reorder_urdf_to_protocol(dof_pos: np.ndarray) -> np.ndarray:
    """Reorder `(29,)` or `(N, 29)` joint values from URDF order to SONIC protocol (IsaacLab) order.

    >>> import numpy as np
    >>> x = np.arange(29, dtype=np.float32)
    >>> float(reorder_urdf_to_protocol(x)[0])
    0.0
    >>> float(reorder_urdf_to_protocol(x)[1])
    6.0
    """
    return dof_pos[..., URDF_TO_PROTOCOL]


def reorder_protocol_to_urdf(dof_pos: np.ndarray) -> np.ndarray:
    """Reorder `(29,)` or `(N, 29)` joint values from SONIC protocol (IsaacLab) order to URDF order.

    >>> import numpy as np
    >>> x = np.arange(29, dtype=np.float32)
    >>> rt = reorder_protocol_to_urdf(reorder_urdf_to_protocol(x))
    >>> bool((rt == x).all())
    True
    """
    return dof_pos[..., PROTOCOL_TO_URDF]


# ── Wire format helpers ──────────────────────────────────────────────────────


def build_header(fields: list[dict], *, protocol_version: int = 1, count: int = 1) -> bytes:
    """Build a 1280-byte null-padded JSON header.

    >>> h = build_header([{"name": "x", "dtype": "f32", "shape": [1]}])
    >>> len(h)
    1280
    """
    header = {"v": protocol_version, "endian": "le", "count": count, "fields": fields}
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        raise ValueError(f"Header JSON too large: {len(header_json)} > {HEADER_SIZE}")
    return header_json.ljust(HEADER_SIZE, b"\x00")


def _add_field(fields: list[dict], binary_parts: list[bytes], name: str, arr: np.ndarray, dtype: np.dtype) -> None:
    arr = np.ascontiguousarray(arr.astype(dtype))
    wire_dtype = _NP_TO_WIRE[arr.dtype.name]
    fields.append({"name": name, "dtype": wire_dtype, "shape": list(arr.shape)})
    binary_parts.append(arr.tobytes())


def pack_v1_message(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    body_quat: np.ndarray,
    frame_index: np.ndarray,
    *,
    left_hand_joints: np.ndarray | None = None,
    right_hand_joints: np.ndarray | None = None,
    catch_up: bool = True,
) -> bytes:
    """Pack a SONIC Protocol v1 (joint-based) ZMQ message.

    Args:
        joint_pos: `[N, 29]` joint positions (f32).
        joint_vel: `[N, 29]` joint velocities (f32).
        body_quat: `[N, 4]` root quaternion wxyz (f64).
        frame_index: `[N]` monotonically increasing frame indices (i64).
        left_hand_joints: Optional `[7]` left Dex3 hand joints (f32).
        right_hand_joints: Optional `[7]` right Dex3 hand joints (f32).
        catch_up: Whether the receiver should reset on frame gaps.

    Returns:
        Complete message bytes: `TOPIC + header + payload`.

    >>> msg = pack_v1_message(
    ...     np.zeros((1, 29), dtype=np.float32),
    ...     np.zeros((1, 29), dtype=np.float32),
    ...     np.array([[1, 0, 0, 0]], dtype=np.float64),
    ...     np.array([0], dtype=np.int64),
    ... )
    >>> msg[:4]
    b'pose'
    """
    fields: list[dict] = []
    binary_parts: list[bytes] = []

    _add_field(fields, binary_parts, "joint_pos", joint_pos, np.dtype(np.float32))
    _add_field(fields, binary_parts, "joint_vel", joint_vel, np.dtype(np.float32))
    _add_field(fields, binary_parts, "body_quat", body_quat, np.dtype(np.float64))
    _add_field(fields, binary_parts, "frame_index", frame_index, np.dtype(np.int64))

    if left_hand_joints is not None:
        _add_field(fields, binary_parts, "left_hand_joints", left_hand_joints, np.dtype(np.float32))
    if right_hand_joints is not None:
        _add_field(fields, binary_parts, "right_hand_joints", right_hand_joints, np.dtype(np.float32))

    catch_up_arr = np.array([1 if catch_up else 0], dtype=np.uint8)
    _add_field(fields, binary_parts, "catch_up", catch_up_arr, np.dtype(np.uint8))

    count = int(joint_pos.shape[0])
    header_bytes = build_header(fields, protocol_version=1, count=count)
    return TOPIC + header_bytes + b"".join(binary_parts)


def unpack_v1_message(data: bytes) -> dict[str, np.ndarray]:
    """Unpack a SONIC v1 message back to named arrays.

    >>> msg = pack_v1_message(
    ...     np.ones((2, 29), dtype=np.float32),
    ...     np.zeros((2, 29), dtype=np.float32),
    ...     np.tile([1, 0, 0, 0], (2, 1)).astype(np.float64),
    ...     np.arange(2, dtype=np.int64),
    ... )
    >>> fields = unpack_v1_message(msg)
    >>> fields["joint_pos"].shape
    (2, 29)
    """
    topic_len = len(TOPIC)
    if data[:topic_len] != TOPIC:
        raise ValueError(f"Expected topic {TOPIC!r}, got {data[:topic_len]!r}")

    header_bytes = data[topic_len : topic_len + HEADER_SIZE]
    header = json.loads(header_bytes.rstrip(b"\x00"))

    payload = data[topic_len + HEADER_SIZE :]
    offset = 0
    result: dict[str, np.ndarray] = {}

    for f in header["fields"]:
        name = f["name"]
        dtype = _WIRE_TO_NP[f["dtype"]]
        shape = tuple(f["shape"])
        nbytes = int(np.prod(shape)) * dtype.itemsize
        raw = payload[offset : offset + nbytes]
        if len(raw) != nbytes:
            raise ValueError(f"Field {name!r}: expected {nbytes} bytes, got {len(raw)}")
        result[name] = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
        offset += nbytes

    return result


def build_command_message(*, start: bool = False, stop: bool = False, planner: bool = True) -> bytes:
    """Build a SONIC v1 command message for gear_sonic_deploy zmq_manager mode.

    >>> msg = build_command_message(start=True)
    >>> msg[:7]
    b'command'
    """
    fields: list[dict] = []
    binary_parts: list[bytes] = []

    _add_field(fields, binary_parts, "start", np.array([int(start)], dtype=np.uint8), np.dtype(np.uint8))
    _add_field(fields, binary_parts, "stop", np.array([int(stop)], dtype=np.uint8), np.dtype(np.uint8))
    _add_field(fields, binary_parts, "planner", np.array([int(planner)], dtype=np.uint8), np.dtype(np.uint8))

    header_bytes = build_header(fields, protocol_version=1, count=1)
    return COMMAND_TOPIC + header_bytes + b"".join(binary_parts)


@dataclass
class SonicTargetConfig(TargetConfig):
    """SONIC v1 publisher settings.

    >>> SonicTargetConfig().port
    5556
    """

    host: str = "*"
    port: int = 5556
    lookahead: int = 50

    def build(self) -> "SonicTarget":
        """Bind and return a SONIC target.

        >>> SonicTargetConfig(port=15556).build()  # doctest: +SKIP
        """
        return SonicTarget(self)


class SonicTarget(Target):
    """Publish canonical humanoid commands using SONIC v1.

    >>> SonicTarget(SonicTargetConfig(port=15556))  # doctest: +SKIP
    """

    def __init__(self, config: SonicTargetConfig) -> None:
        self._config = config
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.bind(f"tcp://{config.host}:{config.port}")
        self._frame_index = 0
        self._qvel = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._previous_qpos: np.ndarray | None = None
        self._previous_time: float | None = None

    def set_command(self, command: Command) -> None:
        """Finite-difference joint velocities, then pack and publish one command.

        >>> target.set_command(command)  # doctest: +SKIP
        """
        assert isinstance(command, BasePoseQPosCommand)
        now = time.monotonic()
        dt = now - self._previous_time if self._previous_time is not None else 0.0
        if self._previous_qpos is None or dt <= 1e-4:
            self._qvel.fill(0.0)
        else:
            raw_qvel = ((command.qpos - self._previous_qpos) / dt).astype(np.float32)
            self._qvel = (0.3 * raw_qvel + 0.7 * self._qvel).astype(np.float32)
        self._previous_qpos = command.qpos.copy()
        self._previous_time = now
        frame_indices = np.arange(
            self._frame_index,
            self._frame_index + self._config.lookahead,
            dtype=np.int64,
        )
        message = pack_v1_message(
            joint_pos=np.tile(reorder_urdf_to_protocol(command.qpos), (self._config.lookahead, 1)),
            joint_vel=np.tile(reorder_urdf_to_protocol(self._qvel), (self._config.lookahead, 1)),
            body_quat=np.tile(command.base_pose[3:7], (self._config.lookahead, 1)),
            frame_index=frame_indices,
            catch_up=True,
        )
        self._socket.send(message, flags=zmq.NOBLOCK)
        self._frame_index += 1

    def close(self) -> None:
        """Close the publisher.

        >>> target.close()  # doctest: +SKIP
        """
        self._socket.close()
        self._context.term()
