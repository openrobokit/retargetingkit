"""Apple Vision Pro hand and head tracking via avp_stream."""

import pickle
import time
from dataclasses import dataclass
from typing import Callable, Generic, TypedDict, TypeVar, cast

import numpy as np
from jaxtyping import Float, Float32, Float64

from retargetingkit.sources.frames import ArmFrame, ArmHandFrame, Frame, HandFrame, HandPose, Side
from retargetingkit.sources.source import Source, SourceConfig
from robokit.xform.numpy import tf_mat_to_pose7


VISION_PRO_25_TO_MANO_21 = (0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24)
VISION_PRO_JOINT_NAMES = (
    "wrist",
    "thumb_knuckle",
    "thumb_intermediate_base",
    "thumb_intermediate_tip",
    "thumb_tip",
    "index_metacarpal",
    "index_knuckle",
    "index_intermediate_base",
    "index_intermediate_tip",
    "index_tip",
    "middle_metacarpal",
    "middle_knuckle",
    "middle_intermediate_base",
    "middle_intermediate_tip",
    "middle_tip",
    "ring_metacarpal",
    "ring_knuckle",
    "ring_intermediate_base",
    "ring_intermediate_tip",
    "ring_tip",
    "little_metacarpal",
    "little_knuckle",
    "little_intermediate_base",
    "little_intermediate_tip",
    "little_tip",
)
VISION_PRO_HAND_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (0, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (0, 15),
    (15, 16),
    (16, 17),
    (17, 18),
    (18, 19),
    (0, 20),
    (20, 21),
    (21, 22),
    (22, 23),
    (23, 24),
)
_OPERATOR_TO_AVP = {
    "right": np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float32),
    "left": np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
}
_AVP_WORLD_TO_CANONICAL = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)
FrameT = TypeVar("FrameT", bound=Frame)


class VisionProFrame(TypedDict):
    """Raw `avp_stream` tracking dictionary."""

    left_wrist: Float64[np.ndarray, "1 4 4"]
    right_wrist: Float64[np.ndarray, "1 4 4"]
    left_fingers: Float64[np.ndarray, "25 4 4"]
    right_fingers: Float64[np.ndarray, "25 4 4"]
    left_arm: Float64[np.ndarray, "J 4 4"]
    right_arm: Float64[np.ndarray, "J 4 4"]
    head: Float64[np.ndarray, "1 4 4"]
    left_pinch_distance: np.float64
    right_pinch_distance: np.float64
    left_wrist_roll: np.float64
    right_wrist_roll: np.float64


def vision_pro_to_mano_keypoints(
    finger_poses: Float[np.ndarray, "25 4 4"],
    side: Side,
) -> Float32[np.ndarray, "21 3"]:
    """Convert 25 wrist-local Vision Pro finger poses to MANO-21 points.

    >>> poses = np.tile(np.eye(4, dtype=np.float32), (25, 1, 1))
    >>> vision_pro_to_mano_keypoints(poses, "right").shape
    (21, 3)
    """
    points = finger_poses[:, :3, 3] @ _OPERATOR_TO_AVP[side]
    mano = points[list(VISION_PRO_25_TO_MANO_21)]
    return (mano - mano[:1]).astype(np.float32)


def vision_pro_wrist_matrix(
    wrist_pose: Float[np.ndarray, "4 4"],
    side: Side,
) -> Float64[np.ndarray, "4 4"]:
    """Convert one Vision Pro wrist pose to the robot end-effector axes.

    >>> vision_pro_wrist_matrix(np.eye(4, dtype=np.float32), "right").shape
    (4, 4)
    """
    wrist = wrist_pose.astype(np.float64).copy()
    wrist[:3, :3] = _AVP_WORLD_TO_CANONICAL @ wrist_pose[:3, :3] @ _OPERATOR_TO_AVP[side]
    wrist[:3, 3] = _AVP_WORLD_TO_CANONICAL @ wrist_pose[:3, 3]
    return wrist


def vision_pro_hand_pose(frame: dict, side: Side) -> HandPose:
    """Convert one Vision Pro hand side to a canonical MANO-21 pose.

    >>> frame = {
    ...     "right_wrist": np.eye(4, dtype=np.float32)[None],
    ...     "right_fingers": np.tile(np.eye(4, dtype=np.float32), (25, 1, 1)),
    ... }
    >>> hand = vision_pro_hand_pose(frame, "right")
    >>> hand.positions.shape
    (21, 3)
    """
    return HandPose(positions=vision_pro_to_mano_keypoints(frame[f"{side}_fingers"], side))


def vision_pro_hand_frame(frame: dict) -> HandFrame:
    """Convert available Vision Pro finger poses to a canonical hand frame.

    >>> raw = {"right_fingers": np.tile(np.eye(4, dtype=np.float32), (25, 1, 1))}
    >>> vision_pro_hand_frame(raw).hands["right"].positions.shape
    (21, 3)
    """
    return HandFrame(
        {side: vision_pro_hand_pose(frame, side) for side in ("left", "right") if f"{side}_fingers" in frame}
    )


def vision_pro_arm_frame(frame: dict) -> ArmFrame:
    """Convert available Vision Pro wrist poses to a canonical arm frame.

    >>> raw = {"right_wrist": np.eye(4, dtype=np.float32)[None]}
    >>> vision_pro_arm_frame(raw).wrists["right"].shape
    (7,)
    """
    return ArmFrame(
        {
            side: tf_mat_to_pose7(vision_pro_wrist_matrix(frame[f"{side}_wrist"][0], side)).astype(np.float32)
            for side in ("left", "right")
            if f"{side}_wrist" in frame
        }
    )


def vision_pro_arm_hand_frame(frame: dict) -> ArmHandFrame:
    """Convert one sample into synchronized canonical arm and hand frames.

    >>> raw = {
    ...     "right_wrist": np.eye(4, dtype=np.float32)[None],
    ...     "right_fingers": np.tile(np.eye(4, dtype=np.float32), (25, 1, 1)),
    ... }
    >>> vision_pro_arm_hand_frame(raw).hand.hands["right"].positions.shape
    (21, 3)
    """
    return ArmHandFrame(vision_pro_arm_frame(frame), vision_pro_hand_frame(frame))


class VisionPro:
    """Apple Vision Pro hand and head tracking via live `avp_stream.VisionProStreamer` or a recorded pickle replay.

    >>> # VisionPro(ip="192.168.1.100")  # doctest: +SKIP
    """

    def __init__(self, ip: str | None = None, recording_path: str | None = None, recording_fps: int = 20):
        self._recording_fps = recording_fps
        if recording_path is not None:
            with open(recording_path, "rb") as f:
                self._frames: list[VisionProFrame] = pickle.load(f)
            self._start_time = time.monotonic()
            self._streamer = None
        else:
            from avp_stream import VisionProStreamer  # type: ignore[import-untyped]

            assert ip is not None
            self._streamer = VisionProStreamer(ip=ip, record=False)
            self._frames = []
            self._start_time = 0.0

    def _get_frame(self) -> VisionProFrame:
        if self._streamer is not None:
            frame = self._streamer.latest
            assert frame is not None
        else:
            elapsed = time.monotonic() - self._start_time
            idx = min(int(elapsed * self._recording_fps), len(self._frames) - 1)
            frame = self._frames[idx]

        for key, shape in {
            "left_wrist": (1, 4, 4),
            "right_wrist": (1, 4, 4),
            "left_fingers": (25, 4, 4),
            "right_fingers": (25, 4, 4),
            "head": (1, 4, 4),
        }.items():
            value = frame[key]
            assert value.shape == shape, f"{key} shape must be {shape}, got {value.shape}"
            assert value.dtype == np.float64, f"{key} dtype must be float64, got {value.dtype}"
            assert np.isfinite(value).all(), f"{key} must be finite"
        for key in ("left_arm", "right_arm"):
            value = frame[key]
            assert value.shape in ((25, 4, 4), (27, 4, 4)), f"{key} shape must be (25|27, 4, 4), got {value.shape}"
            assert value.dtype == np.float64, f"{key} dtype must be float64, got {value.dtype}"
            assert np.isfinite(value).all(), f"{key} must be finite"
        for key in ("left_pinch_distance", "right_pinch_distance", "left_wrist_roll", "right_wrist_roll"):
            value = frame[key]
            assert type(value) is np.float64, f"{key} dtype must be float64, got {type(value).__name__}"
            assert np.isfinite(value), f"{key} must be finite"
        return cast(VisionProFrame, frame)

    def get_head(self) -> Float64[np.ndarray, "1 4 4"]:
        """Return head pose as a `(1, 4, 4)` float64 matrix."""
        return self._get_frame()["head"]

    def get_left_wrist(self) -> Float64[np.ndarray, "1 4 4"]:
        """Return left wrist pose as a `(1, 4, 4)` float64 matrix."""
        return self._get_frame()["left_wrist"]

    def get_right_wrist(self) -> Float64[np.ndarray, "1 4 4"]:
        """Return right wrist pose as a `(1, 4, 4)` float64 matrix."""
        return self._get_frame()["right_wrist"]

    def get_left_fingers(self) -> Float64[np.ndarray, "25 4 4"]:
        """Return left hand 25 joint poses as `(25, 4, 4)` float64."""
        return self._get_frame()["left_fingers"]

    def get_right_fingers(self) -> Float64[np.ndarray, "25 4 4"]:
        """Return right hand 25 joint poses as `(25, 4, 4)` float64."""
        return self._get_frame()["right_fingers"]

    def get_left_pinch_distance(self) -> float:
        """Return left hand pinch distance."""
        return float(self._get_frame()["left_pinch_distance"])

    def get_right_pinch_distance(self) -> float:
        """Return right hand pinch distance."""
        return float(self._get_frame()["right_pinch_distance"])

    def get_left_wrist_roll(self) -> float:
        """Return left wrist roll angle."""
        return float(self._get_frame()["left_wrist_roll"])

    def get_right_wrist_roll(self) -> float:
        """Return right wrist roll angle."""
        return float(self._get_frame()["right_wrist_roll"])

    def get_all(self) -> VisionProFrame:
        """Return all tracking data as a dict (same keys as avp_stream)."""
        return self._get_frame()


class VisionProSource(Source[FrameT], Generic[FrameT]):
    """One Vision Pro connection projected into one canonical frame type.

    >>> # VisionProSource(VisionPro(ip="192.168.1.100"), vision_pro_hand_frame)  # doctest: +SKIP
    """

    def __init__(self, device: VisionPro, convert: Callable[[dict], FrameT]) -> None:
        self._device = device
        self._convert = convert

    def get_frame(self) -> FrameT | None:
        return self._convert(dict(self._device.get_all()))

    def close(self) -> None:
        """No shutdown API on avp_stream; nothing to release."""


@dataclass
class VisionProHandConfig(SourceConfig[HandFrame]):
    """Vision Pro connection projected as hand keypoints.

    >>> VisionProHandConfig()
    VisionProHandConfig(ip=None, recording_path=None, recording_fps=20)
    """

    ip: str | None = None
    recording_path: str | None = None
    recording_fps: int = 20

    def build(self) -> VisionProSource[HandFrame]:
        """Construct one canonical hand source."""
        return VisionProSource(VisionPro(self.ip, self.recording_path, self.recording_fps), vision_pro_hand_frame)


@dataclass
class VisionProArmConfig(SourceConfig[ArmFrame]):
    """Vision Pro connection projected as world-frame wrists.

    >>> VisionProArmConfig()
    VisionProArmConfig(ip=None, recording_path=None, recording_fps=20)
    """

    ip: str | None = None
    recording_path: str | None = None
    recording_fps: int = 20

    def build(self) -> VisionProSource[ArmFrame]:
        """Construct one canonical arm source."""
        return VisionProSource(VisionPro(self.ip, self.recording_path, self.recording_fps), vision_pro_arm_frame)


@dataclass
class VisionProArmHandConfig(SourceConfig[ArmHandFrame]):
    """Vision Pro connection projected as synchronized arm and hand frames.

    >>> VisionProArmHandConfig()
    VisionProArmHandConfig(ip=None, recording_path=None, recording_fps=20)
    """

    ip: str | None = None
    recording_path: str | None = None
    recording_fps: int = 20

    def build(self) -> VisionProSource[ArmHandFrame]:
        """Construct one synchronized canonical arm-and-hand source."""
        return VisionProSource(VisionPro(self.ip, self.recording_path, self.recording_fps), vision_pro_arm_hand_frame)
