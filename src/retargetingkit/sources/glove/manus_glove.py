"""Typed left/right aggregation for Manus glove frames."""

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float32

from retargetingkit.sources.frames import HandFrame, HandPose, Side
from retargetingkit.sources.glove._manus_sdk import (
    SIDE_LEFT,
    SIDE_RIGHT,
    ManusClient,
    ManusSdkFrame,
    resolve_manus_sdk_library_path,
)
from retargetingkit.sources.glove.manus_conversion import hand_pose_from_glove, load_glove_calibration
from retargetingkit.sources.source import Source, SourceConfig


@dataclass(frozen=True)
class ManusGloveFrame:
    """One Manus glove frame with its resolved hand side."""

    glove_id: int
    publish_time: int
    recv_ns: int
    side_code: int
    side: str
    positions: Float32[np.ndarray, "25 3"]
    rotations: Float32[np.ndarray, "25 4"]


def parse_glove_id(value: str | None) -> int | None:
    """Parse a decimal or hexadecimal glove ID.

    >>> parse_glove_id("0xFF")
    255
    >>> parse_glove_id("42")
    42
    >>> parse_glove_id(None) is None
    True
    """
    if value is None:
        return None
    text = value.strip().lower()
    return int(text, 16 if text.startswith("0x") else 10)


class DualManusGloveAggregator:
    """Resolve SDK frames into stable left and right glove streams."""

    def __init__(self, left_override: int | None = None, right_override: int | None = None) -> None:
        self.left_override = left_override
        self.right_override = right_override
        self._glove_side: dict[int, int] = {}
        self._latest: dict[str, ManusGloveFrame | None] = {"left": None, "right": None}
        self._lock = threading.Lock()

    def update(self, frame: ManusSdkFrame) -> None:
        """Resolve and store one SDK frame."""
        assert frame.positions.shape == (25, 3), f"positions shape must be (25, 3), got {frame.positions.shape}"
        assert frame.positions.dtype == np.float32, f"positions dtype must be float32, got {frame.positions.dtype}"
        assert np.isfinite(frame.positions).all(), "positions must be finite"
        assert frame.rotations.shape == (25, 4), f"rotations shape must be (25, 4), got {frame.rotations.shape}"
        assert frame.rotations.dtype == np.float32, f"rotations dtype must be float32, got {frame.rotations.dtype}"
        assert np.isfinite(frame.rotations).all(), "rotations must be finite"
        if self.left_override is not None and frame.glove_id == self.left_override:
            side_code = SIDE_LEFT
        elif self.right_override is not None and frame.glove_id == self.right_override:
            side_code = SIDE_RIGHT
        elif frame.side_code in (SIDE_LEFT, SIDE_RIGHT):
            side_code = self._glove_side.setdefault(frame.glove_id, frame.side_code)
        else:
            side_code = self._glove_side.get(frame.glove_id, 0)
            reserved = {
                side
                for side, override in (
                    (SIDE_LEFT, self.left_override),
                    (SIDE_RIGHT, self.right_override),
                )
                if override is not None
            }
            used = set(self._glove_side.values())
            for candidate in (SIDE_RIGHT, SIDE_LEFT):
                if side_code == 0 and candidate not in reserved and candidate not in used:
                    side_code = candidate
                    self._glove_side[frame.glove_id] = candidate

        side = {SIDE_LEFT: "left", SIDE_RIGHT: "right"}.get(side_code, "unknown")
        output = ManusGloveFrame(
            glove_id=frame.glove_id,
            publish_time=frame.publish_time,
            recv_ns=frame.recv_ns,
            side_code=side_code,
            side=side,
            positions=frame.positions,
            rotations=frame.rotations,
        )
        with self._lock:
            if side != "unknown":
                self._latest[side] = output

    def latest(self) -> tuple[ManusGloveFrame | None, ManusGloveFrame | None]:
        """Return the latest `(left, right)` frames."""
        with self._lock:
            return self._latest["left"], self._latest["right"]

    def glove_ids(self) -> list[int]:
        """Return sorted automatically classified glove IDs."""
        with self._lock:
            return sorted(self._glove_side)


class ManusSource(Source[HandFrame]):
    """One Manus SDK connection streaming the latest left/right hand poses; create at most one per process.

    >>> # ManusSource()  # doctest: +SKIP
    """

    def __init__(
        self,
        left_id: str | None = None,
        right_id: str | None = None,
        use_calibration: bool = True,
        calibration_path: Path | None = None,
    ) -> None:
        self._glove_calib = load_glove_calibration(calibration_path) if use_calibration else None
        self._aggregator = DualManusGloveAggregator(
            left_override=parse_glove_id(left_id),
            right_override=parse_glove_id(right_id),
        )
        self._client = ManusClient(resolve_manus_sdk_library_path())
        self._client.start(self._aggregator.update)

    def get_frame(self) -> HandFrame | None:
        """Return all available hand poses, or None before the first glove packet."""
        left, right = self._aggregator.latest()
        gloves: dict[Side, ManusGloveFrame | None] = {"left": left, "right": right}
        hands: dict[Side, HandPose] = {
            side: hand_pose_from_glove(glove.positions, glove.rotations, self._glove_calib)
            for side, glove in gloves.items()
            if glove is not None
        }
        if not hands:
            return None
        return HandFrame(hands)

    def close(self) -> None:
        """Stop the Manus SDK client."""
        self._client.stop()


@dataclass
class ManusSourceConfig(SourceConfig[HandFrame]):
    """Manus glove connection settings.

    >>> # ManusSourceConfig().build()  # doctest: +SKIP
    """

    left_id: str | None = None
    """Override the left glove ID (decimal or `0x` hex)."""
    right_id: str | None = None
    """Override the right glove ID (decimal or `0x` hex)."""
    use_calibration: bool = True
    """Apply the Manus thumb transform and per-finger scales."""
    calibration_path: Path | None = None
    """Calibration NPZ path; `None` fetches the `manus_calib` Hugging Face asset."""

    def build(self) -> Source[HandFrame]:
        return ManusSource(
            left_id=self.left_id,
            right_id=self.right_id,
            use_calibration=self.use_calibration,
            calibration_path=self.calibration_path,
        )
