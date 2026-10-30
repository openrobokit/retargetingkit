"""Abstract human-motion source contracts."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from retargetingkit.sources.frames import Frame


FrameT = TypeVar("FrameT", bound=Frame)


class Source(ABC, Generic[FrameT]):
    """A device, recording, or stream producing canonical motion frames.


    Built-in implementations include:
        - `VisionProSource`: Apple Vision Pro hands, wrists, or both.
        - `PicoSource`: PICO headset body tracking.
        - `ManusSource`: Manus glove hand keypoints.
        - `AxisStudioSource`: Axis Studio mocap body transforms.
        - `BodyFileSource`: recorded body frames replayed from disk.
    """

    @abstractmethod
    def get_frame(self) -> FrameT | None:
        """Return the latest canonical frame, or None before the first sample."""

    @abstractmethod
    def close(self) -> None:
        """Release the device."""


class SourceConfig(ABC, Generic[FrameT]):
    """Config for Source."""

    @abstractmethod
    def build(self) -> Source[FrameT]:
        """Construct and start the source device."""
