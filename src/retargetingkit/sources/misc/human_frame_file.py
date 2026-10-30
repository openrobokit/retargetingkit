"""Replay ordered humanoid transforms from a .bvh clip or .npz PICO recording, wall-clock paced, optionally looped."""

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from retargetingkit.sources.frames import BODY_JOINTS, BodyFrame
from retargetingkit.sources.source import Source, SourceConfig
from retargetingkit.sources.vr.pico import pico_pose_array_to_transforms
from robokit.helpers.humanoid_retargeting.loaders import load_bvh_motion


logger = logging.getLogger(__name__)


class BodyFileSource(Source[BodyFrame]):
    """Fake-stream ordered `[J, 7]` transforms from a `.bvh` clip or PICO recording."""

    def __init__(
        self,
        path: str | Path,
        human_joint_names: Sequence[str] = BODY_JOINTS,
        loop: bool = True,
        target_fps: float = 0.0,
        use_ankle_as_foot_target: bool = False,
        human_height: float = 1.75,
    ) -> None:
        path = Path(path)
        self.human_joint_names = tuple(human_joint_names)
        suffix = path.suffix.lower()
        logger.info("Loading humanoid motion file: %s", path)

        if suffix == ".bvh":
            load_names = list(self.human_joint_names)
            if use_ankle_as_foot_target:
                for ankle, foot in (("left_ankle", "left_foot"), ("right_ankle", "right_foot")):
                    if foot in load_names and ankle not in load_names:
                        load_names.append(ankle)
            tgt = target_fps if target_fps > 0 else None
            frames, self._fps_val = load_bvh_motion(path, load_names, tgt_fps=tgt)
            if use_ankle_as_foot_target:
                for ankle, foot in (("left_ankle", "left_foot"), ("right_ankle", "right_foot")):
                    if foot in load_names:
                        frames[:, load_names.index(foot)] = frames[:, load_names.index(ankle)]
            self._frames = np.ascontiguousarray(frames[:, [load_names.index(name) for name in self.human_joint_names]])
            n = len(self._frames)
            self._timestamps = np.arange(n, dtype=np.float64) / self._fps_val if self._fps_val > 0 else np.zeros(n)
        elif suffix == ".npz":
            data = np.load(path, allow_pickle=False)
            poses: np.ndarray = data["poses"]
            if poses.ndim != 3 or poses.shape[1:] != (24, 7):
                raise ValueError(f"Expected poses shape (N, 24, 7), got {poses.shape}")
            self._timestamps = data["timestamps_s"].astype(np.float64)
            self._frames = np.stack(
                [
                    pico_pose_array_to_transforms(
                        poses[i],
                        self.human_joint_names,
                        use_ankle_as_foot_target=use_ankle_as_foot_target,
                    )
                    for i in range(poses.shape[0])
                ]
            )
            duration = float(self._timestamps[-1] - self._timestamps[0]) if len(self._timestamps) > 1 else 0.0
            self._fps_val = (len(self._timestamps) - 1) / duration if duration > 0 else 0.0
        else:
            raise ValueError(f"Unsupported file type {suffix!r}; expected .bvh or .npz")

        self._human_height_val = human_height
        self._loop = loop
        self._start_time = time.monotonic()
        self._frame_idx = 0
        self._held = True
        logger.info("Loaded %s: %d frames @ %.1f fps, loop=%s", path.name, len(self._frames), self._fps_val, loop)

    def release(self) -> None:
        self._held = False
        self._start_time = time.monotonic()
        self._frame_idx = 0

    def get_frame(self) -> BodyFrame | None:
        if self._held:
            self._frame_idx = 0
            return BodyFrame(self.human_joint_names, self._frames[0], self._human_height_val)

        elapsed = time.monotonic() - self._start_time
        last_ts = float(self._timestamps[-1])
        if elapsed > last_ts:
            if self._loop and last_ts > 0:
                self._start_time += int(elapsed // last_ts) * last_ts
                elapsed = time.monotonic() - self._start_time
            else:
                return None

        idx = int(np.searchsorted(self._timestamps, elapsed, side="right") - 1)
        self._frame_idx = max(0, min(idx, len(self._frames) - 1))
        return BodyFrame(self.human_joint_names, self._frames[self._frame_idx], self._human_height_val)

    def reset(self) -> None:
        self._start_time = time.monotonic()
        self._frame_idx = 0

    @property
    def fps(self) -> float:
        return self._fps_val

    @property
    def human_height(self) -> float:
        return self._human_height_val

    @property
    def num_frames(self) -> int:
        return len(self._frames)

    @property
    def frame_idx(self) -> int:
        return self._frame_idx

    def close(self) -> None:
        """No external resources to release."""


@dataclass
class BodyFileSourceConfig(SourceConfig[BodyFrame]):
    """BVH or PICO recording playback settings.

    >>> BodyFileSourceConfig(Path("motion.bvh")).path.name
    'motion.bvh'
    """

    path: Path
    joint_names: tuple[str, ...] = BODY_JOINTS
    loop: bool = True
    target_fps: float = 0.0
    use_ankle_as_foot_target: bool = False
    human_height: float = 1.75

    def build(self) -> Source[BodyFrame]:
        source = BodyFileSource(
            self.path,
            self.joint_names,
            self.loop,
            self.target_fps,
            self.use_ankle_as_foot_target,
            self.human_height,
        )
        source.release()
        return source
