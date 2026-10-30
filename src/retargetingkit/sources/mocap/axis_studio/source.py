"""Axis Studio Calc broadcast source."""

import logging
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from retargetingkit.sources.frames import BODY_JOINTS, BodyFrame
from retargetingkit.sources.source import Source, SourceConfig
from robokit.xform.numpy import quaternion_multiply


logger = logging.getLogger(__name__)

_Y_UP_TO_Z_UP_ROT = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
_Y_UP_TO_Z_UP_QUAT_WXYZ = np.array([0.70710677, 0.70710677, 0.0, 0.0], dtype=np.float32)

_AXIS_TO_SMPLX = {
    "Hips": "pelvis",
    "Spine2": "spine3",
    "Spine1": "spine2",
    "Spine": "spine1",
    "LeftUpLeg": "left_hip",
    "LeftLeg": "left_knee",
    "LeftFoot": "left_ankle",
    "LeftToeBase": "left_foot",
    "LeftArm": "left_shoulder",
    "LeftForeArm": "left_elbow",
    "LeftHand": "left_wrist",
    "RightUpLeg": "right_hip",
    "RightLeg": "right_knee",
    "RightFoot": "right_ankle",
    "RightToeBase": "right_foot",
    "RightArm": "right_shoulder",
    "RightForeArm": "right_elbow",
    "RightHand": "right_wrist",
}


def _connect_calc(port: int) -> Any:
    from retargetingkit.sources.mocap.axis_studio import _robotapi

    app = _robotapi.MCPApplication()
    settings = _robotapi.MCPSettings()
    settings.set_udp(port)
    settings.set_calc_data()
    app.set_settings(settings)
    ok, msg = app.open()
    if not ok:
        raise RuntimeError(f"Failed to open Axis Studio connection: {msg}")
    return app


def _collect_joint_tree(joint: Any, out: dict[str, dict[str, Any]], parent_name: str | None = None) -> None:
    name = joint.get_name()
    body_part = joint.get_body_part()
    out[name] = {
        "joint_tag": joint.get_tag(),
        "parent_name": parent_name,
        "local_position": list(joint.get_local_position()),
        "local_rotation_quat": list(joint.get_local_rotation()),
        "global_position": list(body_part.get_position()),
        "global_rotation_quat": list(body_part.get_posture()),
    }
    for child in joint.get_children():
        _collect_joint_tree(child, out, name)


def _iter_avatar_frames(app: Any) -> Iterator[dict[str, Any]]:
    from retargetingkit.sources.mocap.axis_studio import _robotapi

    while True:
        events = app.poll_next_event()
        for event in events:
            if event.event_type != _robotapi.MCPEventType.AvatarUpdated:
                continue
            avatar = _robotapi.MCPAvatar(event.event_data.avatar_handle)
            joints: dict[str, dict[str, Any]] = {}
            _collect_joint_tree(avatar.get_root_joint(), joints)
            yield {
                "timestamp": event.timestamp,
                "avatar_name": avatar.get_name(),
                "joints": joints,
            }
        if not events:
            time.sleep(0.001)


def _mocap_frame_to_transforms(
    frame_data: dict[str, Any],
    human_joint_names: Sequence[str],
    use_ankle_as_foot_target: bool = False,
) -> np.ndarray:
    joints = frame_data["joints"]
    transforms: dict[str, np.ndarray] = {}
    for axis_name, human_name in _AXIS_TO_SMPLX.items():
        joint = joints[axis_name]
        position = np.asarray(joint["global_position"], dtype=np.float32)
        quaternion = np.asarray(joint["global_rotation_quat"], dtype=np.float32)
        rotated = quaternion_multiply(_Y_UP_TO_Z_UP_QUAT_WXYZ, quaternion)
        transforms[human_name] = np.concatenate(
            [position @ _Y_UP_TO_Z_UP_ROT.T, rotated / np.linalg.norm(rotated)]
        ).astype(np.float32)

    if use_ankle_as_foot_target:
        for ankle, foot in (("left_ankle", "left_foot"), ("right_ankle", "right_foot")):
            transforms[foot] = transforms[ankle]
    return np.ascontiguousarray(np.stack([transforms[name] for name in human_joint_names]))


class AxisStudioSource(Source[BodyFrame]):
    """Realtime ordered humanoid transforms from Axis Studio Calc broadcast mode.

    >>> AxisStudioSource(("pelvis",), port=7012)  # doctest: +SKIP
    """

    def __init__(
        self,
        human_joint_names: Sequence[str] = BODY_JOINTS,
        port: int = 7012,
        default_human_height: float = 1.75,
        use_ankle_as_foot_target: bool = False,
    ) -> None:
        self.human_joint_names = tuple(human_joint_names)
        self._human_height = default_human_height
        self._use_ankle = use_ankle_as_foot_target
        self._latest: BodyFrame | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frame_idx = 0
        self._app = _connect_calc(port)
        logger.info("AxisStudioSource listening on UDP port %d (Calc mode)", port)
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self) -> None:
        for frame_data in _iter_avatar_frames(self._app):
            if self._stop.is_set():
                break
            frame = BodyFrame(
                self.human_joint_names,
                _mocap_frame_to_transforms(frame_data, self.human_joint_names, self._use_ankle),
                self._human_height,
            )
            with self._lock:
                self._latest = frame
                self._frame_idx += 1

    def get_frame(self) -> BodyFrame | None:
        """Return the latest converted frame, or `None` before the first packet."""
        with self._lock:
            return self._latest

    def reset(self) -> None:
        """Clear the latest frame and frame counter."""
        with self._lock:
            self._latest = None
            self._frame_idx = 0

    @property
    def human_height(self) -> float:
        """Configured human height in meters."""
        return self._human_height

    @property
    def frame_idx(self) -> int:
        """Number of valid frames received since construction or reset."""
        return self._frame_idx

    def close(self) -> None:
        """Stop polling and close the native client."""
        self._stop.set()
        self._thread.join(timeout=3.0)
        self._app.close()
        logger.info("AxisStudioSource closed")


@dataclass
class AxisStudioSourceConfig(SourceConfig[BodyFrame]):
    """Axis Studio Calc connection settings.

    >>> AxisStudioSourceConfig().port
    7012
    """

    port: int = 7012
    human_height: float = 1.75
    use_ankle_as_foot_target: bool = False
    joint_names: tuple[str, ...] = BODY_JOINTS

    def build(self) -> Source[BodyFrame]:
        return AxisStudioSource(
            self.joint_names,
            self.port,
            self.human_height,
            self.use_ankle_as_foot_target,
        )
