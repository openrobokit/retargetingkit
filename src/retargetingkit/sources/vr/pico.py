"""PICO 4 XR device and live body-and-controller source."""

import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from jaxtyping import Float, Float32

from retargetingkit.sources.frames import BODY_JOINTS as CANONICAL_BODY_JOINTS
from retargetingkit.sources.frames import BodyFrame
from retargetingkit.sources.source import Source, SourceConfig
from robokit.xform.numpy import quaternion_multiply


logger = logging.getLogger(__name__)

# 24 SMPL-standard body joint names (SDK order). Display-friendly PascalCase.
BODY_JOINT_NAMES: tuple[str, ...] = (
    "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee", "Spine2",
    "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot", "Neck", "Left_Collar",
    "Right_Collar", "Head", "Left_Shoulder", "Right_Shoulder", "Left_Elbow", "Right_Elbow",
    "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand",
)  # fmt: skip

# 8 digital button names supported by the SDK.
BUTTON_NAMES: tuple[str, ...] = (
    "A", "B", "X", "Y", "left_menu", "right_menu", "left_axis_click", "right_axis_click",
)  # fmt: skip


class MotionTrackerFrame(TypedDict):
    """One raw PICO Motion Tracker sample."""

    pose: Float32[np.ndarray, "7"]
    velocity: Float32[np.ndarray, "6"]
    acceleration: Float32[np.ndarray, "6"]


def _float32_array(value: object, shape: tuple[int, ...], name: str) -> Float32[np.ndarray, "*shape"]:
    array = np.asarray(value, dtype=np.float32)
    assert array.shape == shape, f"{name} shape must be {shape}, got {array.shape}"
    assert np.isfinite(array).all(), f"{name} must be finite"
    return array


class Pico:
    """PICO 4 XR multi-modal device: body, hands, controllers, buttons, and motion trackers (like `VisionPro`).

    >>> # Pico()  # doctest: +SKIP
    """

    def __init__(self):
        import xrobotoolkit_sdk as xrt  # type: ignore[import-not-found]

        self._xrt = xrt
        self._xrt.init()
        logger.info("PICO XRoboToolkit SDK initialized")

    def is_body_available(self) -> bool:
        """Return True if body tracking data is available."""
        return bool(self._xrt.is_body_data_available())

    def get_body_poses(self) -> Float32[np.ndarray, "24 7"]:
        """Return 24-joint body poses as `(24, 7)` float32 (`[x,y,z, qx,qy,qz,qw]`, Y-up meters)."""
        return _float32_array(self._xrt.get_body_joints_pose(), (24, 7), "body poses")

    def get_body_velocities(self) -> Float32[np.ndarray, "24 6"]:
        """Return 24-joint body velocities as `(24, 6)` float32 (`[vx,vy,vz, wx,wy,wz]`)."""
        return _float32_array(self._xrt.get_body_joints_velocity(), (24, 6), "body velocities")

    def get_body_accelerations(self) -> Float32[np.ndarray, "24 6"]:
        """Return 24-joint body accelerations as `(24, 6)` float32."""
        return _float32_array(self._xrt.get_body_joints_acceleration(), (24, 6), "body accelerations")

    def is_left_hand_active(self) -> bool:
        """Return True if left hand tracking is active."""
        return bool(self._xrt.get_left_hand_is_active())

    def is_right_hand_active(self) -> bool:
        """Return True if right hand tracking is active."""
        return bool(self._xrt.get_right_hand_is_active())

    def get_left_hand(self) -> Float32[np.ndarray, "27 7"] | None:
        """Return left hand 27-joint poses as `(27, 7)` float32, or None if inactive."""
        if not self._xrt.get_left_hand_is_active():
            return None
        return _float32_array(self._xrt.get_left_hand_tracking_state(), (27, 7), "left hand")

    def get_right_hand(self) -> Float32[np.ndarray, "27 7"] | None:
        """Return right hand 27-joint poses as `(27, 7)` float32, or None if inactive."""
        if not self._xrt.get_right_hand_is_active():
            return None
        return _float32_array(self._xrt.get_right_hand_tracking_state(), (27, 7), "right hand")

    def get_headset_pose(self) -> Float32[np.ndarray, "7"]:
        """Return headset pose as `(7,)` float32 `[x, y, z, qx, qy, qz, qw]`."""
        return _float32_array(self._xrt.get_headset_pose(), (7,), "headset pose")

    def get_left_controller_pose(self) -> Float32[np.ndarray, "7"]:
        """Return left controller pose as `(7,)` float32."""
        return _float32_array(self._xrt.get_left_controller_pose(), (7,), "left controller pose")

    def get_right_controller_pose(self) -> Float32[np.ndarray, "7"]:
        """Return right controller pose as `(7,)` float32."""
        return _float32_array(self._xrt.get_right_controller_pose(), (7,), "right controller pose")

    def get_left_trigger(self) -> float:
        """Return left trigger value (0.0-1.0)."""
        return float(self._xrt.get_left_trigger())

    def get_right_trigger(self) -> float:
        """Return right trigger value (0.0-1.0)."""
        return float(self._xrt.get_right_trigger())

    def get_left_grip(self) -> float:
        """Return left grip value (0.0-1.0)."""
        return float(self._xrt.get_left_grip())

    def get_right_grip(self) -> float:
        """Return right grip value (0.0-1.0)."""
        return float(self._xrt.get_right_grip())

    def get_button(self, name: str) -> bool:
        """Return button state. Valid: A, B, X, Y, left_menu, right_menu, left_axis_click, right_axis_click."""
        fn = {
            "A": self._xrt.get_A_button,
            "B": self._xrt.get_B_button,
            "X": self._xrt.get_X_button,
            "Y": self._xrt.get_Y_button,
            "left_menu": self._xrt.get_left_menu_button,
            "right_menu": self._xrt.get_right_menu_button,
            "left_axis_click": self._xrt.get_left_axis_click,
            "right_axis_click": self._xrt.get_right_axis_click,
        }.get(name)
        if fn is None:
            raise ValueError(f"Unknown button: {name!r}. Valid: {', '.join(BUTTON_NAMES)}")
        return bool(fn())

    def get_left_joystick(self) -> Float32[np.ndarray, "2"]:
        """Return left joystick `[x, y]` as `(2,)` float32."""
        return _float32_array(self._xrt.get_left_axis(), (2,), "left joystick")

    def get_right_joystick(self) -> Float32[np.ndarray, "2"]:
        """Return right joystick `[x, y]` as `(2,)` float32."""
        return _float32_array(self._xrt.get_right_axis(), (2,), "right joystick")

    def get_motion_trackers(self) -> dict[str, MotionTrackerFrame]:
        """Return motion-tracker data keyed by serial: `pose` (7,), `velocity` (6,), `acceleration` (6,)."""
        n = self._xrt.num_motion_data_available()
        if n == 0:
            return {}
        poses = _float32_array(self._xrt.get_motion_tracker_pose(), (n, 7), "motion tracker poses")
        velocities = _float32_array(self._xrt.get_motion_tracker_velocity(), (n, 6), "motion tracker velocities")
        accelerations = _float32_array(
            self._xrt.get_motion_tracker_acceleration(), (n, 6), "motion tracker accelerations"
        )
        serials = self._xrt.get_motion_tracker_serial_numbers()
        assert len(serials) == n, f"motion tracker serial count must be {n}, got {len(serials)}"
        result: dict[str, MotionTrackerFrame] = {}
        for i in range(n):
            result[serials[i]] = {
                "pose": poses[i],
                "velocity": velocities[i],
                "acceleration": accelerations[i],
            }
        return result

    def get_timestamp_ns(self) -> int:
        """Return current SDK timestamp in nanoseconds."""
        return int(self._xrt.get_time_stamp_ns())

    def close(self):
        """Shut down the SDK connection."""
        self._xrt.close()
        logger.info("PICO SDK closed")


# ── PICO (24, 7) body pose -> ordered transforms ────────────────────────────
# 90 deg about X: Y-up (SDK) -> Z-up (retarget world).
_Y_UP_TO_Z_UP_ROT: np.ndarray = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
_Y_UP_TO_Z_UP_QUAT_WXYZ: np.ndarray = np.array([0.70710677, 0.70710677, 0.0, 0.0], dtype=np.float32)

# SDK body joint indices (24 SMPL joints) -> SMPL-X canonical names.
PICO_JOINT_NAMES: tuple[str, ...] = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar",
    "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hand", "right_hand",
)  # fmt: skip


def pico_pose_array_to_transforms(
    poses: Float[np.ndarray, "24 7"],
    human_joint_names: Sequence[str],
    use_ankle_as_foot_target: bool = False,
) -> Float32[np.ndarray, "J 7"]:
    """Convert a PICO pose array to ordered Z-up `[J, 7]` transforms.

    >>> poses = np.zeros((24, 7), dtype=np.float32)
    >>> poses[:, 6] = 1.0
    >>> pico_pose_array_to_transforms(poses, ("pelvis", "left_foot")).shape
    (2, 7)
    """
    if poses.shape != (24, 7):
        raise ValueError(f"Expected (24, 7) PICO pose array, got {poses.shape}")
    positions_zup = poses[:, :3].astype(np.float32) @ _Y_UP_TO_Z_UP_ROT.T
    quats_wxyz_yup = poses[:, 3:].astype(np.float32)[:, [3, 0, 1, 2]]
    transforms = np.empty((24, 7), dtype=np.float32)
    transforms[:, :3] = positions_zup
    for idx in range(24):
        rotated = quaternion_multiply(_Y_UP_TO_Z_UP_QUAT_WXYZ, quats_wxyz_yup[idx])
        transforms[idx, 3:] = rotated / np.linalg.norm(rotated)

    if use_ankle_as_foot_target:
        for ankle, foot in (("left_ankle", "left_foot"), ("right_ankle", "right_foot")):
            transforms[PICO_JOINT_NAMES.index(foot)] = transforms[PICO_JOINT_NAMES.index(ankle)]
    return np.ascontiguousarray(transforms[[PICO_JOINT_NAMES.index(name) for name in human_joint_names]])


class PicoSource(Source[BodyFrame]):
    """Live PICO body motion as ordered `[J, 7]` transforms."""

    def __init__(
        self,
        human_joint_names: Sequence[str] = CANONICAL_BODY_JOINTS,
        default_human_height: float = 1.75,
        use_ankle_as_foot_target: bool = False,
    ) -> None:
        self.human_joint_names = tuple(human_joint_names)
        self._human_height = default_human_height
        self._use_ankle = bool(use_ankle_as_foot_target)
        self._latest: BodyFrame | None = None
        self._buttons: dict[str, bool] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frame_idx = 0
        self._pico = Pico()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            frame = (
                BodyFrame(
                    self.human_joint_names,
                    pico_pose_array_to_transforms(
                        self._pico.get_body_poses(),
                        self.human_joint_names,
                        use_ankle_as_foot_target=self._use_ankle,
                    ),
                    self._human_height,
                )
                if self._pico.is_body_available()
                else None
            )
            buttons = {name: self._pico.get_button(name) for name in ("B", "Y")}
            with self._lock:
                if frame is not None:
                    self._latest = frame
                    self._frame_idx += 1
                self._buttons = buttons
            time.sleep(0.005)

    def get_frame(self) -> BodyFrame | None:
        with self._lock:
            return self._latest

    def get_buttons(self) -> dict[str, bool]:
        """Latest B/Y controller button states (empty dict until the first poll)."""
        with self._lock:
            return dict(self._buttons)

    def reset(self) -> None:
        with self._lock:
            self._latest = None
            self._frame_idx = 0

    @property
    def human_height(self) -> float:
        return self._human_height

    @property
    def frame_idx(self) -> int:
        return self._frame_idx

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)
        self._pico.close()
        logger.info("PicoSource closed")


@dataclass
class PicoSourceConfig(SourceConfig[BodyFrame]):
    """Live PICO body-tracking settings.

    >>> PicoSourceConfig().joint_names[0]
    'pelvis'
    """

    human_height: float = 1.75
    use_ankle_as_foot_target: bool = False
    joint_names: tuple[str, ...] = CANONICAL_BODY_JOINTS

    def build(self) -> PicoSource:
        return PicoSource(self.joint_names, self.human_height, self.use_ankle_as_foot_target)
