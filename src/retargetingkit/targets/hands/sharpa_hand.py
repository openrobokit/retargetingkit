"""Sharpa Wave 22-DOF hand driver (SDK at `~/.retargetingkit/sharpa-sdk/`); tactile channels 0-4 right, 5-9 left, thumb first.

Joint order (URDF radians, 22 joints)::

    q[ 0]: right_thumb_CMC_FE    (-0.1745, 1.9199)
    q[ 1]: right_thumb_CMC_AA    (-0.3491, 0.3491)
    q[ 2]: right_thumb_MCP_FE    (-0.5236, 1.3963)
    q[ 3]: right_thumb_MCP_AA    (-0.3491, 0.3491)
    q[ 4]: right_thumb_IP        ( 0.0000, 1.7453)
    q[ 5]: right_index_MCP_FE    (-0.1745, 1.5708)
    q[ 6]: right_index_MCP_AA    (-0.3491, 0.3491)
    q[ 7]: right_index_PIP       ( 0.0000, 1.7453)
    q[ 8]: right_index_DIP       ( 0.0000, 1.3963)
    q[ 9]: right_middle_MCP_FE   (-0.1745, 1.5708)
    q[10]: right_middle_MCP_AA   (-0.3491, 0.3491)
    q[11]: right_middle_PIP      ( 0.0000, 1.7453)
    q[12]: right_middle_DIP      ( 0.0000, 1.3963)
    q[13]: right_ring_MCP_FE     (-0.1745, 1.5708)
    q[14]: right_ring_MCP_AA     (-0.3491, 0.3491)
    q[15]: right_ring_PIP        ( 0.0000, 1.7453)
    q[16]: right_ring_DIP        ( 0.0000, 1.3963)
    q[17]: right_pinky_CMC       ( 0.0000, 0.2618)
    q[18]: right_pinky_MCP_FE    (-0.1745, 1.5708)
    q[19]: right_pinky_MCP_AA    (-0.3491, 0.3491)
    q[20]: right_pinky_PIP       ( 0.0000, 1.7453)
    q[21]: right_pinky_DIP       ( 0.0000, 1.3963)
"""

import os
import sys
import sysconfig
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from jaxtyping import Float, Float64

from retargetingkit.assets import RETARGETINGKIT_HOME
from retargetingkit.targets.target import Target, TargetConfig


NUM_JOINTS = 22
NUM_TACTILE_CHANNELS = 5

JOINT_NAMES = (
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
)

JOINT_LIMIT_LOWER = np.radians(
    [-10, -20, -30, -20, 0, -10, -20, 0, 0, -10, -20, 0, 0, -10, -20, 0, 0, 0, -10, -20, 0, 0]
)
JOINT_LIMIT_UPPER = np.radians(
    [110, 20, 80, 20, 100, 90, 20, 100, 80, 90, 20, 100, 80, 90, 20, 100, 80, 15, 90, 20, 100, 80]
)

_SDK_PATH = RETARGETINGKIT_HOME / "sharpa-sdk"
_sdk_initialized = False


def _sdk_ready() -> bool:
    ext_suffix = str(sysconfig.get_config_var("EXT_SUFFIX"))
    return (
        (_SDK_PATH / "lib" / "libsharpa-wave-sdk.so").is_file()
        and (_SDK_PATH / "lib" / "libSharpaWaveSDKWrapper.so").is_file()
        and (_SDK_PATH / "python" / "sharpa" / "__init__.py").is_file()
        and (_SDK_PATH / "python" / "sharpa" / f"sharpa{ext_suffix}").is_file()
    )


def _init_sdk() -> None:
    global _sdk_initialized
    if _sdk_initialized:
        return
    assert _sdk_ready(), f"Install the Sharpa SDK at {_SDK_PATH}"
    lib_dir = str(_SDK_PATH / "lib")
    python_lib_dir = str(Path(sysconfig.get_config_var("LIBDIR")))
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    ld_entries = current_ld.split(":") if current_ld else []
    if python_lib_dir not in ld_entries:
        current_ld = python_lib_dir + (":" + current_ld if current_ld else "")
        ld_entries.insert(0, python_lib_dir)
    if lib_dir not in ld_entries:
        current_ld = lib_dir + (":" + current_ld if current_ld else "")
    os.environ["LD_LIBRARY_PATH"] = current_ld
    sdk_python_dir = str(_SDK_PATH / "python")
    if sdk_python_dir not in sys.path:
        sys.path.insert(0, sdk_python_dir)
    _sdk_initialized = True


class SharpaHand(Target):
    """Sharpa Wave 22-DOF hand controller via the vendor SDK: position, velocity, torque control and tactile sensing.

    >>> # SharpaHand()  # doctest: +SKIP
    """

    def __init__(
        self,
        speed_coeff: float = 0.3,
        current_coeff: float = 0.3,
        device_sn: str | None = None,
        control_mode: str = "position",
    ):
        _init_sdk()
        from sharpa import ControlMode, ControlSource, SharpaWaveManager  # type: ignore[import-not-found]

        self._manager = SharpaWaveManager.get_instance()
        self._ControlMode = ControlMode

        devices = []
        while device_sn is None or device_sn not in devices:
            devices = self._manager.get_all_device_sn()
            if device_sn is None and devices:
                device_sn = devices[0]
            time.sleep(1)

        self._hand = self._manager.connect(device_sn)
        assert self._hand is not None, f"Failed to connect to {device_sn}"

        self.set_control_mode(control_mode)
        error = self._hand.set_speed_coeff(speed_coeff)
        assert error.code == 0, f"set_speed_coeff failed: {error.message}"
        error = self._hand.set_current_coeff(current_coeff)
        assert error.code == 0, f"set_current_coeff failed: {error.message}"
        error = self._hand.set_control_source(ControlSource.SDK)
        assert error.code == 0, f"set_control_source failed: {error.message}"

        self._hand.start()
        time.sleep(0.5)

    # ---- Joint control ----

    def set_qpos(self, qpos: Float[np.ndarray, "22"], interpolate: bool = True) -> None:
        """Set 22-DOF joint positions in radians."""
        qpos = np.asarray(qpos, dtype=np.float64)
        assert qpos.shape == (NUM_JOINTS,), f"qpos shape must be ({NUM_JOINTS},), got {qpos.shape}"
        assert qpos.dtype == np.float64, f"qpos dtype must be float64, got {qpos.dtype}"
        assert np.isfinite(qpos).all(), "qpos must be finite"
        error = self._hand.set_joint_position(qpos.tolist(), interpolate)
        assert error.code == 0, f"set_joint_position failed: {error.message}"

    def get_qpos(self) -> Float64[np.ndarray, "22"]:
        """Get 22-DOF joint positions in radians."""
        error, qpos = self._hand.get_joint_position_rad()
        assert error.code == 0, f"get_joint_position_rad failed: {error.message}"
        qpos = np.asarray(qpos)
        assert qpos.shape == (NUM_JOINTS,), f"qpos shape must be ({NUM_JOINTS},), got {qpos.shape}"
        assert qpos.dtype == np.float64, f"qpos dtype must be float64, got {qpos.dtype}"
        assert np.isfinite(qpos).all(), "qpos must be finite"
        return qpos

    def get_state(self) -> tuple[Float64[np.ndarray, "22"], Float64[np.ndarray, "22"], Float64[np.ndarray, "22"]]:
        """Get full joint state: (positions_rad, velocities, torques), all shape (22,)."""
        state = self._hand.get_states()
        values = (np.asarray(state.angles), np.asarray(state.velocities), np.asarray(state.torques))
        for name, value in zip(("angles", "velocities", "torques"), values):
            assert value.shape == (NUM_JOINTS,), f"{name} shape must be ({NUM_JOINTS},), got {value.shape}"
            assert value.dtype == np.float64, f"{name} dtype must be float64, got {value.dtype}"
            assert np.isfinite(value).all(), f"{name} must be finite"
        return values

    # ---- Motor settings ----

    def set_control_mode(self, mode: str):
        """Set control mode: 'position', 'floating', 'zero_force', 'mit', or 'admittance'."""
        mode_map = {
            "position": self._ControlMode.POSITION,
            "floating": self._ControlMode.FLOATING,
            "zero_force": self._ControlMode.ZERO_FORCE,
            "mit": self._ControlMode.MIT,
            "admittance": self._ControlMode.ADMITTANCE,
        }
        error = self._hand.set_control_mode(mode_map[mode])
        assert error.code == 0, f"set_control_mode failed: {error.message}"

    def set_speed_coeff(self, coeff: float):
        """Set speed coefficient (0.0 to 1.0)."""
        error = self._hand.set_speed_coeff(coeff)
        assert error.code == 0, f"set_speed_coeff failed: {error.message}"

    def get_speed_coeff(self) -> float:
        """Get current speed coefficient."""
        error, coeff = self._hand.get_speed_coeff()
        assert error.code == 0, f"get_speed_coeff failed: {error.message}"
        return coeff

    def set_current_coeff(self, coeff: float):
        """Set current/force coefficient (0.0 to 1.0)."""
        error = self._hand.set_current_coeff(coeff)
        assert error.code == 0, f"set_current_coeff failed: {error.message}"

    def get_current_coeff(self) -> float:
        """Get current/force coefficient."""
        error, coeff = self._hand.get_current_coeff()
        assert error.code == 0, f"get_current_coeff failed: {error.message}"
        return coeff

    # ---- Tactile ----

    def fetch_tactile_frame(self, channel: int, timeout: float = -1) -> dict | None:
        """Fetch one tactile frame for a finger channel.

        Args:
            channel: Finger index, thumb to pinky: 0-4 on the right hand, 5-9 on the left.
            timeout: Max wait in seconds. -1 for blocking.

        Returns:
            Dict keyed `ts` (timestamp) and `content` (a dict of RAW, DEFORM, and F6 arrays).
        """
        return self._hand.fetch_tactile_frame(channel, timeout)

    def set_tactile_callback(self, callback: Callable):
        """Register a callback for each tactile frame dict: `channel`, `ts`, `content` (RAW/DEFORM/F6), and `shape`."""
        self._hand.set_tactile_callback(callback)

    def calib_tactile(self, num_frames: int = 20, max_retry: int = 10) -> bool:
        """Tare (zero) all tactile sensors. Returns True on success."""
        return self._hand.calib_tactile(num_frames, max_retry)

    def is_tactile_ready(self) -> bool:
        """Check if tactile subsystem is ready."""
        return self._hand.is_tactile_ready()

    def tactile_summary(self) -> str:
        """Get tactile packet statistics as a JSON string."""
        return self._hand.tactile_summary()

    # ---- Device info ----

    def get_device_info(self):
        """Get device info (sn, firmware_version, hand_side, etc.)."""
        return self._hand.get_device_info()

    def get_fault_code(self) -> dict:
        """Get hardware/software fault codes. Returns {FaultCode: [joint_indices]}."""
        return self._hand.get_fault_code()

    def reboot(self):
        """Reboot the device. Must reconnect after."""
        error = self._hand.reboot()
        assert error.code == 0, f"reboot failed: {error.message}"

    # ---- Lifecycle ----

    def close(self):
        """Stop control and disconnect."""
        self._hand.stop()
        self._manager.disconnect_all()


@dataclass
class SharpaHandConfig(TargetConfig):
    """Sharpa Wave connection settings."""

    speed_coeff: float = 0.3
    """Joint speed coefficient (0-1)."""
    current_coeff: float = 0.3
    """Joint current coefficient (0-1)."""
    device_sn: str | None = None
    """Serial number; the first discovered hand is used when unset."""

    def build(self) -> SharpaHand:
        return SharpaHand(self.speed_coeff, self.current_coeff, self.device_sn)
