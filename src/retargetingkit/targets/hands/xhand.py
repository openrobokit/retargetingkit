"""XHand 12-DOF dexterous hand driver: RS485 at 83 Hz (3 Mbaud) via the vendor `xhand_controller` SDK.

Joint order (URDF radians, 12 joints)::

    q[ 0]: right_hand_thumb_bend_joint   ( 0.000,  1.832)
    q[ 1]: right_hand_thumb_rota_joint1  (-0.698,  1.745)
    q[ 2]: right_hand_thumb_rota_joint2  ( 0.000,  1.745)
    q[ 3]: right_hand_index_bend_joint   (-0.174,  0.174)
    q[ 4]: right_hand_index_joint1       ( 0.000,  1.919)
    q[ 5]: right_hand_index_joint2       ( 0.000,  1.919)
    q[ 6]: right_hand_mid_joint1         ( 0.000,  1.919)
    q[ 7]: right_hand_mid_joint2         ( 0.000,  1.919)
    q[ 8]: right_hand_ring_joint1        ( 0.000,  1.919)
    q[ 9]: right_hand_ring_joint2        ( 0.000,  1.919)
    q[10]: right_hand_pinky_joint1       ( 0.000,  1.919)
    q[11]: right_hand_pinky_joint2       ( 0.000,  1.919)
"""

import ctypes
import importlib.util
import multiprocessing as mp
import time
from dataclasses import dataclass
from multiprocessing.sharedctypes import Synchronized, SynchronizedArray

import numpy as np
from jaxtyping import Float, Float64

from retargetingkit.targets.target import Target, TargetConfig


_sdk_initialized = False

NUM_JOINTS = 12

JOINT_NAMES = (
    "right_hand_thumb_bend_joint",
    "right_hand_thumb_rota_joint1",
    "right_hand_thumb_rota_joint2",
    "right_hand_index_bend_joint",
    "right_hand_index_joint1",
    "right_hand_index_joint2",
    "right_hand_mid_joint1",
    "right_hand_mid_joint2",
    "right_hand_ring_joint1",
    "right_hand_ring_joint2",
    "right_hand_pinky_joint1",
    "right_hand_pinky_joint2",
)

JOINT_LIMIT_LOWER = np.array(
    [0.0, -0.698, 0.0, -0.174, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
JOINT_LIMIT_UPPER = np.array(
    [1.832, 1.745, 1.745, 0.174, 1.919, 1.919, 1.919, 1.919, 1.919, 1.919, 1.919, 1.919],
    dtype=np.float64,
)


def _sdk_ready() -> bool:
    return importlib.util.find_spec("xhand_controller") is not None


def initialize_xhand_sdk() -> None:
    """Check once per process that the XHand SDK is installed.

    >>> # initialize_xhand_sdk()  # doctest: +SKIP
    """
    global _sdk_initialized
    if _sdk_initialized:
        return
    assert _sdk_ready(), "Install the vendor xhand_controller wheel"
    _sdk_initialized = True


@dataclass
class XHandConfig(TargetConfig):
    """XHand controller configuration."""

    serial_port: str
    """Serial port for RS485 communication."""

    protocol: str = "RS485"
    """Communication protocol ('RS485' or 'EtherCAT')."""

    baud_rate: int = 3_000_000
    """Baud rate for RS485 serial."""

    dt: float = 1.0 / 83
    """Servo loop timestep (matches 83 Hz firmware rate)."""

    kp: float = 90.0
    """Default position-loop proportional gain."""

    ki: float = 0.0
    """Default position-loop integral gain."""

    kd: float = 0.0
    """Default position-loop derivative gain."""

    tor_max: float = 80.0
    """Default maximum torque."""

    def build(self) -> "XHand":
        xhand = XHand(self)
        xhand.start()
        return xhand


class XHand(Target):
    """XHand 12-DOF controller; a ~83 Hz child process drives the SDK, so `set_qpos` and `get_qpos` never block.

    >>> # XHand(XHandConfig("/dev/ttyUSB0"))  # doctest: +SKIP
    """

    def __init__(self, config: XHandConfig):
        self._config = config
        self._target = mp.Array(ctypes.c_double, NUM_JOINTS)
        self._current = mp.Array(ctypes.c_double, NUM_JOINTS)
        self._running = mp.Value(ctypes.c_bool, True)
        self._ready = mp.Value(ctypes.c_bool, False)
        self._process: mp.Process | None = None

    def start(self):
        """Connect to hardware and start the background servo process."""
        initialize_xhand_sdk()
        self._process = mp.Process(
            target=XHand._servo_loop,
            args=(
                self._config,
                self._target,
                self._current,
                self._running,
                self._ready,
            ),
            daemon=True,
        )
        self._process.start()

        t0 = time.monotonic()
        while not self._ready.value:
            if time.monotonic() - t0 > 10.0:
                raise TimeoutError("XHand background process did not become ready within 10 s")
            time.sleep(0.01)

    def close(self):
        """Signal the background process to stop and wait for it."""
        self._running.value = False
        if self._process is not None:
            self._process.join(timeout=5.0)

    def set_qpos(self, qpos: Float[np.ndarray, "12"]) -> None:
        """Set target joint positions (12-DOF, URDF radians)."""
        qpos = np.asarray(qpos, dtype=np.float64)
        assert qpos.shape == (NUM_JOINTS,), f"qpos shape must be ({NUM_JOINTS},), got {qpos.shape}"
        assert np.isfinite(qpos).all(), "qpos must be finite"
        self._target[:] = qpos

    def get_qpos(self) -> Float64[np.ndarray, "12"]:
        """Read latest joint positions (12-DOF, URDF radians)."""
        qpos = np.array(self._current[:], dtype=np.float64)
        assert qpos.shape == (NUM_JOINTS,), f"qpos shape must be ({NUM_JOINTS},), got {qpos.shape}"
        assert qpos.dtype == np.float64, f"qpos dtype must be float64, got {qpos.dtype}"
        assert np.isfinite(qpos).all(), "qpos must be finite"
        return qpos

    @staticmethod
    def _servo_loop(
        config: XHandConfig,
        target: SynchronizedArray,
        current: SynchronizedArray,
        running: Synchronized,
        ready: Synchronized,
    ) -> None:
        initialize_xhand_sdk()
        from xhand_controller import xhand_control  # type: ignore[import-unresolved]

        device = xhand_control.XHandControl()

        if config.protocol == "RS485":
            rsp = device.open_serial(config.serial_port, config.baud_rate)
            if rsp.error_code != 0:
                raise ConnectionError(
                    f"XHand RS485 open failed: code={rsp.error_code}, {rsp.error_message}",
                )
        elif config.protocol == "EtherCAT":
            devices = device.enumerate_devices("EtherCAT")
            if not devices:
                raise ConnectionError("No XHand EtherCAT devices found")
            rsp = device.open_ethercat(devices[0])
            if rsp.error_code != 0:
                raise ConnectionError(
                    f"XHand EtherCAT open failed: code={rsp.error_code}, {rsp.error_message}",
                )
        else:
            raise ValueError(f"Unknown protocol: {config.protocol}")

        hands = device.list_hands_id()
        if not hands:
            raise ConnectionError("No XHand devices found after opening port")
        hand_id = hands[0]

        time.sleep(0.5)
        err, state = device.read_state(hand_id, True)
        if err.error_code != 0:
            raise RuntimeError(
                f"XHand state read failed: code={err.error_code}, {err.error_message}",
            )
        qpos = np.asarray([state.finger_state[i].position for i in range(NUM_JOINTS)])
        assert qpos.shape == (NUM_JOINTS,), f"qpos shape must be ({NUM_JOINTS},), got {qpos.shape}"
        assert qpos.dtype == np.float64, f"qpos dtype must be float64, got {qpos.dtype}"
        assert np.isfinite(qpos).all(), "qpos must be finite"

        cmd = xhand_control.HandCommand_t()
        for i in range(NUM_JOINTS):
            cmd.finger_command[i].id = i
            cmd.finger_command[i].kp = int(config.kp)
            cmd.finger_command[i].ki = int(config.ki)
            cmd.finger_command[i].kd = int(config.kd)
            cmd.finger_command[i].position = float(np.clip(qpos[i], JOINT_LIMIT_LOWER[i], JOINT_LIMIT_UPPER[i]))
            cmd.finger_command[i].tor_max = int(config.tor_max)
            cmd.finger_command[i].mode = 3

        device.send_command(hand_id, cmd)
        current[:] = qpos
        target[:] = qpos
        ready.value = True

        while running.value:
            loop_t0 = time.perf_counter()

            tgt = np.array(target[:], dtype=np.float64)

            for i in range(NUM_JOINTS):
                cmd.finger_command[i].position = float(np.clip(tgt[i], JOINT_LIMIT_LOWER[i], JOINT_LIMIT_UPPER[i]))

            device.send_command(hand_id, cmd)

            err, state = device.read_state(hand_id, False)
            if err.error_code != 0:
                raise RuntimeError(
                    f"XHand state read failed: code={err.error_code}, {err.error_message}",
                )
            qpos = np.asarray([state.finger_state[i].position for i in range(NUM_JOINTS)])
            assert qpos.shape == (NUM_JOINTS,), f"qpos shape must be ({NUM_JOINTS},), got {qpos.shape}"
            assert qpos.dtype == np.float64, f"qpos dtype must be float64, got {qpos.dtype}"
            assert np.isfinite(qpos).all(), "qpos must be finite"
            current[:] = qpos

            elapsed = time.perf_counter() - loop_t0
            time.sleep(max(0.0, config.dt - elapsed))

        for i in range(NUM_JOINTS):
            cmd.finger_command[i].kp = 0
            cmd.finger_command[i].ki = 0
            cmd.finger_command[i].kd = 0
            cmd.finger_command[i].tor_max = 0
            cmd.finger_command[i].mode = 0

        device.send_command(hand_id, cmd)
        time.sleep(0.1)
        device.close_device()
