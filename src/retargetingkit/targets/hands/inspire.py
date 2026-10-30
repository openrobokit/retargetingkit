"""Inspire Hand 6-DOF serial driver: 115200 baud, 20-byte packets with header `[0xEB, 0x90]`.

Joint order (URDF radians)::

    q[0]: thumb_rotate  (-0.1, 1.3)
    q[1]: thumb_bend    (0.0, 0.5)
    q[2]: index finger   (0.0, 1.7)
    q[3]: middle finger  (0.0, 1.7)
    q[4]: ring finger    (0.0, 1.7)
    q[5]: little finger  (0.0, 1.7)
"""

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from retargetingkit.targets.target import Target, TargetConfig


NUM_JOINTS = 6
JOINT_LIMIT_LOWER = np.array([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
JOINT_LIMIT_UPPER = np.array([1.3, 0.5, 1.7, 1.7, 1.7, 1.7])

# URDF order: [thumb_rotate, thumb_bend, index, middle, ring, little]
# Serial order: [little, ring, middle, index, thumb_bend, thumb_rotate]
_REORDER = [5, 4, 3, 2, 1, 0]

logger = logging.getLogger(__name__)


def urdf_to_serial(q: np.ndarray) -> np.ndarray:
    """Convert 6-DOF URDF joint radians to serial 0-1 normalized values.

    >>> import numpy as np
    >>> urdf_to_serial(np.array([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0]))
    array([1., 1., 1., 1., 1., 1.])
    """
    normalized = (q - JOINT_LIMIT_LOWER) / (JOINT_LIMIT_UPPER - JOINT_LIMIT_LOWER)
    return np.clip(1.0 - normalized[_REORDER], 0.0, 1.0)


def serial_to_urdf(q_serial: np.ndarray) -> np.ndarray:
    """Convert serial 0-1 normalized values to 6-DOF URDF joint radians.

    >>> import numpy as np
    >>> serial_to_urdf(np.ones(6))
    array([-0.1, 0. , 0. , 0. , 0. , 0. ])
    """
    q_serial = np.clip(q_serial, -0.1, 1.1)
    inverted = 1.0 - q_serial[_REORDER]
    return inverted * (JOINT_LIMIT_UPPER - JOINT_LIMIT_LOWER) + JOINT_LIMIT_LOWER


def _checksum(data: bytes | bytearray) -> int:
    return sum(data[2:-1]) & 0xFF


@dataclass
class InspireHandConfig(TargetConfig):
    """Inspire Hand serial settings.

    >>> InspireHandConfig("/dev/ttyUSB0", 1, 115200).serial_port
    '/dev/ttyUSB0'
    """

    serial_port: str
    hand_id: int
    baudrate: int

    def build(self) -> "InspireHand":
        """Connect and return an Inspire Hand target.

        >>> InspireHandConfig("/dev/ttyUSB0", 1, 115200).build()  # doctest: +SKIP
        """
        return InspireHand(self.serial_port, self.hand_id, self.baudrate)


class InspireHand(Target):
    """Non-blocking serial interface for Inspire Hand (6-DOF); a background thread reads positions and writes commands.

    >>> # InspireHand("/dev/ttyUSB0", 1, 115200)  # doctest: +SKIP
    """

    JOINT_NAMES = (
        "thumb_proximal_yaw_joint",
        "thumb_proximal_pitch_joint",
        "index_proximal_joint",
        "middle_proximal_joint",
        "ring_proximal_joint",
        "pinky_proximal_joint",
    )

    def __init__(self, serial_port: str, hand_id: int, baudrate: int):
        import serial

        self._ser = serial.Serial(port=serial_port, baudrate=baudrate, timeout=0.05, write_timeout=0.1, exclusive=True)
        assert self._ser.is_open, f"Failed to open {serial_port}"
        self._id = hand_id
        self._qpos = np.zeros(NUM_JOINTS)
        self._qcmd = np.zeros(NUM_JOINTS)
        self._first_cmd = threading.Event()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Actuated joint names, in set_qpos/get_qpos (URDF) order."""
        return self.JOINT_NAMES

    def set_qpos(self, qpos: np.ndarray) -> None:
        """Set target joint positions in URDF radians (6-DOF)."""
        self._qcmd[:] = urdf_to_serial(qpos)
        self._first_cmd.set()

    def get_qpos(self) -> np.ndarray:
        """Get current joint positions in URDF radians (6-DOF)."""
        return serial_to_urdf(self._qpos.copy())

    def close(self) -> None:
        """Stop background thread and close serial port."""
        self._stop_event.set()
        self._thread.join(timeout=1.0)

    def clear_error(self) -> None:
        """Send error-clear command to the hand."""
        cmd = bytearray([0xEB, 0x90, self._id, 0x04, 0x12, 0xEC, 0x03, 0x01, 0x00])
        cmd[-1] = _checksum(cmd)
        self._send_recv(cmd)

    # -- internals --

    def _send_recv(self, cmd: bytes | bytearray, recv_len: int = 9) -> bytes:
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()
        time.sleep(0.005)
        self._ser.write(cmd)
        return self._ser.read(recv_len)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            qpos = self._read_position()
            if qpos is not None:
                self._qpos[:] = qpos
            if self._first_cmd.is_set():
                self._write_position(self._qcmd)
        self._ser.close()

    def _write_position(self, q: np.ndarray) -> None:
        q_int16 = np.clip((q * 1000).astype(np.int16), 0, 1000)
        cmd = bytearray(20)
        cmd[0:7] = [0xEB, 0x90, self._id, 0x0F, 0x12, 0xCE, 0x05]
        for i in range(6):
            cmd[7 + 2 * i] = q_int16[i] & 0xFF
            cmd[8 + 2 * i] = (q_int16[i] >> 8) & 0xFF
        cmd[19] = _checksum(cmd)
        resp = self._send_recv(cmd)
        if len(resp) != 9:
            logger.warning("SetPosition failed, response len: %d", len(resp))

    def _read_position(self) -> np.ndarray | None:
        cmd = bytearray([0xEB, 0x90, self._id, 0x04, 0x11, 0x0A, 0x06, 0x0C, 0x00])
        cmd[-1] = _checksum(cmd)
        resp = self._send_recv(cmd, 20)
        if len(resp) != 20:
            return None
        if resp[0] != 0x90 or resp[1] != 0xEB or resp[2] != self._id:
            logger.warning("GetPosition bad header: %s", resp[:3].hex())
            return None
        if resp[-1] != _checksum(resp):
            logger.warning("GetPosition bad checksum: got %d, want %d", resp[-1], _checksum(resp))
        q = np.zeros(6)
        for i in range(6):
            q[i] = int.from_bytes(resp[7 + 2 * i : 9 + 2 * i], "little", signed=False) / 1000.0
        return q
