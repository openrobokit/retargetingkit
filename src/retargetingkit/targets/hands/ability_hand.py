"""Ability Hand servo controller using multiprocessing."""

import ctypes
import enum
import multiprocessing as mp
import struct
import time
from dataclasses import dataclass
from multiprocessing.sharedctypes import Synchronized, SynchronizedArray

import numpy as np

from retargetingkit.targets.target import Target, TargetConfig


NUM_JOINTS = 6
_ADDR = 0x50


class ControlMode(enum.IntEnum):
    """Control mode for the Ability Hand servo loop."""

    POS_RAMP = 0
    """Position command, open-loop ramp (default)."""
    VEL_RAMP = 1
    """Velocity command, open-loop ramp."""


@dataclass
class AbilityHandConfig(TargetConfig):
    """Ability Hand controller configuration."""

    port: str
    """Serial port path (e.g. /dev/serial/by-id/...)."""
    dt: float = 1.0 / 160
    """Servo loop timestep in seconds."""
    mode: ControlMode = ControlMode.POS_RAMP
    """Control mode."""
    ramp_deg_per_s: float = 60.0
    """Position ramp rate to avoid firmware current-limiting stalls."""

    def build(self) -> "AbilityHand":
        ability = AbilityHand(self)
        ability.start()
        return ability


def _ppp_stuff(data: bytearray) -> bytes:
    """PPP-frame a message for serial transmission."""
    out = bytearray([0x7E])
    for b in data:
        if b in (0x7E, 0x7D):
            out.append(0x7D)
            out.append(b ^ 0x20)
        else:
            out.append(b)
    out.append(0x7E)
    return bytes(out)


def _ppp_unstuff(framed: bytes) -> bytes:
    """Remove PPP framing from a complete frame."""
    if len(framed) < 2 or framed[0] != 0x7E or framed[-1] != 0x7E:
        return b""
    out = bytearray()
    esc = False
    for b in framed[1:-1]:
        if esc:
            out.append(b ^ 0x20)
            esc = False
        elif b == 0x7D:
            esc = True
        else:
            out.append(b)
    return bytes(out)


def _build_cmd(values_deg: np.ndarray, cmd_mode: int = 0x10) -> bytearray:
    """Build a position (0x10) or velocity (0x20) command for the hand."""
    msg = bytearray([_ADDR, cmd_mode + 1])
    scale = 32767.0 / 150.0 if cmd_mode == 0x10 else 32767.0 / 3000.0
    for v in values_deg:
        msg.extend(struct.pack("<h", int(np.clip(v * scale, -32767, 32767))))
    msg.append((-sum(msg)) & 0xFF)
    return msg


def _parse_positions(payload: bytes) -> np.ndarray | None:
    """Parse 6 joint positions (degrees) from a hand response."""
    buf = np.frombuffer(payload, dtype=np.uint8)
    variant = (buf[0] & 0x0F) + 1
    expected = 38 if variant == 3 else 72
    if buf.size != expected:
        return None
    if np.uint8(-np.sum(np.int8(buf[:-1]))) != buf[-1]:
        return None
    stride = 2 if variant == 3 else 4
    pos = np.zeros(NUM_JOINTS)
    for i in range(NUM_JOINTS):
        pos[i] = struct.unpack("<h", buf[1 + i * stride : 3 + i * stride].tobytes())[0] * 150.0 / 32767.0
    if np.any(np.abs(pos) > 160.0):
        return None
    return pos


class AbilityHand(Target):
    """Ability Hand controller with 80 Hz servo loop in a child process.

    >>> # AbilityHand(AbilityHandConfig("/dev/ttyUSB0"))  # doctest: +SKIP
    """

    def __init__(self, config: AbilityHandConfig):
        self._config = config
        self._target = mp.Array(ctypes.c_double, NUM_JOINTS)
        self._current = mp.Array(ctypes.c_double, NUM_JOINTS)
        self._running = mp.Value(ctypes.c_bool, True)
        self._ready = mp.Value(ctypes.c_bool, False)
        self._process: mp.Process | None = None

    def start(self):
        self._process = mp.Process(
            target=AbilityHand._servo_loop,
            args=(self._config, self._target, self._current, self._running, self._ready),
            daemon=True,
        )
        self._process.start()
        while not self._ready.value:
            time.sleep(0.01)

    def close(self):
        self._running.value = False
        if self._process:
            self._process.join(timeout=5.0)

    def set_qpos(self, qpos: np.ndarray):
        self._target[:] = qpos

    def get_qpos(self) -> np.ndarray:
        return np.array(self._current[:])

    @staticmethod
    def _servo_loop(
        config: AbilityHandConfig,
        target: SynchronizedArray,
        current: SynchronizedArray,
        running: Synchronized,
        ready: Synchronized,
    ) -> None:
        import serial  # type: ignore[import-unresolved]

        ser = serial.Serial(config.port, 460800, timeout=0, write_timeout=0)
        ser.reset_input_buffer()
        time.sleep(0.1)

        stuff_buf = bytearray()
        qpos_deg = np.zeros(NUM_JOINTS)

        # Sync: send a few commands to read current position
        for _ in range(5):
            ser.write(_ppp_stuff(_build_cmd(np.zeros(NUM_JOINTS))))
            time.sleep(config.dt)
            for b in ser.read(512):
                stuff_buf.append(b)
                if b == 0x7E and len(stuff_buf) > 1:
                    payload = _ppp_unstuff(bytes(stuff_buf))
                    stuff_buf = bytearray([0x7E])
                    if payload:
                        pos = _parse_positions(payload)
                        if pos is not None:
                            qpos_deg = pos
        stuff_buf = bytearray()

        cmd_deg = qpos_deg.copy()
        qpos_rad = np.deg2rad(qpos_deg)
        target[:] = qpos_rad
        current[:] = qpos_rad
        ready.value = True

        max_step = config.ramp_deg_per_s * config.dt

        while running.value:
            target_deg = np.rad2deg(np.array(target[:]))
            delta = np.clip(target_deg - cmd_deg, -max_step, max_step)
            cmd_deg += delta

            if config.mode == ControlMode.POS_RAMP:
                ser.write(_ppp_stuff(_build_cmd(cmd_deg)))
            else:
                ser.write(_ppp_stuff(_build_cmd(delta / config.dt, cmd_mode=0x20)))

            for b in ser.read(512):
                stuff_buf.append(b)
                if b == 0x7E and len(stuff_buf) > 1:
                    payload = _ppp_unstuff(bytes(stuff_buf))
                    stuff_buf = bytearray([0x7E])
                    if payload:
                        pos = _parse_positions(payload)
                        if pos is not None:
                            qpos_deg = pos
                            qpos_rad = np.deg2rad(qpos_deg)
                            current[:] = qpos_rad

            time.sleep(config.dt)

        if config.mode == ControlMode.VEL_RAMP:
            ser.write(_ppp_stuff(_build_cmd(np.zeros(NUM_JOINTS), cmd_mode=0x20)))
        ser.close()
