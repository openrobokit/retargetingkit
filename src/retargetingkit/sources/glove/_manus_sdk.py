"""Private ctypes boundary for the official Manus SDK."""

import ctypes
import os
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from jaxtyping import Float32


_SDK_RETURN_CODE_SUCCESS = 0
_SIDE_LEFT = 1
_SIDE_RIGHT = 2
_HAND_MOTION_AUTO = 4
_AXIS_VIEW_X_FROM_VIEWER = 3
_AXIS_POLARITY_POSITIVE_Z = 6

HAND_JOINTS = 25
SIDE_LEFT = _SIDE_LEFT
SIDE_RIGHT = _SIDE_RIGHT

_MAX_HOST_NAME_LENGTH = 256
_MAX_IP_ADDRESS_LENGTH = 40
_MAX_VERSION_LENGTH = 16


class _ManusVec3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]


class _ManusQuaternion(ctypes.Structure):
    _fields_ = [
        ("w", ctypes.c_float),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
    ]


class _ManusTransform(ctypes.Structure):
    _fields_ = [
        ("position", _ManusVec3),
        ("rotation", _ManusQuaternion),
        ("scale", _ManusVec3),
    ]


class _SkeletonNode(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint32), ("transform", _ManusTransform)]


class _ManusTimestamp(ctypes.Structure):
    _fields_ = [("time", ctypes.c_uint64)]


class _RawSkeletonInfo(ctypes.Structure):
    _fields_ = [
        ("gloveId", ctypes.c_uint32),
        ("nodesCount", ctypes.c_uint32),
        ("publishTime", _ManusTimestamp),
    ]


class _SkeletonStreamInfo(ctypes.Structure):
    _fields_ = [
        ("publishTime", _ManusTimestamp),
        ("skeletonsCount", ctypes.c_uint32),
    ]


class _NodeInfo(ctypes.Structure):
    _fields_ = [
        ("nodeId", ctypes.c_uint32),
        ("parentId", ctypes.c_uint32),
        ("chainType", ctypes.c_int),
        ("side", ctypes.c_int),
        ("fingerJointType", ctypes.c_int),
    ]


class _Version(ctypes.Structure):
    _fields_ = [
        ("major", ctypes.c_uint32),
        ("minor", ctypes.c_uint32),
        ("patch", ctypes.c_uint32),
        ("label", ctypes.c_char * _MAX_VERSION_LENGTH),
        ("sha", ctypes.c_char * _MAX_VERSION_LENGTH),
        ("tag", ctypes.c_char * _MAX_VERSION_LENGTH),
    ]


class _ManusHost(ctypes.Structure):
    _fields_ = [
        ("hostName", ctypes.c_char * _MAX_HOST_NAME_LENGTH),
        ("ipAddress", ctypes.c_char * _MAX_IP_ADDRESS_LENGTH),
        ("manusCoreVersion", _Version),
    ]


class _CoordinateSystemVUH(ctypes.Structure):
    _fields_ = [
        ("view", ctypes.c_int),
        ("up", ctypes.c_int),
        ("handedness", ctypes.c_int),
        ("unitScale", ctypes.c_float),
    ]


_ConnectedCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(_ManusHost))
_DisconnectedCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(_ManusHost))
_LogCallback = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32)
_SkeletonStreamCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(_SkeletonStreamInfo))


@dataclass(frozen=True)
class ManusSdkFrame:
    """Raw typed frame emitted by the Manus SDK boundary."""

    glove_id: int
    publish_time: int
    recv_ns: int
    side_code: int
    positions: Float32[np.ndarray, "25 3"]
    rotations: Float32[np.ndarray, "25 4"]


def resolve_manus_sdk_library_path() -> Path:
    """Resolve the packaged Manus SDK shared library.

    >>> resolve_manus_sdk_library_path().name  # doctest: +SKIP
    'libManusSDK_Integrated.so'
    """
    from manus_sdk_runtime import library_path  # type: ignore[import-not-found]

    return Path(library_path()).resolve()


class ManusClient:
    """Callback-driven client for the official Manus SDK."""

    def __init__(self, library_path: Path) -> None:
        self._dll_directory = None
        if sys.platform == "win32":
            self._dll_directory = os.add_dll_directory(str(library_path.parent))
        self._lib = ctypes.cdll.LoadLibrary(str(library_path))
        self._setup_functions()
        self._python_cb: Callable[[ManusSdkFrame], None] | None = None
        self._lock = threading.Lock()
        self._node_info_cache: dict[int, list[_NodeInfo]] = {}
        self._node_info_lock = threading.Lock()
        self._c_on_connect = _ConnectedCallback(self._on_connect)
        self._c_on_disconnect = _DisconnectedCallback(self._on_disconnect)
        self._c_on_log = _LogCallback(self._on_log)
        self._c_on_skeleton = _SkeletonStreamCallback(self._on_raw_skeleton_stream)

    def _setup_functions(self) -> None:
        lib = self._lib
        lib.CoreSdk_InitializeIntegrated.argtypes = []
        lib.CoreSdk_InitializeIntegrated.restype = ctypes.c_int
        lib.CoreSdk_ShutDown.argtypes = []
        lib.CoreSdk_ShutDown.restype = ctypes.c_int
        lib.CoreSdk_LookForHosts.argtypes = [ctypes.c_uint32, ctypes.c_bool]
        lib.CoreSdk_LookForHosts.restype = ctypes.c_int
        lib.CoreSdk_GetNumberOfAvailableHostsFound.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        lib.CoreSdk_GetNumberOfAvailableHostsFound.restype = ctypes.c_int
        lib.CoreSdk_GetAvailableHostsFound.argtypes = [ctypes.POINTER(_ManusHost), ctypes.c_uint32]
        lib.CoreSdk_GetAvailableHostsFound.restype = ctypes.c_int
        lib.CoreSdk_ConnectToHost.argtypes = [_ManusHost]
        lib.CoreSdk_ConnectToHost.restype = ctypes.c_int
        lib.CoreSdk_InitializeCoordinateSystemWithVUH.argtypes = [_CoordinateSystemVUH, ctypes.c_bool]
        lib.CoreSdk_InitializeCoordinateSystemWithVUH.restype = ctypes.c_int
        lib.CoreSdk_RegisterCallbackForOnConnect.argtypes = [_ConnectedCallback]
        lib.CoreSdk_RegisterCallbackForOnConnect.restype = ctypes.c_int
        lib.CoreSdk_RegisterCallbackForOnDisconnect.argtypes = [_DisconnectedCallback]
        lib.CoreSdk_RegisterCallbackForOnDisconnect.restype = ctypes.c_int
        lib.CoreSdk_RegisterCallbackForOnLog.argtypes = [_LogCallback]
        lib.CoreSdk_RegisterCallbackForOnLog.restype = ctypes.c_int
        lib.CoreSdk_RegisterCallbackForRawSkeletonStream.argtypes = [_SkeletonStreamCallback]
        lib.CoreSdk_RegisterCallbackForRawSkeletonStream.restype = ctypes.c_int
        lib.CoreSdk_SetRawSkeletonHandMotion.argtypes = [ctypes.c_int]
        lib.CoreSdk_SetRawSkeletonHandMotion.restype = ctypes.c_int
        lib.CoreSdk_GetRawSkeletonInfo.argtypes = [ctypes.c_uint32, ctypes.POINTER(_RawSkeletonInfo)]
        lib.CoreSdk_GetRawSkeletonInfo.restype = ctypes.c_int
        lib.CoreSdk_GetRawSkeletonData.argtypes = [ctypes.c_uint32, ctypes.POINTER(_SkeletonNode), ctypes.c_uint32]
        lib.CoreSdk_GetRawSkeletonData.restype = ctypes.c_int
        lib.CoreSdk_GetRawSkeletonNodeCount.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
        lib.CoreSdk_GetRawSkeletonNodeCount.restype = ctypes.c_int
        lib.CoreSdk_GetRawSkeletonNodeInfoArray.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(_NodeInfo),
            ctypes.c_uint32,
        ]
        lib.CoreSdk_GetRawSkeletonNodeInfoArray.restype = ctypes.c_int
        lib.ManusHost_Init.argtypes = [ctypes.POINTER(_ManusHost)]
        lib.ManusHost_Init.restype = None
        lib.CoordinateSystemVUH_Init.argtypes = [ctypes.POINTER(_CoordinateSystemVUH)]
        lib.CoordinateSystemVUH_Init.restype = None

    def _on_connect(self, host_ptr: Any) -> None:
        host = host_ptr.contents
        print(f"[manus] Connected to host: {host.hostName.decode()}", flush=True)
        self._lib.CoreSdk_SetRawSkeletonHandMotion(_HAND_MOTION_AUTO)

    def _on_disconnect(self, host_ptr: Any) -> None:
        host = host_ptr.contents
        print(f"[manus] Disconnected from host: {host.hostName.decode()}", flush=True)

    def _on_log(self, severity: int, message: Any, length: int) -> None:
        levels = {0: "DEBUG", 1: "INFO", 2: "WARN", 3: "ERROR"}
        level = levels.get(severity, "INFO")
        text = message[:length].decode(errors="replace") if message else ""
        print(f"[manus][{level}] {text}", flush=True)

    def _refresh_node_info(self, glove_id: int) -> None:
        node_count = ctypes.c_uint32(0)
        result = self._lib.CoreSdk_GetRawSkeletonNodeCount(ctypes.c_uint32(glove_id), ctypes.byref(node_count))
        if result != _SDK_RETURN_CODE_SUCCESS or node_count.value == 0:
            return
        info_array = (_NodeInfo * node_count.value)()
        result = self._lib.CoreSdk_GetRawSkeletonNodeInfoArray(ctypes.c_uint32(glove_id), info_array, node_count.value)
        if result != _SDK_RETURN_CODE_SUCCESS:
            return
        with self._node_info_lock:
            self._node_info_cache[glove_id] = list(info_array)

    def _on_raw_skeleton_stream(self, stream_info_ptr: Any) -> None:
        with self._lock:
            callback = self._python_cb
        if callback is None:
            return

        stream_info = stream_info_ptr.contents
        for skeleton_index in range(stream_info.skeletonsCount):
            raw_info = _RawSkeletonInfo()
            result = self._lib.CoreSdk_GetRawSkeletonInfo(skeleton_index, ctypes.byref(raw_info))
            if result != _SDK_RETURN_CODE_SUCCESS or raw_info.nodesCount == 0:
                continue
            assert raw_info.nodesCount == HAND_JOINTS, (
                f"Manus node count must be {HAND_JOINTS}, got {raw_info.nodesCount}"
            )

            nodes = (_SkeletonNode * raw_info.nodesCount)()
            result = self._lib.CoreSdk_GetRawSkeletonData(skeleton_index, nodes, raw_info.nodesCount)
            if result != _SDK_RETURN_CODE_SUCCESS:
                continue

            glove_id = raw_info.gloveId
            with self._node_info_lock:
                needs_refresh = glove_id not in self._node_info_cache
            if needs_refresh:
                self._refresh_node_info(glove_id)
            with self._node_info_lock:
                node_info = self._node_info_cache.get(glove_id)

            positions = np.zeros((HAND_JOINTS, 3), dtype=np.float32)
            rotations = np.zeros((HAND_JOINTS, 4), dtype=np.float32)
            side_votes: Counter[int] = Counter()
            for index in range(HAND_JOINTS):
                node = nodes[index]
                positions[index] = (
                    node.transform.position.x,
                    node.transform.position.y,
                    node.transform.position.z,
                )
                rotations[index] = (
                    node.transform.rotation.x,
                    node.transform.rotation.y,
                    node.transform.rotation.z,
                    node.transform.rotation.w,
                )
                if node_info and index < len(node_info):
                    side = node_info[index].side
                    if side in (SIDE_LEFT, SIDE_RIGHT):
                        side_votes[side] += 1

            assert positions.shape == (HAND_JOINTS, 3), f"positions shape must be ({HAND_JOINTS}, 3)"
            assert positions.dtype == np.float32, f"positions dtype must be float32, got {positions.dtype}"
            assert np.isfinite(positions).all(), "positions must be finite"
            assert rotations.shape == (HAND_JOINTS, 4), f"rotations shape must be ({HAND_JOINTS}, 4)"
            assert rotations.dtype == np.float32, f"rotations dtype must be float32, got {rotations.dtype}"
            assert np.isfinite(rotations).all(), "rotations must be finite"

            side_code = side_votes.most_common(1)[0][0] if side_votes else 0
            callback(
                ManusSdkFrame(
                    glove_id=glove_id,
                    publish_time=stream_info.publishTime.time,
                    recv_ns=time.monotonic_ns(),
                    side_code=side_code,
                    positions=positions,
                    rotations=rotations,
                )
            )

    def start(self, callback: Callable[[ManusSdkFrame], None]) -> None:
        """Initialize the SDK and start emitting frames from its callback thread."""
        with self._lock:
            self._python_cb = callback

        result = self._lib.CoreSdk_InitializeIntegrated()
        assert result == _SDK_RETURN_CODE_SUCCESS, f"CoreSdk_InitializeIntegrated failed: {result}"

        self._lib.CoreSdk_RegisterCallbackForOnConnect(self._c_on_connect)
        self._lib.CoreSdk_RegisterCallbackForOnDisconnect(self._c_on_disconnect)
        self._lib.CoreSdk_RegisterCallbackForOnLog(self._c_on_log)
        self._lib.CoreSdk_RegisterCallbackForRawSkeletonStream(self._c_on_skeleton)

        coordinate_system = _CoordinateSystemVUH()
        self._lib.CoordinateSystemVUH_Init(ctypes.byref(coordinate_system))
        coordinate_system.handedness = _SIDE_RIGHT
        coordinate_system.up = _AXIS_POLARITY_POSITIVE_Z
        coordinate_system.view = _AXIS_VIEW_X_FROM_VIEWER
        coordinate_system.unitScale = 1.0
        result = self._lib.CoreSdk_InitializeCoordinateSystemWithVUH(coordinate_system, True)
        assert result == _SDK_RETURN_CODE_SUCCESS, f"CoreSdk_InitializeCoordinateSystemWithVUH failed: {result}"

        result = self._lib.CoreSdk_LookForHosts(2, False)
        assert result == _SDK_RETURN_CODE_SUCCESS, f"CoreSdk_LookForHosts failed: {result}"

        host_count = ctypes.c_uint32(0)
        result = self._lib.CoreSdk_GetNumberOfAvailableHostsFound(ctypes.byref(host_count))
        if result == _SDK_RETURN_CODE_SUCCESS and host_count.value > 0:
            hosts = (_ManusHost * host_count.value)()
            self._lib.CoreSdk_GetAvailableHostsFound(hosts, host_count.value)
            result = self._lib.CoreSdk_ConnectToHost(hosts[0])
            assert result == _SDK_RETURN_CODE_SUCCESS, f"CoreSdk_ConnectToHost failed: {result}"
        else:
            host = _ManusHost()
            self._lib.ManusHost_Init(ctypes.byref(host))
            result = self._lib.CoreSdk_ConnectToHost(host)
            assert result == _SDK_RETURN_CODE_SUCCESS, f"CoreSdk_ConnectToHost (empty) failed: {result}"

        print("[manus] Started successfully", flush=True)

    def stop(self) -> None:
        """Stop the SDK and clear cached node metadata."""
        self._lib.CoreSdk_ShutDown()
        with self._node_info_lock:
            self._node_info_cache.clear()
        print("[manus] Stopped", flush=True)
