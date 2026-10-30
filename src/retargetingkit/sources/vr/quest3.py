"""Quest3 controller poses via NVIDIA Isaac Teleop; the SDK loads lazily so imports need no running OpenXR runtime."""

import os
import shlex
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from retargetingkit.sources.frames import Frame
from retargetingkit.sources.source import Source, SourceConfig
from robokit.xform.numpy import quaternion_to_matrix, rot_tl_to_tf_mat


LEFT = "left"
RIGHT = "right"
SIDES = (LEFT, RIGHT)


@dataclass
class Quest3Config(SourceConfig["Quest3Frame"]):
    """Quest3 Isaac Teleop stream configuration."""

    app_name: str = "RetargetingKitQuest3"
    sides: tuple[str, ...] = (RIGHT,)
    verbose: bool = True
    cloudxr_env_path: str = "~/.cloudxr/run/cloudxr.env"

    def build(self) -> "Quest3":
        """Start the Isaac Teleop session.

        >>> Quest3Config().build()  # doctest: +SKIP
        """
        return Quest3(self)


@dataclass(frozen=True)
class Quest3ControllerFrame:
    """Tracked state for one Quest3 controller."""

    pose: np.ndarray
    """Controller pose as a `(4, 4)` transform matrix."""
    primary_click: bool
    secondary_click: bool


@dataclass(frozen=True)
class Quest3Frame(Frame):
    """One Quest3 sample. Untracked configured controllers are absent."""

    timestamp: float
    timestamp_ns: int
    controllers: dict[str, Quest3ControllerFrame]


def _load_cloudxr_env(path: str) -> None:
    env_path = Path(path).expanduser()
    for line in env_path.read_text().splitlines():
        parts = shlex.split(line, comments=True)
        if len(parts) == 2 and parts[0] == "export":
            key, value = parts[1].split("=", 1)
            os.environ[key] = value


class Quest3(Source[Quest3Frame]):
    """Quest3 controller source backed by Isaac Teleop; run `python -m isaacteleop.cloudxr` before constructing it."""

    def __init__(self, config: Quest3Config) -> None:
        self._config = config
        assert len(config.sides) > 0, "Expected at least one controller side"
        for side in config.sides:
            assert side in SIDES, f"Unknown Quest3 controller side: {side}"
        _load_cloudxr_env(config.cloudxr_env_path)

        from isaacteleop.retargeting_engine.deviceio_source_nodes import (  # type: ignore[import-not-found]
            ControllersSource,
        )
        from isaacteleop.retargeting_engine.interface import OutputCombiner  # type: ignore[import-not-found]
        from isaacteleop.retargeting_engine.tensor_types import ControllerInputIndex  # type: ignore[import-not-found]
        from isaacteleop.teleop_session_manager import (  # type: ignore[import-not-found]
            TeleopSession,
            TeleopSessionConfig,
        )

        controllers = ControllersSource(name="controllers")
        outputs = {}
        for side in config.sides:
            input_device = ControllersSource.LEFT if side == LEFT else ControllersSource.RIGHT
            outputs[f"{side}_controller"] = controllers.output(input_device)

        self._session = TeleopSession(
            TeleopSessionConfig(
                app_name=config.app_name,
                trackers=[],
                pipeline=OutputCombiner(outputs),
                verbose=config.verbose,
            )
        )
        self._session.__enter__()
        self._controller_index = ControllerInputIndex

    def close(self) -> None:
        """Close the Isaac Teleop session."""
        self._session.__exit__(None, None, None)

    def get_frame(self) -> Quest3Frame:
        """Read one sample from Isaac Teleop.

        >>> Quest3(Quest3Config()).get_frame()  # doctest: +SKIP
        """
        result = self._session.step()
        timestamp = time.time()
        timestamp_ns = time.time_ns()
        controllers = {}
        for side in self._config.sides:
            group = result[f"{side}_controller"]
            if group.is_none or not bool(group[self._controller_index.GRIP_IS_VALID]):
                continue
            position = np.asarray(group[self._controller_index.GRIP_POSITION], dtype=np.float64).reshape(3)
            orientation_xyzw = np.asarray(group[self._controller_index.GRIP_ORIENTATION], dtype=np.float64).reshape(4)
            rotation = quaternion_to_matrix(orientation_xyzw[[3, 0, 1, 2]])
            controllers[side] = Quest3ControllerFrame(
                pose=rot_tl_to_tf_mat(rotation, position),
                primary_click=bool(group[self._controller_index.PRIMARY_CLICK]),
                secondary_click=bool(group[self._controller_index.SECONDARY_CLICK]),
            )
        return Quest3Frame(
            timestamp=timestamp,
            timestamp_ns=timestamp_ns,
            controllers=controllers,
        )
