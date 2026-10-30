"""PICO-to-G1-and-dual-Inspire teleoperation preset."""

from retargetingkit.configs.teleop.config import HumanoidHandTeleopConfig
from retargetingkit.retargeting.humanoid.online import HumanoidRetargeterConfig
from retargetingkit.retargeting.humanoid.presets.g1_pico import g1_pico
from retargetingkit.sources.vr.pico import PicoSourceConfig
from retargetingkit.targets.hands.inspire import InspireHandConfig
from retargetingkit.targets.humanoids.sonic import SonicTargetConfig


PICO_G1_INSPIRE_CONFIG = HumanoidHandTeleopConfig(
    source=PicoSourceConfig(human_height=1.75, use_ankle_as_foot_target=True),
    retargeter=HumanoidRetargeterConfig(preset=g1_pico),
    targets=(SonicTargetConfig(host="*", port=5556),),
    left_hand=InspireHandConfig(serial_port="/dev/ttyUSB0", hand_id=1, baudrate=115200),
    right_hand=InspireHandConfig(serial_port="/dev/ttyUSB1", hand_id=1, baudrate=115200),
    open_qpos=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    closed_qpos=(1.3, 0.27, 0.85, 0.85, 0.85, 0.85),
)
