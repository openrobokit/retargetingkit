"""PICO-to-G1-SONIC teleoperation preset."""

from retargetingkit.configs.teleop.config import HumanoidTeleopConfig
from retargetingkit.retargeting.humanoid.online import HumanoidRetargeterConfig
from retargetingkit.retargeting.humanoid.presets.g1_pico import g1_pico
from retargetingkit.sources.vr.pico import PicoSourceConfig
from retargetingkit.targets.humanoids.sonic import SonicTargetConfig


PICO_G1_SONIC_CONFIG = HumanoidTeleopConfig(
    source=PicoSourceConfig(human_height=1.75, use_ankle_as_foot_target=True),
    retargeter=HumanoidRetargeterConfig(preset=g1_pico),
    targets=(SonicTargetConfig(host="*", port=5556),),
)
