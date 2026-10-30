"""Recorded-PICO-to-G1-SONIC-and-Viser teleoperation preset."""

from pathlib import Path

from retargetingkit.configs.teleop.config import HumanoidTeleopConfig
from retargetingkit.retargeting.humanoid.online import HumanoidRetargeterConfig
from retargetingkit.retargeting.humanoid.presets.g1_pico import g1_pico
from retargetingkit.sources.misc.human_frame_file import BodyFileSourceConfig
from retargetingkit.targets.humanoids.sonic import SonicTargetConfig
from retargetingkit.targets.viser import ViserConfig
from robokit.assets.robots.humanoids import unitree_g1


PICOFILE_G1_SONIC_VISER_CONFIG = HumanoidTeleopConfig(
    source=BodyFileSourceConfig(
        path=Path("assets/pico_recordings/default.npz"),
        loop=True,
        use_ankle_as_foot_target=True,
        human_height=1.75,
    ),
    retargeter=HumanoidRetargeterConfig(preset=g1_pico),
    targets=(
        SonicTargetConfig(host="*", port=5556),
        ViserConfig((unitree_g1.URDF_PATH,)),
    ),
)
