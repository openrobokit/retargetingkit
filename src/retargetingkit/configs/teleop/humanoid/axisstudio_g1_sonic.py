"""Axis-Studio-to-G1-SONIC teleoperation preset."""

from retargetingkit.configs.teleop.config import HumanoidTeleopConfig
from retargetingkit.retargeting.humanoid.online import HumanoidRetargeterConfig
from retargetingkit.retargeting.humanoid.presets.g1 import g1_aligned
from retargetingkit.sources.mocap.axis_studio import AxisStudioSourceConfig
from retargetingkit.targets.humanoids.sonic import SonicTargetConfig


AXISSTUDIO_G1_SONIC_CONFIG = HumanoidTeleopConfig(
    source=AxisStudioSourceConfig(port=7012, human_height=1.75, use_ankle_as_foot_target=True),
    retargeter=HumanoidRetargeterConfig(preset=g1_aligned),
    targets=(SonicTargetConfig(host="*", port=5556),),
)
