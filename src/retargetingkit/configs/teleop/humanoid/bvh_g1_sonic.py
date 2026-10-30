"""Sample-BVH-to-G1-SONIC teleoperation preset."""

from retargetingkit.assets import sample_bvh
from retargetingkit.configs.teleop.config import HumanoidTeleopConfig
from retargetingkit.retargeting.humanoid.online import HumanoidRetargeterConfig
from retargetingkit.retargeting.humanoid.presets.g1 import g1_aligned
from retargetingkit.sources.misc.human_frame_file import BodyFileSourceConfig
from retargetingkit.targets.humanoids.sonic import SonicTargetConfig


BVH_G1_SONIC_CONFIG = HumanoidTeleopConfig(
    source=BodyFileSourceConfig(
        path=sample_bvh.EXAMPLE_BVH,
        loop=True,
        use_ankle_as_foot_target=True,
        human_height=1.75,
    ),
    retargeter=HumanoidRetargeterConfig(preset=g1_aligned),
    targets=(SonicTargetConfig(host="*", port=5556),),
)
