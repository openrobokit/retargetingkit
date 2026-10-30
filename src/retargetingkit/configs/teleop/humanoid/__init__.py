"""Humanoid teleop config registry."""

from retargetingkit.configs.teleop.humanoid.axisstudio_g1_sonic import AXISSTUDIO_G1_SONIC_CONFIG
from retargetingkit.configs.teleop.humanoid.axisstudio_g1_sonic_viser import AXISSTUDIO_G1_SONIC_VISER_CONFIG
from retargetingkit.configs.teleop.humanoid.bvh_g1_sonic import BVH_G1_SONIC_CONFIG
from retargetingkit.configs.teleop.humanoid.bvh_g1_sonic_viser import BVH_G1_SONIC_VISER_CONFIG
from retargetingkit.configs.teleop.humanoid.pico_g1_sonic import PICO_G1_SONIC_CONFIG
from retargetingkit.configs.teleop.humanoid.pico_g1_sonic_viser import PICO_G1_SONIC_VISER_CONFIG
from retargetingkit.configs.teleop.humanoid.picofile_g1_sonic import PICOFILE_G1_SONIC_CONFIG
from retargetingkit.configs.teleop.humanoid.picofile_g1_sonic_viser import PICOFILE_G1_SONIC_VISER_CONFIG


_HUMANOID_TELEOP_CONFIGS = {
    "bvh_g1_sonic": BVH_G1_SONIC_CONFIG,
    "bvh_g1_sonic_viser": BVH_G1_SONIC_VISER_CONFIG,
    "axisstudio_g1_sonic": AXISSTUDIO_G1_SONIC_CONFIG,
    "axisstudio_g1_sonic_viser": AXISSTUDIO_G1_SONIC_VISER_CONFIG,
    "pico_g1_sonic": PICO_G1_SONIC_CONFIG,
    "pico_g1_sonic_viser": PICO_G1_SONIC_VISER_CONFIG,
    "picofile_g1_sonic": PICOFILE_G1_SONIC_CONFIG,
    "picofile_g1_sonic_viser": PICOFILE_G1_SONIC_VISER_CONFIG,
}
