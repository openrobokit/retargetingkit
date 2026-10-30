"""RoboKit full-body retargeting with the calibrated G1 kinematic model."""

from dataclasses import replace

from retargetingkit.retargeting.humanoid.presets.g1 import g1_offline_mapping
from robokit.helpers.full_body_retargeting import FullBodyRetargeter as RoboKitFullBodyRetargeter
from robokit.helpers.full_body_retargeting import FullBodyRetargetingConfig
from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOffline


class FullBodyRetargeter(RoboKitFullBodyRetargeter):
    """Keep the toe frames required by RoboKit's internal G1 mappings."""

    def __init__(
        self, config: FullBodyRetargetingConfig | None = None, device: str = "cuda:0", load_meshes: bool = False
    ) -> None:
        super().__init__(config, device, load_meshes)
        self._kin_robot = g1_offline_mapping.load_robot()
        self._kin = HumanoidRetargeterOffline(
            replace(self._kin.retargeting_config, urdf_path=g1_offline_mapping.urdf_path),
            self._kin.offline_config,
            robot=self._kin_robot,
            device=device,
        )
