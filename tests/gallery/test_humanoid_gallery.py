"""Interaction gallery presets must resolve their calibrated frames and collision assets."""

import os

import pytest

from retargetingkit.gallery import humanoid_contact, humanoid_terrain


pytestmark = pytest.mark.skipif(
    os.environ.get("RETARGETINGKIT_ASSET_TESTS") != "1",
    reason="set RETARGETINGKIT_ASSET_TESTS=1 with locally prepared gallery assets",
)


class TestHumanoidGallery:
    @pytest.mark.parametrize("robot_name", ("g1", "gr3", "h1_2", "bhl"))
    @pytest.mark.parametrize("motion", ("smplh", "mocap"))
    def test_interaction_assets(self, robot_name: str, motion: str) -> None:
        gallery = humanoid_contact if motion == "smplh" else humanoid_terrain
        config = gallery._load_preset(robot_name)
        robot = config.load_robot()
        assert set(config.link_mapping) <= set(robot.spec.link_names)
        assert len(robot.spec.collision_spheres_link_indices) > 0
        assert config.human_root_name == ("Pelvis" if motion == "smplh" else "Spine1")
        if robot_name == "g1":
            assert "left_ankle_roll_sphere_5_link" in config.link_mapping
            assert "right_ankle_roll_sphere_5_link" in config.link_mapping
