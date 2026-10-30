from pathlib import Path

import pytest
import torch

from retargetingkit.assets import neural_hand
from retargetingkit.retargeting.hand.neural import IKModel, NeuralHandAssetRetargeterConfig
from retargetingkit.retargeting.hand.presets import HANDS
from retargetingkit.targets.hands.sharpa_hand import JOINT_NAMES as SHARPA_JOINT_NAMES
from retargetingkit.targets.hands.xhand import JOINT_NAMES as XHAND_JOINT_NAMES


class TestHandPresets:
    def test_resolved_robot_layouts_and_cat_order(self) -> None:
        expected = {
            "sharpa": (22, 5),
            "allegro": (16, 4),
            "xhand": (12, 5),
        }

        assert set(HANDS) == set(expected)
        for name, (joint_count, fingertip_count) in expected.items():
            spec = HANDS[name]
            flattened = tuple(joint for finger in spec.finger_joint_names for joint in finger)

            assert flattened == spec.joint_names
            assert spec.joint_count == joint_count
            assert len(spec.human_keypoint_indices) == fingertip_count
            assert len(spec.fingertip_links) == fingertip_count
            assert spec.joint_lower.shape == (joint_count,)
            assert spec.joint_upper.shape == (joint_count,)
            assert bool((spec.joint_lower < spec.joint_upper).all())
            assert spec.collision_spheres_path is None or spec.collision_spheres_path.is_file()
            assert spec.collision_ignore_path is None or spec.collision_ignore_path.is_file()

    def test_hardware_order_matches_resolved_spec(self) -> None:
        assert HANDS["sharpa"].joint_names == SHARPA_JOINT_NAMES
        assert HANDS["xhand"].joint_names == XHAND_JOINT_NAMES


class TestNeuralCheckpointAssets:
    def test_registry_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        requested: list[str] = []

        def materialize(patterns: list[str], *_args) -> Path:
            requested.extend(patterns)
            return Path("/cache")

        monkeypatch.setattr(neural_hand, "fetch", materialize)
        assert neural_hand.CHECKPOINTS == {
            "allegro_motion_alignment": "allegro",
            "sharpa_motion_alignment": "sharpa",
            "sharpa_motion_alignment_contact": "sharpa",
            "sharpa_motion_alignment_contact_inhand": "sharpa",
            "xhand_motion_alignment": "xhand",
        }
        for name in neural_hand.CHECKPOINTS:
            assert neural_hand.get(name) == Path("/cache/models/ik_model") / name
        assert requested == [f"models/ik_model/{name}/**" for name in neural_hand.CHECKPOINTS]

    def test_asset_config_resolves_only_on_build(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spec = HANDS["allegro"]
        torch.save(IKModel(spec.finger_joint_sizes).state_dict(), tmp_path / "weights.pt")
        requested: list[str] = []

        def resolve(name: str) -> Path:
            requested.append(name)
            return tmp_path

        monkeypatch.setattr(neural_hand, "get", resolve)
        config = NeuralHandAssetRetargeterConfig("allegro_motion_alignment", spec)
        assert requested == []

        retargeter = config.build("cpu")

        assert requested == ["allegro_motion_alignment"]
        assert retargeter.side == "right"
