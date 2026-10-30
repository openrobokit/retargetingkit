from pathlib import Path

import huggingface_hub
import pytest

from retargetingkit.assets import sample_bvh
from robokit.assets.robots.arms import xarm7


class TestOfflineAssets:
    def test_assets_use_fixtures(self):
        root = Path(__file__).parent / "fixtures"
        assert xarm7.URDF_PATH.is_relative_to(root / "robokit")
        assert xarm7.URDF_PATH.is_file()
        assert sample_bvh.EXAMPLE_BVH == root / "retargetingkit/data/sample_bvh/example.bvh"
        assert sample_bvh.EXAMPLE_BVH.is_file()

    @pytest.mark.parametrize("name", ["snapshot_download", "hf_hub_download", "repo_info", "list_repo_tree"])
    def test_hub_calls_are_forbidden(self, name: str, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "0")
        api = huggingface_hub if name.endswith("download") else huggingface_hub.HfApi()
        with pytest.raises(AssertionError, match="Hugging Face access is forbidden"):
            getattr(api, name)(repo_id="test/forbidden")
