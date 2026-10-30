"""Use local assets and reject Hugging Face calls, including during collection."""

from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config):
    patch = pytest.MonkeyPatch()
    config.add_cleanup(patch.undo)
    patch.setenv("HF_HUB_OFFLINE", "1")
    import huggingface_hub
    from huggingface_hub import constants

    from retargetingkit import assets as retargeting_assets
    from robokit import assets

    root = Path(__file__).parent / "tests/fixtures"

    def local_fetch(patterns=None, revision=None, repo_id: str = assets.HF_REPO_ID, home=None) -> Path:
        return root / ("robokit" if repo_id == assets.HF_REPO_ID else "retargetingkit")

    def deny_hf(*args, **kwargs):
        raise AssertionError("Tests must use local fixtures; Hugging Face access is forbidden.")

    patch.setattr(assets, "fetch", local_fetch)
    patch.setattr(retargeting_assets, "ASSETS_DIR", root)
    patch.setattr(retargeting_assets, "REVISION", "retargetingkit")
    patch.setattr(constants, "HF_HUB_OFFLINE", True)
    patch.setattr(huggingface_hub, "snapshot_download", deny_hf)
    patch.setattr(huggingface_hub, "hf_hub_download", deny_hf)
    patch.setattr(huggingface_hub.HfApi, "repo_info", deny_hf)
    patch.setattr(huggingface_hub.HfApi, "list_repo_tree", deny_hf)
