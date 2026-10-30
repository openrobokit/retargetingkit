"""Sample BVH motion (`data/sample_bvh/` on the HF dataset)."""

from pathlib import Path

from retargetingkit.assets import HF_REPO_ID, RETARGETINGKIT_HOME, REVISION
from robokit.assets import fetch


EXAMPLE_BVH: Path = (
    fetch(["data/sample_bvh/example.bvh"], REVISION, HF_REPO_ID, RETARGETINGKIT_HOME) / "data/sample_bvh/example.bvh"
)
