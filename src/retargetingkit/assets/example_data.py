"""Example recorded hand data (`data/example_data/` on the HF dataset)."""

from pathlib import Path

from retargetingkit.assets import HF_REPO_ID, RETARGETINGKIT_HOME, REVISION
from robokit.assets import fetch


_MANUS, _AVP = "data/example_data/manus_data_right.pkl", "data/example_data/avp_data_right.pkl"
MANUS_DATA_RIGHT_PKL: Path = fetch([_MANUS], REVISION, HF_REPO_ID, RETARGETINGKIT_HOME) / _MANUS
AVP_DATA_RIGHT_PKL: Path = fetch([_AVP], REVISION, HF_REPO_ID, RETARGETINGKIT_HOME) / _AVP


def root() -> Path:
    """Fetch all `data/example_data/` and return the directory (for walking over bundles)."""
    return fetch(["data/example_data/**"], REVISION, HF_REPO_ID, RETARGETINGKIT_HOME) / "data/example_data"
