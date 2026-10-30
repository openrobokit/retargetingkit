"""DexMachina hand-object interactions stored on Hugging Face."""

from pathlib import Path

from retargetingkit.assets import HF_REPO_ID, RETARGETINGKIT_HOME, REVISION
from robokit.assets import fetch


def root() -> Path:
    """Fetch the DexMachina HOI dataset and return its directory.

    >>> root().name  # doctest: +SKIP
    'dexmachina_hoi'
    """
    return fetch(["data/dexmachina_hoi/**"], REVISION, HF_REPO_ID, RETARGETINGKIT_HOME) / "data/dexmachina_hoi"
