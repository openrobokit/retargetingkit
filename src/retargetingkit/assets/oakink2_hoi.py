"""Canonical OakInk2 hand-object clips stored on Hugging Face."""

from pathlib import Path

from retargetingkit.assets import HF_REPO_ID, RETARGETINGKIT_HOME, REVISION
from robokit.assets import fetch


def root() -> Path:
    """Fetch the OakInk2 HOI clips and return their directory.

    >>> root().name  # doctest: +SKIP
    'oakink2_hoi'
    """
    return fetch(["data/oakink2_hoi/**"], REVISION, HF_REPO_ID, RETARGETINGKIT_HOME) / "data/oakink2_hoi"
