"""Named neural-hand checkpoints stored on Hugging Face."""

from pathlib import Path

from retargetingkit.assets import HF_REPO_ID, RETARGETINGKIT_HOME, REVISION
from robokit.assets import fetch


CHECKPOINTS: dict[str, str] = {
    "allegro_motion_alignment": "allegro",
    "sharpa_motion_alignment": "sharpa",
    "sharpa_motion_alignment_contact": "sharpa",
    "sharpa_motion_alignment_contact_inhand": "sharpa",
    "xhand_motion_alignment": "xhand",
}


def get(name: str) -> Path:
    """Fetch one named checkpoint and return its local directory.

    >>> get("allegro_motion_alignment").name  # doctest: +SKIP
    'allegro_motion_alignment'
    """
    CHECKPOINTS[name]
    return fetch([f"models/ik_model/{name}/**"], REVISION, HF_REPO_ID, RETARGETINGKIT_HOME) / "models/ik_model" / name
