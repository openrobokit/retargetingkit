"""GigaHands hand-object clips from RoboKit's Hugging Face dataset."""

from pathlib import Path

from robokit.assets import fetch


def root() -> Path:
    """Fetch the GigaHands HOI dataset and return its directory.

    >>> root().name  # doctest: +SKIP
    'gigahands'
    """
    return fetch(["datasets/gigahands/**"]) / "datasets/gigahands"
