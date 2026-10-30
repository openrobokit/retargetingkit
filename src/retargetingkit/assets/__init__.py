"""HuggingFace assets: each `retargetingkit.assets.<kind>` accessor fetches only its own files with RoboKit's `fetch`."""

from pathlib import Path


HF_REPO_ID = "openrobokit/retargetingkit"
RETARGETINGKIT_HOME = Path.home() / ".retargetingkit"
ASSETS_DIR = RETARGETINGKIT_HOME / "assets"
REVISION = "4718aabb469e6c80f39309ea29643d80ca9bc997"
