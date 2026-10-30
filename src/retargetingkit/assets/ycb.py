"""YCB google_16k meshes, not re-hosted: fetched from the YCB S3 into `~/.retargetingkit/ycb/<obj>/` on first use."""

import tarfile
import urllib.request
from pathlib import Path

from retargetingkit.assets import RETARGETINGKIT_HOME


YCB_BASE_URL = "https://ycb-benchmarks.s3.amazonaws.com/data/google/{}_google_16k.tgz"
YCB_OBJECTS = (
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "012_strawberry",
    "013_apple",
    "014_lemon",
    "015_peach",
    "016_pear",
    "017_orange",
    "018_plum",
    "021_bleach_cleanser",
    "025_mug",
    "056_tennis_ball",
    "058_golf_ball",
    "061_foam_brick",
    "077_rubiks_cube",
)

YCB_DIR = RETARGETINGKIT_HOME / "ycb"


def ycb(obj: str) -> Path:
    """Download+extract YCB `<obj>` into the cache on first use and return its directory."""
    target = YCB_DIR / obj
    if (target / "google_16k").is_dir():
        return target
    YCB_DIR.mkdir(parents=True, exist_ok=True)
    tgz = YCB_DIR / f"{obj}.tgz"
    urllib.request.urlretrieve(YCB_BASE_URL.format(obj), tgz)
    with tarfile.open(tgz) as tar:
        tar.extractall(YCB_DIR)
    tgz.unlink()
    return target
