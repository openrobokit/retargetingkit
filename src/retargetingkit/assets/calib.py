"""Named calibration and neural-training assets stored on Hugging Face."""

from pathlib import Path

from retargetingkit.assets import HF_REPO_ID, RETARGETINGKIT_HOME, REVISION
from robokit.assets import fetch


_ASSETS: dict[str, str] = {
    "manus_calib": "data/calib_data/manus_calib.npz",
    "manus_unsupervised": "data/neural_hand/human/manus/unsupervised.npz",
    "manus_motion": "data/neural_hand/human/manus/motion.npz",
    "mano_alex": "data/neural_hand/human/mano/alex.npy",
    "sharpa_open": "data/neural_hand/pairs/sharpa/open.npz",
    "sharpa_pinch": "data/neural_hand/pairs/sharpa/pinch.npz",
    "sharpa_rest": "data/neural_hand/pairs/sharpa/rest.npz",
    "xhand_open": "data/neural_hand/pairs/xhand/open.npz",
    "xhand_open_motion": "data/neural_hand/pairs/xhand/open_motion.npz",
    "xhand_index_motion": "data/neural_hand/pairs/xhand/index_motion.npz",
    "xhand_pinch": "data/neural_hand/pairs/xhand/pinch.npz",
    "xhand_rest": "data/neural_hand/pairs/xhand/rest.npz",
    "allegro_open": "data/neural_hand/pairs/allegro/open.npz",
    "allegro_open_motion": "data/neural_hand/pairs/allegro/open_motion.npz",
    "allegro_index_motion": "data/neural_hand/pairs/allegro/index_motion.npz",
    "allegro_pinch": "data/neural_hand/pairs/allegro/pinch.npz",
    "allegro_index_pinch_motion": "data/neural_hand/pairs/allegro/index_pinch_motion.npz",
    "allegro_middle_pinch_motion": "data/neural_hand/pairs/allegro/middle_pinch_motion.npz",
    "allegro_ring_pinch_motion": "data/neural_hand/pairs/allegro/ring_pinch_motion.npz",
    "allegro_rest": "data/neural_hand/pairs/allegro/rest.npz",
    "sharpa_contact_align_inhand": "data/neural_hand/geo/sharpa/inhand.npz",
    "sharpa_contact_align_ycb": "data/neural_hand/geo/sharpa/ycb_capture.npz",
    "sharpa_contact_grasp_ycb": "data/neural_hand/geo/sharpa/ycb_grasps.npz",
}


def get(name: str) -> Path:
    """Fetch one named calibration asset and return its local materialized path.

    >>> get("manus_motion").name  # doctest: +SKIP
    'motion.npz'
    """
    return fetch([_ASSETS[name]], REVISION, HF_REPO_ID, RETARGETINGKIT_HOME) / _ASSETS[name]
