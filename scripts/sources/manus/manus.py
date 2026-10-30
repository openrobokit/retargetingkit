"""Small Manus stream wrapper used by the playground scripts."""

import numpy as np

from retargetingkit.sources.glove._manus_sdk import ManusClient, resolve_manus_sdk_library_path
from retargetingkit.sources.glove.manus_conversion import manus_positions_to_fixed_wrist_local
from retargetingkit.sources.glove.manus_glove import (
    DualManusGloveAggregator,
    ManusGloveFrame,
)


class Manus:
    """Manage the Manus SDK client and expose the latest frame for each hand."""

    def __init__(self, left_glove_id: int | None = None, right_glove_id: int | None = None) -> None:
        self.aggregator = DualManusGloveAggregator(left_override=left_glove_id, right_override=right_glove_id)
        self.client = ManusClient(resolve_manus_sdk_library_path())

    def start(self) -> None:
        self.client.start(self.aggregator.update)

    def stop(self) -> None:
        self.client.stop()

    def latest(self, hand: str) -> ManusGloveFrame | None:
        left, right = self.aggregator.latest()
        return right if hand == "right" else left

    def latest_wrist_local(self, hand: str) -> np.ndarray | None:
        frame = self.latest(hand)
        if frame is None:
            return None
        return manus_positions_to_fixed_wrist_local(frame.positions, frame.rotations)
