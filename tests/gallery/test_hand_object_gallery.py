"""Clip switching must clear hands and contact markers absent from the new clip."""

import importlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import viser

from retargetingkit.retargeting.hand_object.offline import (
    ArticulatedObjectTrack,
    HandObjectClip,
    HandObjectMotion,
    HandObjectRetargeter,
    HandTrack,
    ObjectPart,
    RobotHandTrack,
)


class TestHandObjectGallery:
    @pytest.mark.parametrize("dataset", ("dexmachina", "oakink2", "gigahands"))
    def test_switch_single_and_bimanual_clips(
        self, dataset: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
    ) -> None:
        gallery = importlib.import_module(f"retargetingkit.gallery.hand_object_{dataset}")
        pose = np.tile(np.array([0, 0, 0, 1, 0, 0, 0], np.float32), (3, 1))
        hand = HandTrack(
            np.zeros((3, 21, 3), np.float32),
            pose,
            np.zeros((3, 16, 3), np.float32),
            np.zeros((3, 16, 3), np.float32),
            np.zeros((3, 16), np.int8),
            np.ones((3, 16), bool),
        )
        part = ObjectPart(
            "box", np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32), np.array([[0, 1, 2]], np.int32)
        )
        obj = ArticulatedObjectTrack("box", pose, np.zeros((3, 0)), (part,), pose[:, None])
        clips = {
            "right": HandObjectClip("right", 30, {"right": hand}, obj),
            "both": HandObjectClip("both", 30, {"right": hand, "left": hand}, obj),
            "left": HandObjectClip("left", 30, {"left": hand}, obj),
        }
        (tmp_path / "manifest.json").write_text(
            json.dumps(
                {
                    "clips": [{"label": name, "path": name} for name in clips],
                    "wrist_coordinate": {
                        "right": {"palm_forward_axis": "+x", "four_fingers_up_axis": "+z"},
                        "left": {"palm_forward_axis": "-x", "four_fingers_up_axis": "+z"},
                    },
                }
            )
        )
        asset = importlib.import_module(f"retargetingkit.assets.{dataset}_hoi")
        monkeypatch.setattr(asset, "root", lambda: tmp_path)
        monkeypatch.setattr(gallery, "load_hand_object_clip", lambda path: clips[path.name])

        def solve(retargeter: HandObjectRetargeter, clip: HandObjectClip) -> HandObjectMotion:
            if dataset == "gigahands":
                assert retargeter.specs["right"].target_coord_spec.palm_forward_axis == "x"
                assert retargeter.specs["left"].target_coord_spec.palm_forward_axis == "-x"
                assert all(spec.target_coord_spec.four_fingers_up_axis == "z" for spec in retargeter.specs.values())
            return HandObjectMotion(
                clip,
                {
                    side: RobotHandTrack(np.tile(retargeter.robots[side].spec.zero_q, (3, 1)), pose)
                    for side in clip.hands
                },
            )

        monkeypatch.setattr(HandObjectRetargeter, "solve", solve)
        server = viser.ViserServer(port=0)
        request.addfinalizer(server.stop)
        monkeypatch.setattr(gallery, "viser", SimpleNamespace(ViserServer=lambda: server))
        callbacks = []
        dropdown = SimpleNamespace(value="01 right", on_update=callbacks.append)
        monkeypatch.setattr(server.gui, "add_dropdown", lambda *args, **kwargs: dropdown)
        selections = [
            ("01 right", {"right"}),
            ("02 both", {"right", "left"}),
            ("03 left", {"left"}),
            ("01 right", {"right"}),
        ]
        observed = []

        def advance(_seconds: float) -> None:
            _, present = selections[len(observed)]
            for side in ("right", "left"):
                frame = server.scene.get_handle_by_name(f"/robot/{side}")
                assert frame is not None and frame.visible == (side in present)
                marker = server.scene.get_handle_by_name(f"/contacts/{side}/0")
                assert marker is not None and marker.visible == (side in present)
            observed.append(dropdown.value)
            if len(observed) == len(selections):
                raise StopIteration("playback verified")
            dropdown.value = selections[len(observed)][0]
            callbacks[0](None)

        monkeypatch.setattr(gallery, "time", SimpleNamespace(time=time.time, sleep=advance))
        monkeypatch.setattr(sys, "argv", ["gallery"])
        with pytest.raises(StopIteration, match="playback verified"):
            gallery.main()
        assert len(observed) == len(selections)
