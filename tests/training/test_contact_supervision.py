from pathlib import Path
from typing import cast

import numpy as np
import torch
import trimesh
import yaml

from retargetingkit.retargeting.hand.presets import HANDS
from retargetingkit.training.hand.contact import (
    FINGER_NAMES,
    HandContactModel,
    YcbObjectModel,
    fit_object_translation,
    nearest_pad_gaps,
)
from retargetingkit.training.hand.data import ContactCaptureData, GraspSpecData
from retargetingkit.training.hand.kinematics import load_hand_robot
from robokit.geom import MeshGeom, WarpScene


class TestContactSupervision:
    def test_generation_config_is_independent_from_training_sources(self) -> None:
        path = Path(__file__).parents[2] / "scripts/teleop/neural_hand/configs/contact/sharpa_ycb_contact.yaml"
        config = yaml.safe_load(path.read_text())

        assert config["init_pose_asset"] == "sharpa_contact_align_inhand"
        assert config["human_reference_asset"] == "manus_unsupervised"
        assert "pair" not in config
        assert "contact" not in config

    def test_robokit_geometry(self) -> None:
        device = torch.device("cpu")
        mesh = cast(trimesh.Trimesh, trimesh.creation.box(extents=(0.1, 0.1, 0.1)))
        scene = WarpScene(1, device).add(MeshGeom([mesh], np.array([0, 1], dtype=np.int32)))
        obj = YcbObjectModel(mesh, scene)
        points = torch.tensor([[[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]], device=device, requires_grad=True)
        translation = torch.zeros((1, 3), device=device, requires_grad=True)

        distance = obj.signed_distance(points, translation, torch.eye(3, device=device).unsqueeze(0))
        distance.sum().backward()
        torch.testing.assert_close(distance, torch.tensor([[0.05, -0.05]], device=device))
        assert points.grad is not None
        assert translation.grad is not None

        spheres = torch.tensor([[[0.1 + 0.01 * finger, 0.0, 0.0] for finger in range(5)]], device=device)
        indices = tuple(torch.tensor([finger], device=device) for finger in range(5))
        _, gaps, contacts = nearest_pad_gaps(
            spheres,
            indices,
            torch.full((5,), 0.01, device=device),
            obj,
            torch.zeros((1, 3), device=device),
            torch.eye(3, device=device).unsqueeze(0),
        )
        torch.testing.assert_close(gaps, torch.tensor([[0.04, 0.05, 0.06, 0.07, 0.08]], device=device))
        torch.testing.assert_close(contacts, spheres)

    def test_contact_fk_and_capture_translation(self) -> None:
        device = torch.device("cpu")
        spec = HANDS["sharpa"]
        robot = load_hand_robot(spec)
        joint_limits = torch.from_numpy(robot.spec.actuated_joint_limits).float().to(device)
        pad_link_names = tuple(f"right_{finger}_elastomer" for finger in FINGER_NAMES)
        contact_model = HandContactModel.build(
            robot,
            joint_limits,
            pad_link_names,
            spec.collision_max_active_pairs,
        )
        joints = torch.zeros((2, 22), device=device, requires_grad=True)
        spheres = contact_model.spheres(joints)
        spheres.square().mean().backward()
        assert spheres.shape == (2, 40, 3)
        assert [indices.numel() for indices in contact_model.pad_indices] == [1, 1, 1, 1, 1]
        assert joints.grad is not None

        mesh = cast(trimesh.Trimesh, trimesh.creation.box(extents=(0.1, 0.1, 0.1)))
        obj = YcbObjectModel(mesh, WarpScene(1, device).add(MeshGeom([mesh], np.array([0, 1], dtype=np.int32))))
        fingertips = torch.tensor([[0.1, 0.0, 0.0]] * 5, device=device)
        translation, gap, targets = fit_object_translation(
            obj,
            fingertips,
            torch.zeros((5, 3), device=device),
            torch.ones(5, device=device),
            torch.eye(3, device=device).unsqueeze(0),
            torch.zeros((1, 3), device=device),
            torch.ones((1, 1, 3), device=device),
            0.0,
            0.0,
            100,
            0.02,
        )
        torch.testing.assert_close(translation, torch.tensor([[0.1, 0.0, 0.0]], device=device), atol=1e-3, rtol=0)
        assert gap < 1e-3
        torch.testing.assert_close(torch.as_tensor(targets), fingertips, rtol=0, atol=1e-3)

    def test_data_schema_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "training_data.npz"
        grasps = GraspSpecData(
            qpos=np.zeros((2, 22), dtype=np.float32),
            object_translation=np.zeros((2, 3), dtype=np.float32),
            object_rotvec=np.zeros((2, 3), dtype=np.float32),
            contact_mask=np.ones((2, 5), dtype=bool),
            contact_points=np.zeros((2, 5, 3), dtype=np.float32),
            object_names=np.array(["006_mustard_bottle", "025_mug"]),
        )
        data = ContactCaptureData(
            grasps,
            np.zeros((2, 25, 3), dtype=np.float32),
            np.zeros((2, 25, 4), dtype=np.float32),
        )

        data.save(path)
        loaded = ContactCaptureData.load(path)
        torch.testing.assert_close(torch.from_numpy(loaded.grasps.qpos), torch.from_numpy(grasps.qpos), rtol=0, atol=0)
        torch.testing.assert_close(torch.from_numpy(loaded.grasps.contact_mask), torch.from_numpy(grasps.contact_mask))
        assert loaded.grasps.object_names.tolist() == grasps.object_names.tolist()
        with np.load(path) as raw:
            assert set(raw.files) == {
                "grasp_q",
                "object_t_init",
                "object_rotvec_fixed",
                "contact_finger_mask",
                "contact_target_points",
                "object_names",
                "right_timestamps",
                "right_positions",
                "right_rotations",
            }
