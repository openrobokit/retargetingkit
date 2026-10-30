import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from retargetingkit.retargeting.hand.neural import IKModel, NeuralHandRetargeterConfig
from retargetingkit.retargeting.hand.presets import HANDS
from retargetingkit.training.hand import metrics
from retargetingkit.training.hand.data import (
    JointSupervision,
    load_contact_supervision,
    load_human_dataset,
    load_pair_supervision,
)
from retargetingkit.training.hand.objective import ObjectiveContext, compute_objective
from retargetingkit.training.hand.train import TrainConfig, train_from_config


CONFIG_DIR = Path(__file__).parents[2] / "scripts/teleop/neural_hand/configs/training"


class _InputModel(nn.Module):
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return points[:, 0, 0, None].expand(-1, 22)


class _ConstantFinger(nn.Module):
    def __init__(self, values: tuple[float, ...]) -> None:
        super().__init__()
        self.register_buffer("values", torch.tensor(values))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.values.expand(len(points), -1)


class TestNeuralRetargeting:
    def test_canonical_configs(self) -> None:
        paths = {path.name: path for path in CONFIG_DIR.glob("*.yaml")}
        assert set(paths) == {
            "allegro_motion_alignment.yaml",
            "sharpa_motion_alignment.yaml",
            "sharpa_motion_alignment_contact.yaml",
            "sharpa_motion_alignment_contact_inhand.yaml",
            "xhand_motion_alignment.yaml",
        }
        configs = {name: TrainConfig.from_yaml(path) for name, path in paths.items()}
        config = configs["sharpa_motion_alignment_contact.yaml"]
        assert config.output.name == "sharpa_motion_alignment_contact"
        assert config.objective.motion.formula == "motion_alignment"
        assert config.robot_samples == 1_000_000
        assert metrics.PINCH_THRESHOLD_M == 0.015
        assert metrics.CURVATURE_STEP_M == 0.002
        assert metrics.INTERFINGER_MARGIN_SCALE == 0.3
        assert metrics.INTERFINGER_MIN_MARGIN_M == 0.001
        assert metrics.COLLISION_MARGIN_M == 0.0
        assert config.objective.collision == 5.0
        assert [source.weight for source in config.data.pair_sources] == [1.0, 1.0, 1.0]
        assert [source.weight for source in config.data.contact_sources] == [5.0, 1.0]
        inhand = configs["sharpa_motion_alignment_contact_inhand.yaml"]
        assert inhand.output.name == "sharpa_motion_alignment_contact_inhand"
        assert inhand.objective.contact == 400.0
        assert [source.path.name for source in inhand.data.contact_sources] == ["inhand.npz"]
        assert [source.contact_only for source in inhand.data.contact_sources] == [False]
        weighted = configs["sharpa_motion_alignment.yaml"]
        assert weighted.output.name == "sharpa_motion_alignment"
        assert [source.weight for source in weighted.data.pair_sources] == [1.5, 1.0, 1.0]
        assert weighted.objective.motion.weight == 60.0
        assert weighted.objective.pinch == 750.0
        assert weighted.objective.interfinger == 150.0
        assert weighted.objective.collision == 20.0
        allegro = configs["allegro_motion_alignment.yaml"]
        assert allegro.hand == "allegro"
        assert allegro.output.name == "allegro_motion_alignment"
        assert allegro.objective.pair == 40.0
        assert allegro.objective.pair_position == 5000.0
        assert allegro.objective.motion.formula == "motion_alignment"
        assert [source.path.parent.name for source in allegro.data.pair_sources] == ["allegro"] * 8
        assert [source.weight for source in allegro.data.pair_sources] == [6.0, 4.0, 0.2, 10.0, 2.0, 2.0, 2.0, 1.0]
        assert configs["xhand_motion_alignment.yaml"].hand == "xhand"
        xhand = configs["xhand_motion_alignment.yaml"]
        assert xhand.output.name == "xhand_motion_alignment"
        assert xhand.objective.pair == 320.0
        assert xhand.objective.motion.formula == "motion_alignment"
        assert xhand.objective.collision == 640.0
        assert [source.path.parent.name for source in xhand.data.pair_sources] == ["xhand"] * 5
        assert [source.weight for source in xhand.data.pair_sources] == [6.0, 4.0, 0.2, 1.0, 1.0]
        assert xhand.optimization.epochs == 500
        assert xhand.optimization.seed == 7

    def test_data_roles_and_full_support_frames(self, tmp_path: Path) -> None:
        config = TrainConfig.from_yaml(CONFIG_DIR / "sharpa_motion_alignment_contact.yaml")
        spec = HANDS["sharpa"]
        pair = load_pair_supervision(config.data.pair_sources, spec)
        contact = load_contact_supervision(config.data.contact_sources, spec)
        assert pair.keypoints.shape == (6, 5, 3)
        assert pair.target_qpos.shape == (6, 22)
        torch.testing.assert_close(pair.sample_weight, torch.ones(6))
        weighted_config = TrainConfig.from_yaml(CONFIG_DIR / "sharpa_motion_alignment.yaml")
        weighted_pair = load_pair_supervision(weighted_config.data.pair_sources, spec)
        torch.testing.assert_close(weighted_pair.sample_weight, torch.tensor([1.5, 1.0, 1.0, 1.0, 1.0, 1.0]))
        xhand_config = TrainConfig.from_yaml(CONFIG_DIR / "xhand_motion_alignment.yaml")
        xhand_pair = load_pair_supervision(xhand_config.data.pair_sources, HANDS["xhand"])
        assert xhand_pair.keypoints.shape == (388, 5, 3)
        assert bool((xhand_pair.joint_mask[1:46, :3] == 0).all())
        assert bool((xhand_pair.joint_mask[1:46, 3:] == 1).all())
        assert bool((xhand_pair.joint_mask[46:383, :3] == 0).all())
        assert bool((xhand_pair.joint_mask[46:383, 3:6] == 1).all())
        assert bool((xhand_pair.joint_mask[46:383, 6:] == 0).all())
        torch.testing.assert_close(xhand_pair.sample_weight[1:46], torch.full((45,), 4.0))
        torch.testing.assert_close(xhand_pair.sample_weight[46:383], torch.full((337,), 0.2))
        allegro_config = TrainConfig.from_yaml(CONFIG_DIR / "allegro_motion_alignment.yaml")
        allegro_pair = load_pair_supervision(
            allegro_config.data.pair_sources,
            HANDS["allegro"],
        )
        assert allegro_pair.keypoints.shape == (407, 4, 3)
        assert bool((allegro_pair.joint_mask[1:41, :12] == 1).all())
        assert bool((allegro_pair.joint_mask[1:41, 12:] == 0).all())
        assert bool((allegro_pair.joint_mask[41:378, :4] == 1).all())
        assert bool((allegro_pair.joint_mask[41:378, 4:] == 0).all())
        torch.testing.assert_close(
            allegro_pair.joint_mask[378:382],
            torch.tensor(
                [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=torch.float32
            ).repeat_interleave(4, dim=1),
        )
        torch.testing.assert_close(allegro_pair.sample_weight[1:41], torch.full((40,), 4.0))
        torch.testing.assert_close(allegro_pair.sample_weight[41:378], torch.full((337,), 0.2))
        torch.testing.assert_close(allegro_pair.sample_weight[378:382], torch.full((4,), 10.0))
        torch.testing.assert_close(
            allegro_pair.joint_mask[382:406],
            torch.tensor(
                [[1, 0, 0, 0]] * 8 + [[0, 1, 0, 0]] * 6 + [[0, 0, 1, 0]] * 10,
                dtype=torch.float32,
            ).repeat_interleave(4, dim=1),
        )
        torch.testing.assert_close(allegro_pair.sample_weight[382:406], torch.full((24,), 2.0))
        assert contact.keypoints.shape == (53, 5, 3)
        assert contact.target_qpos.shape == (53, 22)
        torch.testing.assert_close(contact.sample_weight.sum(), torch.tensor(6.0))
        assert float(contact.target_qpos.abs().max()) <= 1.0

        frames = np.zeros((3, 21, 3), dtype=np.float32)
        frames[1, 4, 0] = 0.01
        frames[2, 4, 0] = 0.02
        path = tmp_path / "frames.npy"
        np.save(path, frames)
        np.random.seed(0)
        human = load_human_dataset([path], spec, training_samples=2, voxel_size_m=0.001)
        assert human.samples.shape == (2, 5, 3)
        assert human.support_frames.shape == (3, 5, 3)
        torch.testing.assert_close(human.support_frames[:, 0, 0], torch.tensor([0.0, 0.01, 0.02]))

    def test_cat_model_checkpoint(self, tmp_path: Path) -> None:
        spec = HANDS["sharpa"]
        model = IKModel(spec.finger_joint_sizes)
        model.networks = nn.ModuleList(
            _ConstantFinger(tuple(float(finger) for _ in names)) for finger, names in enumerate(spec.finger_joint_names)
        )
        output = model(torch.zeros(2, 5, 3))
        expected = torch.cat(
            [torch.full((2, len(names)), float(finger)) for finger, names in enumerate(spec.finger_joint_names)],
            dim=1,
        )
        torch.testing.assert_close(output, expected)

        checkpoint = tmp_path / "checkpoint"
        trained = IKModel(spec.finger_joint_sizes)
        checkpoint.mkdir()
        torch.save(trained.state_dict(), checkpoint / "weights.pt")
        (checkpoint / "train.yaml").write_text("hand: sharpa\n")
        retargeter = NeuralHandRetargeterConfig(checkpoint, spec).build("cpu")
        assert retargeter.retarget_batch(np.zeros((2, 21, 3), np.float32)).shape == (2, 22)
        assert {path.name for path in checkpoint.iterdir()} == {"weights.pt", "train.yaml"}

    def test_loss_scopes(self) -> None:
        config = TrainConfig.from_yaml(CONFIG_DIR / "sharpa_motion_alignment_contact.yaml").objective
        config = replace(
            config,
            pair=1.0,
            pair_position=1.0,
            contact=1.0,
            motion=replace(config.motion, weight=0.0),
            conformal=0.0,
            curvature=0.0,
            pinch=0.0,
            interfinger=0.0,
            collision=1.0,
        )
        pair = JointSupervision(
            torch.ones(1, 5, 3),
            torch.zeros(1, 22),
            torch.ones(1, 22),
            torch.ones(1),
        )
        contact = JointSupervision(
            torch.full((1, 5, 3), 0.5),
            torch.zeros(1, 22),
            torch.ones(1, 22),
            torch.ones(1),
        )
        collision_batches: list[int] = []

        def collision(qpos: torch.Tensor) -> torch.Tensor:
            collision_batches.append(len(qpos))
            return qpos.new_zeros((len(qpos), 1))

        context = ObjectiveContext(
            robot_keypoints=lambda qpos: qpos[:, (0, 4, 8, 12, 16), None].expand(-1, -1, 3),
            motion_workspace=None,
            finger_joint_sizes=(4, 4, 4, 4, 6),
            finger_pairs=(),
            collision_residual=collision,
            generator=torch.Generator().manual_seed(0),
        )
        terms = compute_objective(_InputModel(), torch.zeros(3, 5, 3), pair, contact, context, config)
        torch.testing.assert_close(terms.pair, torch.tensor(1.0))
        torch.testing.assert_close(terms.pair_position, torch.tensor(3.0))
        torch.testing.assert_close(terms.contact, torch.tensor(0.25))
        assert collision_batches == [3]

    def test_cosine_warmup_and_ema_checkpoint_roundtrip(self, tmp_path: Path) -> None:
        # Every shipped motion_alignment config takes this path; this const/no-EMA roundtrip does not.
        checkpoint = tmp_path / "checkpoint"
        config_path = tmp_path / "train.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "hand": "allegro",
                    "output": str(checkpoint),
                    "data": {
                        "unsupervised": ["mano_alex"],
                        "pair": [],
                        "contact": [],
                        "training_samples": 4,
                        "voxel_size_m": 0.003,
                    },
                    "robot_samples": 8,
                    "objective": {
                        "pair": 0.0,
                        "pair_position": 0.0,
                        "contact": 0.0,
                        "motion": {
                            "formula": "cosine",
                            "weight": 1.0,
                            "directions": 4,
                            "temperature": 0.05,
                            "perturb_min_m": 0.001,
                            "perturb_max_m": 0.011,
                        },
                        "conformal": 0.0,
                        "curvature": 0.0,
                        "pinch": 0.0,
                        "interfinger": 0.0,
                        "collision": 0.0,
                    },
                    "optimization": {
                        "epochs": 4,
                        "batch_size": 4,
                        "learning_rate": 0.01,
                        "schedule": "cosine",
                        "warmup_epochs": 2,
                        "ema_decay": 0.9,
                        "seed": 0,
                    },
                },
                sort_keys=False,
            )
        )

        assert train_from_config(config_path, "cpu") == checkpoint

        spec = HANDS["allegro"]
        retargeter = NeuralHandRetargeterConfig(checkpoint, spec).build("cpu")
        assert retargeter.retarget_batch(np.zeros((2, 21, 3), np.float32)).shape == (2, 16)
        weights = torch.load(checkpoint / "weights.pt", map_location="cpu", weights_only=True)
        assert all(bool(torch.isfinite(value).all()) for value in weights.values())

    def test_warmup_then_cosine_matches_the_closed_form_schedule(self) -> None:
        # The schedule is built from torch primitives; pin it to the formula it replaced.
        base, epochs, warmup = 1e-3, 12, 3
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = optim.AdamW([parameter], lr=base)
        schedule = SequentialLR(
            optimizer,
            [
                LinearLR(optimizer, start_factor=1.0 / warmup, total_iters=max(warmup - 1, 1)),
                CosineAnnealingLR(optimizer, T_max=epochs - warmup),
            ],
            milestones=[warmup],
        )

        actual = []
        for _ in range(epochs):
            actual.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            schedule.step()

        expected = [
            base * (epoch + 1) / warmup
            if epoch < warmup
            else base * 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup) / (epochs - warmup)))
            for epoch in range(epochs)
        ]
        torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected))

    def test_one_epoch_checkpoint_roundtrip(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "checkpoint"
        config_path = tmp_path / "train.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "hand": "allegro",
                    "output": str(checkpoint),
                    "data": {
                        "unsupervised": ["mano_alex"],
                        "pair": [],
                        "contact": [],
                        "training_samples": 4,
                        "voxel_size_m": 0.003,
                    },
                    "robot_samples": 8,
                    "objective": {
                        "pair": 0.0,
                        "pair_position": 0.0,
                        "contact": 0.0,
                        "motion": {
                            "formula": "cosine",
                            "weight": 1.0,
                            "directions": 4,
                            "temperature": 0.05,
                            "perturb_min_m": 0.001,
                            "perturb_max_m": 0.011,
                        },
                        "conformal": 0.0,
                        "curvature": 0.0,
                        "pinch": 0.0,
                        "interfinger": 0.0,
                        "collision": 0.0,
                    },
                    "optimization": {
                        "epochs": 1,
                        "batch_size": 4,
                        "learning_rate": 0.0001,
                        "schedule": "const",
                        "warmup_epochs": 0,
                        "ema_decay": 0.0,
                        "seed": 0,
                    },
                },
                sort_keys=False,
            )
        )

        assert train_from_config(config_path, "cpu") == checkpoint
        provenance = yaml.safe_load((checkpoint / "train.yaml").read_text())["provenance"]["asset_sha256"]
        assert list(provenance) == ["data/neural_hand/human/mano/alex.npy"]
        spec = HANDS["allegro"]
        retargeter = NeuralHandRetargeterConfig(checkpoint, spec).build("cpu")
        assert retargeter.retarget_batch(np.zeros((2, 21, 3), np.float32)).shape == (2, 16)
