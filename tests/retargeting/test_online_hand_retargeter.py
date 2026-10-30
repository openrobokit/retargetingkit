from dataclasses import replace

import numpy as np
import torch
import warp as wp

from robokit.assets.robots.end_effectors import inspire_hand
from robokit.helpers.hand_retargeting import HandRetargeterOnline
from robokit.helpers.hand_retargeting.presets import inspire
from robokit.robo import Robot


def _open_hand_points() -> np.ndarray:
    pts = np.zeros((21, 3), dtype=np.float32)
    for finger, x in enumerate([-0.04, -0.02, 0.0, 0.02, 0.04]):
        for j in range(4):
            pts[1 + finger * 4 + j] = (x, 0.02 * (j + 1), 0.0)
    return pts


class TestRobokitOnlineHandRetargeter:
    @classmethod
    def setup_class(cls) -> None:
        wp.init()
        cls.robot = Robot.load(
            inspire_hand.URDF_PATH_RIGHT,
            load_collision_spheres=True,
            collision_spheres_path=inspire_hand.COLLISION_SPHERE_PATH_RIGHT,
            self_collision_ignore_path=inspire_hand.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT,
        )
        cls.retargeter = HandRetargeterOnline(
            config=inspire.online,
            robot=cls.robot,
            spec=replace(inspire.spec, floating_base=False),
        )
        cls.retargeter.warmup(1)
        cls.limits = cls.robot.spec.actuated_joint_limits

    def test_output_shape_matches_joint_names(self) -> None:
        q = self.retargeter.solve_numpy(_open_hand_points())[0, 7:]
        assert q.shape == (len(self.robot.spec.actuated_joint_names),) == (6,)

    def test_output_within_limits(self) -> None:
        q = self.retargeter.solve_numpy(_open_hand_points())[0, 7:]
        assert np.all(q >= self.limits[:, 0] - 1e-5)
        assert np.all(q <= self.limits[:, 1] + 1e-5)

    def test_reset_is_deterministic(self) -> None:
        points = _open_hand_points()
        self.retargeter.reset()
        q1 = self.retargeter.solve_numpy(points)[0, 7:]
        self.retargeter.reset()
        q2 = self.retargeter.solve_numpy(points)[0, 7:]
        torch.testing.assert_close(torch.from_numpy(q1), torch.from_numpy(q2), rtol=1e-5, atol=1e-5)

    def test_left_preset(self) -> None:
        robot = Robot.load(inspire_hand.URDF_PATH_LEFT)
        retargeter = HandRetargeterOnline(
            config=inspire.online_left,
            robot=robot,
            spec=replace(inspire.spec_left, floating_base=False),
        )
        retargeter.warmup(1)
        assert retargeter.solve_numpy(_open_hand_points())[0, 7:].shape == (6,)
