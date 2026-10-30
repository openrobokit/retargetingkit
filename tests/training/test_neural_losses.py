import torch

from retargetingkit.training.hand.metrics import interfinger_loss, pinch_loss


class TestPinchLoss:
    def test_only_pinching_human_samples_contribute(self) -> None:
        # Sample 0 pinches (human gap 0.005 < 0.015), sample 1 does not (gap 0.05).
        human = torch.tensor([[[0.0, 0.0, 0.0], [0.005, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]]])
        robot = torch.tensor([[[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.30, 0.0, 0.0]]])

        actual = pinch_loss(human, robot, ((0, 1),), 0.015)

        # Only sample 0 is masked in: 0.02^2 / 1 == 4e-4. Sample 1's 0.3 gap is ignored.
        torch.testing.assert_close(actual, torch.tensor(4.0e-4), rtol=0.0, atol=1e-9)

    def test_no_pinch_samples_gives_zero(self) -> None:
        human = torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]])
        robot = torch.tensor([[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]]])

        actual = pinch_loss(human, robot, ((0, 1),), 0.015)

        torch.testing.assert_close(actual, torch.tensor(0.0), rtol=0.0, atol=1e-9)

    def test_each_finger_pair_adds_a_term(self) -> None:
        human = torch.zeros(1, 3, 3)
        robot = torch.zeros(1, 3, 3)
        robot[0, 1, 0] = 0.1
        robot[0, 2, 0] = 0.1

        one_pair = pinch_loss(human, robot, ((0, 1),), 0.015)
        two_pairs = pinch_loss(human, robot, ((0, 1), (0, 2)), 0.015)

        torch.testing.assert_close(two_pairs, 2.0 * one_pair)


class TestInterfingerLoss:
    def test_matching_order_is_free(self) -> None:
        # Robot gaps grow with human gaps by the same amount, so no margin is violated.
        human = torch.tensor([[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]]])
        robot = torch.tensor([[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]]])

        actual = interfinger_loss(human, robot, ((0, 1),), 0.3, 0.001)

        torch.testing.assert_close(actual, torch.tensor(0.0), rtol=0.0, atol=1e-9)

    def test_inverted_order_is_penalized(self) -> None:
        # Human says sample 1 is the wider pinch; the robot reverses it.
        human = torch.tensor([[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]]])
        robot = torch.tensor([[[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]])

        actual = interfinger_loss(human, robot, ((0, 1),), 0.3, 0.001)

        # human_difference=-0.04, robot_difference=+0.04, sign=-1; margin=max(0.3*0.04, 0.001)=0.012; relu(0.052).
        torch.testing.assert_close(actual, torch.tensor(0.052), rtol=0.0, atol=1e-6)

    def test_min_margin_floors_tiny_human_differences(self) -> None:
        # human_difference is 0, so margin falls back to min_margin_m.
        human = torch.zeros(2, 2, 3)
        robot = torch.zeros(2, 2, 3)

        actual = interfinger_loss(human, robot, ((0, 1),), 0.3, 0.004)

        torch.testing.assert_close(actual, torch.tensor(0.004), rtol=0.0, atol=1e-9)
