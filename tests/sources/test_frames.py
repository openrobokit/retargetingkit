import numpy as np

from retargetingkit.sources.frames import MANO21, MANO21_JOINTS, ArmFrame, ArmHandFrame, HandFrame, HandPose


class TestHandFrame:
    def test_defaults(self) -> None:
        pose = HandPose(positions=np.zeros((21, 3), dtype=np.float32))
        frame = HandFrame({"right": pose})
        assert frame.hands["right"] is pose
        assert pose.positions.shape == (21, 3)
        assert pose.orientations.shape == (21, 4)
        assert pose.valid.shape == (21,)
        assert bool(pose.valid.all())
        assert pose.joint_names == MANO21_JOINTS

    def test_layout_consistent(self) -> None:
        assert len(MANO21_JOINTS) == 21
        assert len(set(MANO21_JOINTS)) == 21
        assert MANO21.WRIST == 0
        assert MANO21_JOINTS[MANO21.INDEX_TIP] == "index_tip"
        tips = (MANO21.THUMB_TIP, MANO21.INDEX_TIP, MANO21.MIDDLE_TIP, MANO21.RING_TIP, MANO21.PINKY_TIP)
        assert tips == (4, 8, 12, 16, 20)


class TestArmFrame:
    def test_arm_and_hand_observations_stay_independent(self) -> None:
        arm = ArmFrame({"right": np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)})
        hand = HandFrame({"right": HandPose(np.zeros((21, 3), dtype=np.float32))})
        frame = ArmHandFrame(arm, hand)

        assert frame.arm.wrists["right"].shape == (7,)
        assert frame.hand.hands["right"].positions.shape == (21, 3)
