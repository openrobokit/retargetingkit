import ctypes
import multiprocessing as mp
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

import retargetingkit.targets.hands.xhand as xhand_module
from retargetingkit.targets.hands.xhand import NUM_JOINTS, XHand, XHandConfig


class TestXHandServo:
    def test_continuous_state_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        running = mp.Value(ctypes.c_bool, True)
        ready = mp.Value(ctypes.c_bool, False)
        target = mp.Array(ctypes.c_double, NUM_JOINTS)
        current = mp.Array(ctypes.c_double, NUM_JOINTS)
        error = SimpleNamespace(error_code=0, error_message="")
        sent_modes: list[tuple[int, ...]] = []
        sent_positions: list[tuple[float, ...]] = []
        force_updates: list[bool] = []
        read_count = 0

        def send_command(hand_id: int, command: Any) -> SimpleNamespace:
            assert hand_id == 7
            sent_modes.append(tuple(finger.mode for finger in command.finger_command))
            sent_positions.append(tuple(finger.position for finger in command.finger_command))
            return error

        def read_state(hand_id: int, force_update: bool) -> tuple[SimpleNamespace, SimpleNamespace]:
            nonlocal read_count
            assert hand_id == 7
            force_updates.append(force_update)
            read_count += 1
            if read_count == 3:
                running.value = False
            state = SimpleNamespace(
                finger_state=[SimpleNamespace(position=0.05 + read_count / 100 + i / 1000) for i in range(NUM_JOINTS)]
            )
            return error, state

        device = MagicMock()
        device.open_serial.return_value = error
        device.list_hands_id.return_value = [7]
        device.send_command.side_effect = send_command
        device.read_state.side_effect = read_state
        command = SimpleNamespace(finger_command=[SimpleNamespace() for _ in range(NUM_JOINTS)])
        xhand_control = SimpleNamespace(
            XHandControl=MagicMock(return_value=device),
            HandCommand_t=MagicMock(return_value=command),
        )
        monkeypatch.setitem(sys.modules, "xhand_controller", SimpleNamespace(xhand_control=xhand_control))
        monkeypatch.setattr(xhand_module, "initialize_xhand_sdk", MagicMock())
        monkeypatch.setattr(xhand_module.time, "sleep", MagicMock())

        XHand._servo_loop(XHandConfig("/dev/null"), target, current, running, ready)

        assert ready.value
        assert read_count == 3
        assert force_updates == [True, False, False]
        assert len(sent_modes) == 4
        assert sent_modes[0] == (3,) * NUM_JOINTS
        assert sent_modes[-1] == (0,) * NUM_JOINTS
        first_state = torch.tensor([0.06 + i / 1000 for i in range(NUM_JOINTS)], dtype=torch.float64)
        torch.testing.assert_close(torch.tensor(sent_positions[0], dtype=torch.float64), first_state)
        torch.testing.assert_close(torch.tensor(target[:], dtype=torch.float64), first_state)
        torch.testing.assert_close(
            torch.tensor(current[:], dtype=torch.float64),
            torch.tensor([0.08 + i / 1000 for i in range(NUM_JOINTS)], dtype=torch.float64),
        )
