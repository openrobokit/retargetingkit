"""Read and print XHand joint positions."""

import argparse
import sys
import time

import numpy as np
from jaxtyping import Float64

from retargetingkit.targets.hands.xhand import JOINT_NAMES, NUM_JOINTS, initialize_xhand_sdk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-port", default="/dev/ttyUSB1")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--samples", type=int, default=0)
    args = parser.parse_args()

    initialize_xhand_sdk()
    from xhand_controller import xhand_control  # type: ignore[import-unresolved]

    device = xhand_control.XHandControl()
    response = device.open_serial(args.serial_port, 3_000_000)
    assert response.error_code == 0, f"open_serial failed: code={response.error_code}, {response.error_message}"
    hands = device.list_hands_id()
    assert hands, "no XHand devices found"
    print(f"hand_id={hands[0]} joint_names={JOINT_NAMES}")
    time.sleep(0.5)

    for _ in range(args.samples or sys.maxsize):
        error, state = device.read_state(hands[0], True)
        assert error.error_code == 0, f"read_state failed: code={error.error_code}, {error.error_message}"
        qpos: Float64[np.ndarray, "12"] = np.asarray(
            [state.finger_state[index].position for index in range(NUM_JOINTS)]
        )
        assert qpos.shape == (NUM_JOINTS,), f"qpos shape must be ({NUM_JOINTS},), got {qpos.shape}"
        assert qpos.dtype == np.float64, f"qpos dtype must be float64, got {qpos.dtype}"
        assert np.isfinite(qpos).all(), "qpos must be finite"
        print(f"qpos_rad={qpos.tolist()} shape={qpos.shape} dtype={qpos.dtype} finite=True")
        time.sleep(1.0 / args.hz)
    device.close_device()


if __name__ == "__main__":
    main()
