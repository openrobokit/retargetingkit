"""Benchmark AbilityHand control modes: sweep ramp rate and servo hz for POS_RAMP and VEL_RAMP."""

import argparse
import signal
import sys
import time

import numpy as np
from scipy.signal import butter, sosfiltfilt  # type: ignore[import-untyped]

from retargetingkit.targets.hands.ability_hand import AbilityHand, AbilityHandConfig, ControlMode


OPEN = np.deg2rad([9, 9, 9, 9, 9, -9]).astype(np.float64)
CLOSED = np.deg2rad([70, 70, 70, 70, 70, -70]).astype(np.float64)
SAMPLE_DT = 0.005  # 200 Hz sampling
FILTER_CUTOFF_HZ = 10.0
SETTLE_THRESH = 0.05  # rad


def compute_metrics(
    log: list[tuple[float, np.ndarray, np.ndarray]],
) -> tuple[float, float, float]:
    """Return (rms_error, mean_jerk, mean_settle_time) from a recorded trace."""
    times = np.array([t for t, _, _ in log])
    targets = np.array([tgt for _, tgt, _ in log])
    actuals = np.array([act for _, _, act in log])

    rms_error = float(np.sqrt(np.mean((targets - actuals) ** 2)))

    # Low-pass filter before jerk to be fair across servo rates
    dt = float(np.median(np.diff(times)))
    sos = butter(4, FILTER_CUTOFF_HZ, btype="low", fs=1.0 / dt, output="sos")
    jerks = []
    for j in range(actuals.shape[1]):
        filtered = sosfiltfilt(sos, actuals[:, j])
        jerks.append(float(np.mean(np.abs(np.diff(filtered, n=3) / dt**3))))
    mean_jerk = float(np.mean(jerks))

    # Settle time per target transition
    settle_times = []
    for i in range(1, len(times)):
        if not np.allclose(targets[i], targets[i - 1]):
            target = targets[i]
            for k in range(i, len(times)):
                if np.max(np.abs(actuals[k] - target)) < SETTLE_THRESH:
                    settle_times.append(times[k] - times[i])
                    break
            else:
                settle_times.append(times[-1] - times[i])

    return rms_error, mean_jerk, float(np.mean(settle_times)) if settle_times else 0.0


def main():
    parser = argparse.ArgumentParser(description="Benchmark AbilityHand control modes")
    parser.add_argument(
        "--port",
        default="/dev/serial/by-id/usb-Prolific_Technology_Inc._USB-Serial_Controller_BUBZb11A922-if00-port0",
    )
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--ramp-rates", type=float, nargs="+", default=[15, 30, 60, 120])
    parser.add_argument("--servo-rates", type=float, nargs="+", default=[40, 80, 160])
    args = parser.parse_args()

    configs: list[tuple[str, AbilityHandConfig]] = []
    for hz in args.servo_rates:
        for ramp in args.ramp_rates:
            for mode in ControlMode:
                configs.append((mode.name, AbilityHandConfig(args.port, dt=1.0 / hz, mode=mode, ramp_deg_per_s=ramp)))

    results: list[tuple[str, float, float, float, float, float]] = []
    for idx, (label, config) in enumerate(configs):
        hz = 1.0 / config.dt
        print(f"\n[{idx + 1}/{len(configs)}] {label}  ramp={config.ramp_deg_per_s}  hz={hz:.0f}")
        hand = AbilityHand(config)
        hand.start()
        time.sleep(0.5)
        signal.signal(signal.SIGINT, lambda *_: (hand.close(), sys.exit(1)))

        log: list[tuple[float, np.ndarray, np.ndarray]] = []
        hand.set_qpos(OPEN)
        time.sleep(0.5)
        for rep in range(args.reps):
            for name, target in [("close", CLOSED), ("open", OPEN)]:
                print(f"    rep {rep + 1}/{args.reps}: {name}")
                hand.set_qpos(target)
                t0 = time.time()
                while time.time() - t0 < 3.0:
                    now = time.time()
                    log.append((now, target.copy(), hand.get_qpos()))
                    time.sleep(max(0, SAMPLE_DT - (time.time() - now)))

        hand.close()
        rms_err, mean_jerk, settle = compute_metrics(log)
        results.append((label, config.ramp_deg_per_s, hz, rms_err, mean_jerk, settle))
        print(f"    rms={rms_err:.4f}  jerk={mean_jerk:.1f}  settle={settle:.3f}")
        time.sleep(1.0)

    print("\n" + "=" * 75)
    print(f"{'mode':>10} {'ramp':>5} {'hz':>4}  {'rms_err':>8} {'jerk':>10} {'settle':>7}")
    print("-" * 75)
    for label, ramp, hz, err, jerk, settle in results:
        print(f"{label:>10} {ramp:>5.0f} {hz:>4.0f}  {err:>8.4f} {jerk:>10.1f} {settle:>7.3f}")


if __name__ == "__main__":
    main()
