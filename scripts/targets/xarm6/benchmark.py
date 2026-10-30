"""Benchmark XArm6 control modes: sweep max_velocity and kp scales for SERVO and VELOCITY."""

import argparse
import signal
import sys
import time

import numpy as np
from scipy.signal import butter, sosfiltfilt  # type: ignore[import-untyped]

from retargetingkit.targets.arms.xarm6 import XArm6, XArm6Config, XArmMode


DISTANCE = 0.10  # 10 cm
SAMPLE_DT = 0.004  # 250 Hz
FILTER_CUTOFF_HZ = 5.0
SETTLE_THRESH = 0.005  # 5 mm
HOLD_TIME = 3.0

DEFAULT_KP = np.array([10.0, 10.0, 5.0, 5.0, 5.0, 5.0])
DEFAULT_KD = np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25])
DEFAULT_MAX_VEL = np.array([0.8, 0.8, 0.8, 1.0, 1.0, 3.0])


def compute_metrics(
    log: list[tuple[float, np.ndarray, np.ndarray]],
) -> tuple[float, float, float]:
    """Return (rms_error, mean_jerk, mean_settle_time) from EE position traces."""
    times = np.array([t for t, _, _ in log])
    targets = np.array([tgt for _, tgt, _ in log])
    actuals = np.array([act for _, _, act in log])

    rms_error = float(np.sqrt(np.mean((targets - actuals) ** 2)))

    dt = float(np.median(np.diff(times)))
    sos = butter(4, FILTER_CUTOFF_HZ, btype="low", fs=1.0 / dt, output="sos")
    jerks = []
    for j in range(actuals.shape[1]):
        filtered = sosfiltfilt(sos, actuals[:, j])
        jerks.append(float(np.mean(np.abs(np.diff(filtered, n=3) / dt**3))))
    mean_jerk = float(np.mean(jerks))

    settle_times: list[float] = []
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
    parser = argparse.ArgumentParser(description="Benchmark XArm6 control modes")
    parser.add_argument("--ip", default="192.168.1.228")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--kp-scales", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--max-vel-scales", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    args = parser.parse_args()

    configs: list[tuple[str, float, float, XArm6Config]] = []
    for mv_s in args.max_vel_scales:
        configs.append(
            ("SERVO", 0.0, mv_s, XArm6Config(args.ip, mode=XArmMode.SERVO, max_velocity=DEFAULT_MAX_VEL * mv_s))
        )
    for kp_s in args.kp_scales:
        for mv_s in args.max_vel_scales:
            configs.append(
                (
                    "VELOCITY",
                    kp_s,
                    mv_s,
                    XArm6Config(
                        args.ip,
                        mode=XArmMode.VELOCITY,
                        kp=DEFAULT_KP * kp_s,
                        kd=DEFAULT_KD * kp_s,
                        max_velocity=DEFAULT_MAX_VEL * mv_s,
                    ),
                )
            )

    axes = [("Z", 2), ("Y", 1)]
    results: list[tuple[str, float, float, str, float, float, float]] = []

    for idx, (label, kp_s, mv_s, config) in enumerate(configs):
        print(f"\n[{idx + 1}/{len(configs)}] {label}  kp_s={kp_s:.1f}  mv_s={mv_s:.1f}")
        arm = XArm6(config)
        arm.start()
        time.sleep(0.5)
        signal.signal(signal.SIGINT, lambda *_: (arm.close(), sys.exit(1)))

        home_pose = arm.get_ee_pose()

        for axis_name, axis_idx in axes:
            offset_pose = home_pose.copy()
            offset_pose[axis_idx] += DISTANCE

            log: list[tuple[float, np.ndarray, np.ndarray]] = []
            arm.set_qpos(arm.home_qpos)
            time.sleep(0.5)

            for rep in range(args.reps):
                for direction, target_pose in [("out", offset_pose), ("home", home_pose)]:
                    print(f"    {axis_name} rep {rep + 1}/{args.reps}: {direction}")
                    arm.set_eepose(target_pose)
                    target_xyz = target_pose[:3].copy()
                    t0 = time.time()
                    while time.time() - t0 < HOLD_TIME:
                        now = time.time()
                        log.append((now, target_xyz, arm.get_ee_pose()[:3]))
                        time.sleep(max(0, SAMPLE_DT - (time.time() - now)))

            rms_err, mean_jerk, settle = compute_metrics(log)
            results.append((label, kp_s, mv_s, axis_name, rms_err, mean_jerk, settle))
            print(f"    {axis_name}: rms={rms_err:.5f}  jerk={mean_jerk:.1f}  settle={settle:.3f}")

        arm.close()
        time.sleep(1.0)

    print("\n" + "=" * 75)
    print(f"{'mode':>10} {'kp_sc':>5} {'mv_sc':>5} {'axis':>4}  {'rms(m)':>8} {'jerk(m/s3)':>10} {'settle(s)':>9}")
    print("-" * 75)
    for label, kp_s, mv_s, axis, err, jerk, settle in results:
        kp_str = f"{kp_s:.1f}" if kp_s > 0 else "-"
        print(f"{label:>10} {kp_str:>5} {mv_s:>5.1f} {axis:>4}  {err:>8.5f} {jerk:>10.1f} {settle:>9.3f}")


if __name__ == "__main__":
    main()
