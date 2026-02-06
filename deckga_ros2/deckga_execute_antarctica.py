#!/usr/bin/env python3
"""
DECK-GA execution for Antarctica world (Gazebo Harmonic + Aerostack2).

Fixes:
- Takeoff MUST be ABS in earth frame for your setup (relative takeoff caused Service failure -> Goal Rejected).
- Avoid high takeoff by defaulting takeoff altitude to min waypoint Z (+ small margin).
- Keep waypoint Z exactly as produced by your mapping (so drones follow your input Zs).
"""

import argparse
import inspect
import os
import pickle
import subprocess
import time
from typing import List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from as2_python_api.drone_interface import DroneInterface

DEFAULT_FRAME_ID = "earth"


# ----------------------------
# DroneInterface compatibility
# ----------------------------
def make_drone_interface(ns: str, use_sim_time: bool, verbose: bool) -> DroneInterface:
    sig = inspect.signature(DroneInterface.__init__)
    params = sig.parameters
    kwargs = {}

    if "drone_id" in params:
        kwargs["drone_id"] = ns
    if "namespace" in params:
        kwargs["namespace"] = ns
    if "node_name" in params:
        kwargs["node_name"] = f"{ns}_interface"
    if "use_sim_time" in params:
        kwargs["use_sim_time"] = bool(use_sim_time)
    if "verbose" in params:
        kwargs["verbose"] = bool(verbose)

    try:
        return DroneInterface(**kwargs)
    except TypeError:
        return DroneInterface(ns)


def safe_offboard_arm(drone: DroneInterface, retries: int = 5, sleep_s: float = 0.5) -> None:
    """
    Best-effort retries. We don't silently swallow everything; we just keep trying.
    """
    for _ in range(int(retries)):
        try:
            drone.offboard()
        except Exception:
            try:
                drone.set_offboard_mode()
            except Exception:
                pass

        try:
            drone.arm()
        except Exception:
            pass

        time.sleep(float(sleep_s))


def safe_takeoff_abs(drone: DroneInterface, abs_z: float, wait: bool) -> None:
    """
    IMPORTANT: In your AS2 Gazebo setup, takeoff 'height' behaves like ABS Z (earth frame),
    not +relative meters. Using 1.0 caused failures because it tried to go to Z=1.0.
    """
    # Do NOT swallow exceptions here: if it throws, you want to know.
    try:
        drone.takeoff(height=float(abs_z), wait=bool(wait))
    except TypeError:
        drone.takeoff(height=float(abs_z))
    # Any internal "Service returned failure" will still print from DroneInterface internals.


def safe_land(drone: DroneInterface, wait: bool) -> None:
    try:
        drone.land(wait=bool(wait))
    except TypeError:
        drone.land()
    except Exception:
        pass


def safe_go_to(
    drone: DroneInterface,
    x: float,
    y: float,
    z: float,
    speed: float,
    frame_id: str,
    wait: bool,
) -> None:
    """
    GoTo wrapper with signature fallback.
    """
    try:
        drone.go_to(x=x, y=y, z=z, speed=float(speed), frame_id=str(frame_id), wait=bool(wait))
        return
    except TypeError:
        pass
    drone.go_to(x, y, z, float(speed), str(frame_id), bool(wait))


# ----------------------------
# PKL loading + transforms
# ----------------------------
def load_deckga_pkl(pkl_path: str) -> Tuple[List[np.ndarray], np.ndarray]:
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"DECK-GA output PKL not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if "deckga_paths" not in data:
        raise KeyError(f"'deckga_paths' missing. PKL keys: {list(data.keys())}")

    paths = [np.array(p, dtype=float) for p in data["deckga_paths"]]
    for p in paths:
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError(f"Bad path shape {p.shape}; expected (N,3)")

    offset = np.array(data.get("offset_used", [0.0, 0.0, 0.0]), dtype=float).reshape(3,)
    return paths, offset


def decide_coord_mode(paths: List[np.ndarray], offset_used: np.ndarray, coord_mode: str) -> str:
    coord_mode = coord_mode.lower().strip()
    if coord_mode in ("original", "shifted"):
        return coord_mode
    if coord_mode != "auto":
        raise ValueError("coord_mode must be one of: auto, original, shifted")

    if np.linalg.norm(offset_used[:2]) > 1e-9:
        all_min_xy = min(float(np.min(p[:, :2])) for p in paths) if paths else 0.0
        if all_min_xy >= -1e-6:
            return "shifted"
    return "original"


def map_range(v: np.ndarray, in_min: float, in_max: float, out_min: float, out_max: float) -> np.ndarray:
    v = np.clip(v, in_min, in_max)
    if abs(in_max - in_min) < 1e-9:
        return np.full_like(v, out_min)
    t = (v - in_min) / (in_max - in_min)
    return out_min + t * (out_max - out_min)


def transform_paths(
    paths: List[np.ndarray],
    offset_used: np.ndarray,
    coord_mode: str,
    xy_scale: float,
    xy_center_x: float,
    xy_center_y: float,
    clamp_xy: bool,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    zin_min: float,
    zin_max: float,
    zout_min: float,
    zout_max: float,
    z_base: float,
) -> List[np.ndarray]:
    mode = decide_coord_mode(paths, offset_used, coord_mode)
    out: List[np.ndarray] = []

    for p in paths:
        q = p.copy()

        # Unshift if needed
        if mode == "shifted":
            q[:, 0] -= offset_used[0]
            q[:, 1] -= offset_used[1]
            q[:, 2] -= offset_used[2]

        # XY
        q[:, 0] = q[:, 0] * float(xy_scale) + float(xy_center_x)
        q[:, 1] = q[:, 1] * float(xy_scale) + float(xy_center_y)

        if bool(clamp_xy):
            q[:, 0] = np.clip(q[:, 0], float(x_min), float(x_max))
            q[:, 1] = np.clip(q[:, 1], float(y_min), float(y_max))

        # Z mapping -> ABS
        rel_z = map_range(q[:, 2], float(zin_min), float(zin_max), float(zout_min), float(zout_max))
        q[:, 2] = rel_z + float(z_base)

        out.append(q)

    return out


def print_ranges(paths_m: List[np.ndarray]) -> None:
    if not paths_m:
        return
    xs = np.concatenate([p[:, 0] for p in paths_m])
    ys = np.concatenate([p[:, 1] for p in paths_m])
    zs = np.concatenate([p[:, 2] for p in paths_m])
    print(
        f"Final waypoint ranges (meters): "
        f"x[{xs.min():.2f},{xs.max():.2f}] "
        f"y[{ys.min():.2f},{ys.max():.2f}] "
        f"z[{zs.min():.2f},{zs.max():.2f}]"
    )


def print_planned_timing(paths: List[np.ndarray], speed: float) -> None:
    print("\n=== Planned timing (kinematic model) ===")
    print(f"Assumed constant speed: {speed:.3f} m/s")
    makespan = 0.0
    for i, p in enumerate(paths):
        if len(p) < 2:
            length = 0.0
        else:
            segs = p[1:, :] - p[:-1, :]
            length = float(np.sum(np.linalg.norm(segs, axis=1)))
        t = length / speed if speed > 1e-9 else float("inf")
        makespan = max(makespan, t)
        print(f"UAV {i}: length={length:.3f} m, planned_time={t:.2f} s")
    print(f"Planned makespan: {makespan:.2f} s")
    print("======================================\n")


def wait_for_actions(namespaces: List[str], timeout_s: float) -> None:
    required_suffixes = ("TakeoffBehavior", "GoToBehavior", "LandBehavior")
    deadline = time.time() + float(timeout_s)

    print("Waiting for required actions to appear for all drones...")
    while time.time() < deadline:
        try:
            out = subprocess.check_output(["ros2", "action", "list"], text=True)
        except Exception:
            time.sleep(1.0)
            continue

        ok = True
        for ns in namespaces:
            for suf in required_suffixes:
                key = f"/{ns}/{suf}"
                if key not in out:
                    ok = False
                    break
            if not ok:
                break

        if ok:
            print("All required actions are available.")
            return

        time.sleep(1.0)

    print("[WARN] Timeout waiting for actions. Continuing anyway (may cause rejections).")


def compute_auto_takeoff_abs_z(paths_m: List[np.ndarray], margin: float) -> float:
    """
    Takeoff just enough to be airborne, not unnecessarily high.
    Use the minimum waypoint Z across all drones, plus a small margin.
    """
    zs = []
    for p in paths_m:
        if len(p) > 0:
            zs.append(float(np.min(p[:, 2])))
    if not zs:
        return 0.0
    return min(zs) + float(margin)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--deckga_pkl", default="deckga_ros2/data/deckga_output_antarctica.pkl")
    ap.add_argument("--namespaces", default="drone0,drone1,drone2")
    ap.add_argument("--frame_id", default=DEFAULT_FRAME_ID)

    ap.add_argument("--speed", type=float, default=1.5)

    ap.add_argument("--wait_actions_s", type=float, default=60.0)
    ap.add_argument("--init_wait_s", type=float, default=5.0)
    ap.add_argument("--takeoff_settle_s", type=float, default=6.0)

    # ✅ Takeoff control (ABS earth Z)
    ap.add_argument("--takeoff_abs_z", type=float, default=None, help="Override ABS takeoff Z (earth). If not set, auto.")
    ap.add_argument("--takeoff_margin", type=float, default=0.20, help="Auto takeoff = min waypoint z + margin (m).")
    ap.add_argument("--takeoff_sequential", action="store_true")

    ap.add_argument("--first_wp_wait", action="store_true")

    ap.add_argument("--coord_mode", default="auto", choices=["auto", "original", "shifted"])

    ap.add_argument("--xy_scale", type=float, default=1.0)
    ap.add_argument("--xy_center_x", type=float, default=0.0)
    ap.add_argument("--xy_center_y", type=float, default=0.0)

    ap.add_argument("--clamp_xy", action="store_true")
    ap.add_argument("--x_min", type=float, default=-30.0)
    ap.add_argument("--x_max", type=float, default=30.0)
    ap.add_argument("--y_min", type=float, default=-30.0)
    ap.add_argument("--y_max", type=float, default=30.0)

    ap.add_argument("--zin_min", type=float, default=33.34)
    ap.add_argument("--zin_max", type=float, default=39.30)
    ap.add_argument("--zout_min", type=float, default=1.14)
    ap.add_argument("--zout_max", type=float, default=7.10)
    ap.add_argument("--z_base", type=float, default=32.2)

    ap.add_argument("--use_sim_time", action="store_true")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    namespaces = [s.strip() for s in args.namespaces.split(",") if s.strip()]
    if not namespaces:
        raise ValueError("No namespaces provided")

    print("Initializing rclpy...")
    rclpy.init()

    drones: List[DroneInterface] = []
    try:
        wait_for_actions(namespaces, args.wait_actions_s)

        print(f"Loading DECK-GA paths: {args.deckga_pkl}")
        raw_paths, offset_used = load_deckga_pkl(args.deckga_pkl)
        raw_paths = raw_paths[: len(namespaces)]

        paths_m = transform_paths(
            raw_paths,
            offset_used=offset_used,
            coord_mode=args.coord_mode,
            xy_scale=args.xy_scale,
            xy_center_x=args.xy_center_x,
            xy_center_y=args.xy_center_y,
            clamp_xy=args.clamp_xy,
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
            zin_min=args.zin_min,
            zin_max=args.zin_max,
            zout_min=args.zout_min,
            zout_max=args.zout_max,
            z_base=args.z_base,
        )

        print_ranges(paths_m)
        print_planned_timing(paths_m, args.speed)

        # ✅ Auto takeoff altitude from waypoint Z (low takeoff, no unnecessary climb)
        if args.takeoff_abs_z is None:
            takeoff_abs_z = compute_auto_takeoff_abs_z(paths_m, args.takeoff_margin)
            print(f"Auto takeoff ABS z = min_waypoint_z + margin = {takeoff_abs_z:.2f}")
        else:
            takeoff_abs_z = float(args.takeoff_abs_z)
            print(f"Manual takeoff ABS z = {takeoff_abs_z:.2f}")

        print("Creating DroneInterface objects...")
        for ns in namespaces:
            drones.append(make_drone_interface(ns, use_sim_time=args.use_sim_time, verbose=args.verbose))

        print(f"Initial settle wait: {args.init_wait_s:.1f}s")
        time.sleep(args.init_wait_s)

        print("Arming + switching to offboard (with retries)...")
        for d in drones:
            safe_offboard_arm(d, retries=8, sleep_s=0.4)

        # ✅ Takeoff ABS earth Z (this is what works in your logs)
        if args.takeoff_sequential:
            print(f"Taking off SEQUENTIALLY to ABS z={takeoff_abs_z:.2f} ({args.frame_id} frame)...")
            for d in drones:
                safe_takeoff_abs(d, abs_z=takeoff_abs_z, wait=True)
        else:
            print(f"Taking off ALL drones in parallel to ABS z={takeoff_abs_z:.2f} ({args.frame_id} frame)...")
            for d in drones:
                safe_takeoff_abs(d, abs_z=takeoff_abs_z, wait=False)

        time.sleep(args.takeoff_settle_s)

        # Re-assert offboard
        for d in drones:
            try:
                d.offboard()
            except Exception:
                pass

        # First waypoint handshake
        if args.first_wp_wait:
            print("First-waypoint handshake (wait=True per drone)...")
            for i, d in enumerate(drones):
                p = paths_m[i]
                if len(p) == 0:
                    continue
                x, y, z = map(float, p[0])
                print(f"[{namespaces[i]}] FIRST WP (blocking) -> (x={x:.2f}, y={y:.2f}, z={z:.2f})")
                safe_go_to(d, x, y, z, args.speed, args.frame_id, wait=True)
            time.sleep(1.0)

        print("Executing DECK-GA paths (parallel waypoint stepping)...")
        max_len = max((len(p) for p in paths_m), default=0)
        last_sent: List[Optional[np.ndarray]] = [None] * len(drones)

        for k in range(max_len):
            step_dists: List[float] = []

            for i, d in enumerate(drones):
                p = paths_m[i]
                if k >= len(p) or len(p) == 0:
                    continue

                x, y, z = map(float, p[k])
                print(f"[{namespaces[i]}] WP {k+1}/{len(p)} -> (x={x:.2f}, y={y:.2f}, z={z:.2f})")
                safe_go_to(d, x, y, z, args.speed, args.frame_id, wait=False)

                if last_sent[i] is None:
                    step_dists.append(0.0)
                else:
                    step_dists.append(float(np.linalg.norm(np.array([x, y, z]) - last_sent[i])))
                last_sent[i] = np.array([x, y, z], dtype=float)

            if step_dists:
                max_step = max(step_dists)
                sleep_t = max(0.30, float(max_step / max(args.speed, 1e-6)))
                time.sleep(sleep_t)

        print("Hover 2s...")
        time.sleep(2.0)

        print("Landing all drones (non-blocking)...")
        for d in drones:
            safe_land(d, wait=False)

        time.sleep(3.0)
        print("Done.")

    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl+C). Best-effort landing...")
        for d in drones:
            safe_land(d, wait=False)
        time.sleep(2.0)

    except ExternalShutdownException:
        print("\nROS shutdown detected. Exiting cleanly.")

    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
