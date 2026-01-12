#!/usr/bin/env python3
"""
deckga_execute.py

DECK-GA path execution for Aerostack2 (multi-UAV).

Primary requirement:
- RViz visualization and execution MUST apply the exact same coordinate transform.

Transform pipeline (must match rviz_paths_node.py):
    (A) Optional unshift (subtract offset_used) depending on coord_mode
    (B) Scaling: XY and Z
    (C) Z offset and Z minimum clamp

This version hardcodes the defaults you want so you can run only:
    python3 deckga_execute.py --deckga_pkl <.../deckga_output.pkl>

Default transform and flight params (match RViz):
    coord_mode = "original"
    scale_xy   = 0.05
    scale_z    = 0.05
    z_offset   = 2.5
    z_min      = 3.0
    takeoff_z  = 3.0
    takeoff_wait = True
    speed      = 1.2
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import rclpy
from as2_python_api.drone_interface import DroneInterface


# =========================
# Hardcoded defaults (match RViz)
# =========================

DEFAULT_DECKGA_PKL = "/home/dipraj/Documents/GitHub/ROS2_MultiUAV_3D/deckga_ros2/data/deckga_output.pkl"

DEFAULT_COORD_MODE = "original"   # "original" | "shifted" | "auto"
DEFAULT_SCALE_XY = 0.05
DEFAULT_SCALE_Z = 0.05
DEFAULT_Z_OFFSET = 2.5
DEFAULT_Z_MIN = 3.0               # ensures all commanded Z >= 3m

DEFAULT_TAKEOFF_Z = 3.0
DEFAULT_TAKEOFF_WAIT = True
DEFAULT_SPEED = 1.2

DEFAULT_FRAME_ID = "earth"
DEFAULT_TOPIC = "/deckga/markers"  # not used here, kept for parity/clarity


# =========================
# Transform (shared logic)
# =========================

def _stack_all_points(paths: Sequence[np.ndarray]) -> np.ndarray:
    pts: List[np.ndarray] = []
    for p in paths:
        if p.size == 0:
            continue
        pts.append(p)
    if not pts:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(pts)


def _auto_is_shifted(all_pts: np.ndarray, offset_used: Optional[np.ndarray]) -> bool:
    """
    Heuristic:
      - If all X/Y are non-negative and the mean is noticeably positive relative to offset_used,
        treat paths as shifted and subtract offset_used once.
    """
    if offset_used is None or all_pts.shape[0] == 0:
        return False

    min_xy = all_pts[:, :2].min(axis=0)
    mean_xy = all_pts[:, :2].mean(axis=0)

    if min_xy[0] < -1e-6 or min_xy[1] < -1e-6:
        return False

    return (mean_xy[0] >= 0.20 * offset_used[0]) or (mean_xy[1] >= 0.20 * offset_used[1])


@dataclass(frozen=True)
class TransformConfig:
    coord_mode: str           # "original" | "shifted" | "auto"
    scale_xy: float
    scale_z: float
    z_offset: float
    z_min: float


def transform_paths(
    raw_paths: Sequence[Any],
    offset_used: Optional[np.ndarray],
    tf: TransformConfig,
) -> List[np.ndarray]:
    """
    Apply the EXACT same transform used by rviz_paths_node.py.

    Returns:
        List[np.ndarray], each Nx3 in Gazebo/Aerostack coordinates.
    """
    paths_np: List[np.ndarray] = []
    for p in raw_paths:
        arr = np.asarray(p, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Each path must be Nx3. Got shape {arr.shape}")
        paths_np.append(arr)

    all_pts = _stack_all_points(paths_np)

    if tf.coord_mode == "shifted":
        do_unshift = True
    elif tf.coord_mode == "original":
        do_unshift = False
    else:
        do_unshift = _auto_is_shifted(all_pts, offset_used)

    out: List[np.ndarray] = []
    for arr in paths_np:
        a = arr.copy()

        if do_unshift and offset_used is not None:
            a = a - offset_used

        a[:, 0] *= float(tf.scale_xy)
        a[:, 1] *= float(tf.scale_xy)
        a[:, 2] *= float(tf.scale_z)

        a[:, 2] += float(tf.z_offset)

        if float(tf.z_min) > 0.0:
            a[:, 2] = np.maximum(a[:, 2], float(tf.z_min))

        out.append(a)

    return out


def enforce_closed_tour(path: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Ensure the tour ends at the first waypoint (so RViz + execution match)."""
    if path.size == 0:
        return path
    p0 = path[0]
    pN = path[-1]
    if np.linalg.norm(p0 - pN) > eps:
        path = np.vstack([path, p0])
    return path


# =========================
# IO
# =========================

def load_deckga_output(path: str) -> Tuple[List[Any], Optional[np.ndarray]]:
    with open(path, "rb") as f:
        data: Dict[str, Any] = pickle.load(f)

    if "deckga_paths" not in data:
        raise KeyError(f"'{path}' missing key 'deckga_paths'. Keys: {list(data.keys())}")

    raw_paths = data["deckga_paths"]
    offset_used = data.get("offset_used", None)

    if offset_used is not None:
        offset_used = np.asarray(offset_used, dtype=float).reshape(3,)

    return list(raw_paths), offset_used


# =========================
# Aerostack helpers
# =========================

def safe_offboard_arm(d: DroneInterface) -> None:
    d.offboard()
    d.arm()


def safe_takeoff(d: DroneInterface, height: float, wait: bool) -> None:
    try:
        d.takeoff(height=height, wait=wait)
        return
    except TypeError:
        pass

    try:
        d.takeoff(height=height)
        return
    except TypeError:
        pass

    d.takeoff(height)


def safe_go_to(
    d: DroneInterface,
    x: float,
    y: float,
    z: float,
    speed: float,
    frame_id: str,
    wait: bool,
) -> None:
    try:
        d.go_to(x=x, y=y, z=z, speed=speed, frame_id=frame_id, wait=wait)
        return
    except TypeError:
        pass

    try:
        d.go_to(x, y, z, speed=speed, frame_id=frame_id, wait=wait)
        return
    except TypeError:
        pass

    d.go_to(x, y, z)


def safe_land_disarm(d: DroneInterface, wait: bool) -> None:
    try:
        d.land(wait=wait)
    except TypeError:
        try:
            d.land()
        except Exception:
            pass
    try:
        d.disarm()
    except Exception:
        pass


def shutdown_drone(d: DroneInterface) -> None:
    try:
        d.shutdown()
    except Exception:
        pass


# =========================
# Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # You want to run with only --deckga_pkl, so everything else has correct defaults.
    parser.add_argument("--deckga_pkl", default=DEFAULT_DECKGA_PKL)
    parser.add_argument("--num_uavs", type=int, default=3)
    parser.add_argument("--uav_prefix", default="drone")

    # Accept both --frame_id and user-friendly --frame
    parser.add_argument("--frame_id", default=DEFAULT_FRAME_ID)
    parser.add_argument("--frame", dest="frame_id", default=argparse.SUPPRESS)

    parser.add_argument(
        "--coord_mode",
        default=DEFAULT_COORD_MODE,
        choices=["auto", "shifted", "original"],
    )

    # Legacy + new scaling options
    parser.add_argument("--scale", type=float, default=None, help="Legacy: sets BOTH --scale_xy and --scale_z")
    parser.add_argument("--scale_xy", type=float, default=DEFAULT_SCALE_XY)
    parser.add_argument("--scale_z", type=float, default=DEFAULT_SCALE_Z)

    parser.add_argument("--z_offset", type=float, default=DEFAULT_Z_OFFSET)
    parser.add_argument("--z_min", type=float, default=DEFAULT_Z_MIN)

    parser.add_argument("--takeoff_z", type=float, default=DEFAULT_TAKEOFF_Z)
    parser.add_argument("--takeoff_wait", action=argparse.BooleanOptionalAction, default=DEFAULT_TAKEOFF_WAIT)
    parser.add_argument("--init_wait_s", type=float, default=5.0)

    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    parser.add_argument("--hover_s", type=float, default=2.0)

    # Timing / pacing
    parser.add_argument("--fixed_dt", action="store_true", help="Disable auto pacing; use constant --dt_s.")
    parser.add_argument("--dt_s", type=float, default=2.5, help="Fixed dt (if --fixed_dt) or minimum dt (auto).")
    parser.add_argument("--dt_margin", type=float, default=0.5, help="Extra seconds added in auto pacing.")
    parser.add_argument("--dt_max", type=float, default=10.0, help="Maximum sleep per step in auto pacing.")

    parser.add_argument("--use_sim_time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    if args.scale is not None:
        args.scale_xy = float(args.scale)
        args.scale_z = float(args.scale)

    tf = TransformConfig(
        coord_mode=str(args.coord_mode),
        scale_xy=float(args.scale_xy),
        scale_z=float(args.scale_z),
        z_offset=float(args.z_offset),
        z_min=float(args.z_min),
    )

    print("Initializing rclpy...")
    rclpy.init()

    drones: List[DroneInterface] = []
    try:
        print(f"Loading DECK-GA paths: {args.deckga_pkl}")
        raw_paths, offset_used = load_deckga_output(args.deckga_pkl)

        paths_m = transform_paths(raw_paths=raw_paths, offset_used=offset_used, tf=tf)
        paths_m = [enforce_closed_tour(p) for p in paths_m]

        if len(paths_m) < args.num_uavs:
            for _ in range(args.num_uavs - len(paths_m)):
                paths_m.append(np.zeros((0, 3), dtype=float))

        print("Creating DroneInterface objects...")
        for i in range(args.num_uavs):
            ns = f"{args.uav_prefix}{i}"
            drones.append(
                DroneInterface(
                    drone_id=ns,
                    verbose=bool(args.verbose),
                    use_sim_time=bool(args.use_sim_time),
                )
            )

        print(f"Waiting {args.init_wait_s:.1f} seconds for behavior servers...")
        time.sleep(float(args.init_wait_s))

        print("Arming + switching to offboard for all drones...")
        for d in drones:
            safe_offboard_arm(d)

        print(f"Taking off all drones to {args.takeoff_z:.2f} m (wait={bool(args.takeoff_wait)})...")
        for d in drones:
            safe_takeoff(d, height=float(args.takeoff_z), wait=bool(args.takeoff_wait))

        time.sleep(1.0)

        print("Executing DECK-GA paths (execution transform is identical to RViz transform)...")
        max_len = max((len(p) for p in paths_m), default=0)

        last_sent: List[Optional[np.ndarray]] = [None] * args.num_uavs

        for k in range(max_len):
            step_dists: List[float] = []

            for i, d in enumerate(drones):
                p = paths_m[i]
                if k >= len(p) or len(p) == 0:
                    continue

                x, y, z = (float(p[k, 0]), float(p[k, 1]), float(p[k, 2]))
                print(f"[{args.uav_prefix}{i}] WP {k+1}/{len(p)} -> (x={x:.2f}, y={y:.2f}, z={z:.2f})")

                safe_go_to(
                    d,
                    x=x,
                    y=y,
                    z=z,
                    speed=float(args.speed),
                    frame_id=str(args.frame_id),
                    wait=False,
                )

                prev = last_sent[i]
                curr = np.array([x, y, z], dtype=float)
                step_dists.append(0.0 if prev is None else float(np.linalg.norm(curr - prev)))
                last_sent[i] = curr

            if args.fixed_dt:
                sleep_s = float(args.dt_s)
            else:
                seg_time = (max(step_dists) / max(float(args.speed), 1e-6)) if step_dists else 0.0
                sleep_s = max(float(args.dt_s), seg_time + float(args.dt_margin))
                sleep_s = min(sleep_s, float(args.dt_max))

            time.sleep(sleep_s)

        print(f"Hovering {args.hover_s:.2f} seconds, then landing...")
        time.sleep(float(args.hover_s))

        for d in drones:
            safe_land_disarm(d, wait=True)

    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl+C). Attempting safe landing...", file=sys.stderr)
        for d in drones:
            try:
                safe_land_disarm(d, wait=False)
            except Exception:
                pass
        time.sleep(2.0)

    finally:
        for d in drones:
            shutdown_drone(d)

        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
