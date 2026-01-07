#!/usr/bin/env python3
"""
deckga_execute.py

Execute DECK-GA paths in Aerostack2 (multi-UAV).

Coordinate mapping (MUST match rviz_paths_node.py):
- x_cmd = x_raw * SCALE_XY
- y_cmd = y_raw * SCALE_XY
- z_cmd = max(Z_MIN, z_raw * SCALE_Z)

Flight policy:
- Take off to TAKEOFF_Z (1.0 m by default)
- Immediately follow DECK-GA waypoints with variable altitude (z_raw * SCALE_Z)
- Never command z below Z_MIN

Also includes:
- Dynamic waypoint pacing based on max leg length / speed (+ buffer)
- Spawn/startpoint sanity check (XY only)
"""

import time
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import rclpy
from as2_python_api.drone_interface import DroneInterface

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------

DATA_FILE = Path(__file__).resolve().parent / "data" / "deckga_output.pkl"

FRAME_ID = "earth"
FLIGHT_SPEED = 1.2  # m/s

# Timing: dynamic sleep = max_leg/speed + TIME_BUFFER, but never below DT_MIN
DT_MIN = 2.5
TIME_BUFFER = 1.0

# Scaling (keep consistent with planning + RViz)
SCALE_XY = 0.05
SCALE_Z = 0.05

# Altitude policy (what you requested)
TAKEOFF_Z = 1.0
Z_MIN = 1.0  # safety clamp (must be >= 0, recommended >= 1.0 in sim)

# Spawn sanity check (AFTER scaling)
# Matches DECK_GA.py --start_points "10,0,0;50,20,0;90,0,0" with SCALE_XY=0.05
EXPECTED_SPAWN_XY = [
    (0.50, 0.00),  # drone0
    (2.50, 1.00),  # drone1
    (4.50, 0.00),  # drone2
]
SPAWN_WARN_DIST = 1.0  # meters


# -------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------

def load_paths(pkl_path: Path, num_uavs: int = 3) -> List[np.ndarray]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    paths = data.get("deckga_paths", [])
    while len(paths) < num_uavs:
        paths.append([])

    return [np.asarray(p, dtype=float) for p in paths[:num_uavs]]


def remove_duplicate_closure(path: np.ndarray) -> np.ndarray:
    """If first == last (closed tour), remove the last duplicated point."""
    if path is None or len(path) < 2:
        return path
    return path[:-1] if np.allclose(path[0], path[-1]) else path


def sanitize_and_scale(path_xyz: np.ndarray) -> List[Tuple[float, float, float]]:
    """
    XY: scaled by SCALE_XY
    Z : use DECK-GA z variation (z_raw*SCALE_Z) but clamp at Z_MIN
    """
    if path_xyz is None or len(path_xyz) == 0:
        return []

    path_xyz = remove_duplicate_closure(path_xyz)

    out: List[Tuple[float, float, float]] = []
    for p in path_xyz:
        x = float(p[0]) * SCALE_XY
        y = float(p[1]) * SCALE_XY

        z = float(p[2]) * SCALE_Z
        if z < Z_MIN:
            z = Z_MIN

        out.append((x, y, z))

    return out


def check_startpoints_vs_spawn(paths_scaled: List[List[Tuple[float, float, float]]]) -> None:
    """Warn if first waypoint is far from expected spawn XY."""
    for i, p in enumerate(paths_scaled):
        if not p:
            print(f"[WARN] drone{i}: empty path")
            continue

        x0, y0, _ = p[0]
        if i < len(EXPECTED_SPAWN_XY):
            ex, ey = EXPECTED_SPAWN_XY[i]
            d = float(((x0 - ex) ** 2 + (y0 - ey) ** 2) ** 0.5)
            if d > SPAWN_WARN_DIST:
                print(
                    f"[WARN] drone{i}: first WP ({x0:.2f},{y0:.2f}) is {d:.2f} m from expected spawn ({ex:.2f},{ey:.2f}).\n"
                    f"       Fix by aligning DECK_GA --start_points with Aerostack spawn layout\n"
                    f"       OR update EXPECTED_SPAWN_XY here.\n"
                )


def takeoff_all(drones: List[DroneInterface], height: float) -> None:
    for d in drones:
        d.offboard()
        d.arm()
        d.takeoff(height=height, wait=True)


def compute_step_sleep_seconds(
    prev_points: List[Optional[Tuple[float, float, float]]],
    next_points: List[Optional[Tuple[float, float, float]]],
) -> float:
    """Dynamic dt = (max distance any drone must travel)/speed + buffer, min DT_MIN."""
    max_dist = 0.0
    for a, b in zip(prev_points, next_points):
        if a is None or b is None:
            continue
        ax, ay, az = a
        bx, by, bz = b
        dist = float(((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2) ** 0.5)
        max_dist = max(max_dist, dist)

    dynamic_dt = (max_dist / max(FLIGHT_SPEED, 1e-6)) + TIME_BUFFER
    return max(DT_MIN, dynamic_dt)


def go_to_all(drones: List[DroneInterface], paths: List[List[Tuple[float, float, float]]]) -> None:
    """Issue waypoint k to all drones (wait=False), then sleep sufficiently."""
    max_len = max((len(p) for p in paths), default=0)
    if max_len == 0:
        print("[WARN] No waypoints to execute (all paths empty).")
        return

    last_cmd: List[Optional[Tuple[float, float, float]]] = [None] * len(drones)

    for k in range(max_len):
        next_cmd: List[Optional[Tuple[float, float, float]]] = [None] * len(drones)

        for i, (d, p) in enumerate(zip(drones, paths)):
            if k < len(p):
                x, y, z = p[k]
                next_cmd[i] = (x, y, z)

                print(f"[drone{i}] WP {k+1}/{len(p)} → (x={x:.2f}, y={y:.2f}, z={z:.2f})")
                d.go_to(
                    x=x,
                    y=y,
                    z=z,
                    speed=FLIGHT_SPEED,
                    frame_id=FRAME_ID,
                    wait=False,
                )

        time.sleep(compute_step_sleep_seconds(last_cmd, next_cmd))
        last_cmd = next_cmd


def land_all(drones: List[DroneInterface]) -> None:
    for d in drones:
        d.land(wait=True)
        d.disarm()


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------

def main():
    print("Initializing rclpy...")
    rclpy.init()

    print("Loading DECK-GA paths...")
    raw_paths = load_paths(DATA_FILE, num_uavs=3)
    paths = [sanitize_and_scale(p) for p in raw_paths]

    # Sanity check: startpoints vs spawn layout
    check_startpoints_vs_spawn(paths)

    print("Creating DroneInterface objects...")
    drones = [DroneInterface(drone_id=ns, verbose=False, use_sim_time=True) for ns in ["drone0", "drone1", "drone2"]]

    print("Waiting 5 seconds for behavior servers...")
    time.sleep(5.0)

    try:
        print(f"Taking off all drones to {TAKEOFF_Z:.1f} m...")
        takeoff_all(drones, height=TAKEOFF_Z)

        print("Executing DECK-GA paths (scaled, variable-z)...")
        go_to_all(drones, paths)

        print("Hovering 3 seconds, then landing...")
        time.sleep(3.0)
        land_all(drones)

    finally:
        print("Shutting down...")
        for d in drones:
            d.shutdown()
        rclpy.shutdown()

    print("Mission complete.")


if __name__ == "__main__":
    main()
