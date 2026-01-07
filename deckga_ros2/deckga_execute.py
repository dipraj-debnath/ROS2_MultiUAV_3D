#!/usr/bin/env python3
"""
deckga_execute.py

Executes DECK-GA paths in Aerostack2 (multi-UAV).

Improvements in this version:
1) Uses GA z, but never descends below takeoff altitude:
   - TAKEOFF_Z = 3.5
   - MIN_Z     = 3.5  (must be >= TAKEOFF_Z)
2) Dynamic waypoint timing:
   - Sleeps long enough for the longest leg among drones to be feasible at FLIGHT_SPEED.
3) Start point / spawn layout sanity check:
   - Warns if the first waypoint for each drone is far from expected spawn XY.
"""

import time
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rclpy
from as2_python_api.drone_interface import DroneInterface

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------

# Where DECK_GA writes output (you already overwrite this each run)
DATA_FILE = Path(__file__).resolve().parent / "data" / "deckga_output.pkl"

# Motion / pacing
FLIGHT_SPEED = 1.2        # m/s
DT_MIN = 2.5              # minimum seconds between waypoint "steps"
TIME_BUFFER = 1.0         # extra seconds added to dynamic timing (stability)

# Scaling (must match your RViz marker scaling and mission expectations)
SCALE_XY = 0.05
SCALE_Z = 0.05

# Altitude policy (Option B): use GA z, but never descend below takeoff
TAKEOFF_Z = 3.5
MIN_Z = 3.5               # IMPORTANT: must be >= TAKEOFF_Z

# Frame
FRAME_ID = "earth"

# Expected spawn XY (AFTER scaling) for drone0/drone1/drone2.
# These must match how your Aerostack example spawns drones.
# With DECK_GA start_points "10,0,0;50,20,0;90,0,0" and SCALE_XY=0.05 -> (0.5,0.0), (2.5,1.0), (4.5,0.0)
EXPECTED_SPAWN_XY = [
    (0.50, 0.00),  # drone0
    (2.50, 1.00),  # drone1
    (4.50, 0.00),  # drone2
]
SPAWN_WARN_DIST = 1.0     # meters; warn if first waypoint is farther than this


# -------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------

def load_paths(pkl_path: Path, num_uavs: int = 3) -> List[np.ndarray]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    paths = data.get("deckga_paths", [])
    # Ensure exactly num_uavs entries (pad missing with empty)
    while len(paths) < num_uavs:
        paths.append([])

    out = []
    for p in paths[:num_uavs]:
        out.append(np.asarray(p, dtype=float))
    return out


def remove_duplicate_closure(path: np.ndarray) -> np.ndarray:
    """
    If path is a closed tour (first==last), remove the last point.
    Aerostack does not need the duplicated final waypoint.
    """
    if path is None or len(path) < 2:
        return path
    if np.allclose(path[0], path[-1]):
        return path[:-1]
    return path


def sanitize_and_scale(path_xyz: np.ndarray) -> List[Tuple[float, float, float]]:
    """
    Scales XY and Z, clamps z to MIN_Z, and returns list of (x,y,z).
    """
    if path_xyz is None or len(path_xyz) == 0:
        return []

    path_xyz = remove_duplicate_closure(path_xyz)

    out: List[Tuple[float, float, float]] = []
    for p in path_xyz:
        x = float(p[0]) * SCALE_XY
        y = float(p[1]) * SCALE_XY
        z = float(p[2]) * SCALE_Z

        # Critical: never go below MIN_Z (must be >= TAKEOFF_Z)
        if z < MIN_Z:
            z = MIN_Z

        out.append((x, y, z))

    return out


def check_startpoints_vs_spawn(paths_scaled: List[List[Tuple[float, float, float]]]) -> None:
    """
    Warn if the first commanded waypoint for each drone is far from expected spawn XY.
    This is the most common reason a drone appears to "go weird" immediately.
    """
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
                    f"       Fix by aligning DECK_GA --start_points with Aerostack spawn layout OR edit EXPECTED_SPAWN_XY.\n"
                )
        else:
            print(f"[WARN] drone{i}: no EXPECTED_SPAWN_XY entry to validate against.")


def takeoff_all(drones: List[DroneInterface], height: float) -> None:
    for d in drones:
        d.offboard()
        d.arm()
        d.takeoff(height=height, wait=True)


def compute_step_sleep_seconds(prev_points, next_points) -> float:
    """
    Dynamic time: (max distance any drone must travel) / speed + buffer,
    but never less than DT_MIN.
    """
    max_dist = 0.0
    for a, b in zip(prev_points, next_points):
        if a is None or b is None:
            continue
        ax, ay, az = a
        bx, by, bz = b
        dist = float(((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2) ** 0.5)
        if dist > max_dist:
            max_dist = dist

    dynamic_dt = (max_dist / max(FLIGHT_SPEED, 1e-6)) + TIME_BUFFER
    return max(DT_MIN, dynamic_dt)


def go_to_all(drones: List[DroneInterface], paths: List[List[Tuple[float, float, float]]]) -> None:
    """
    Sends the k-th waypoint to all drones (non-blocking), then sleeps long enough.
    """
    max_len = max((len(p) for p in paths), default=0)
    if max_len == 0:
        print("[WARN] No waypoints to execute (all paths empty).")
        return

    # Track last commanded positions for timing
    last_cmd = [None] * len(drones)

    for k in range(max_len):
        next_cmd = [None] * len(drones)

        # Issue commands for this step
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

        # Sleep long enough for the slowest leg to be feasible
        sleep_s = compute_step_sleep_seconds(last_cmd, next_cmd)
        time.sleep(sleep_s)

        # Update last commanded
        last_cmd = next_cmd


def land_all(drones: List[DroneInterface]) -> None:
    for d in drones:
        d.land(wait=True)
        d.disarm()


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------

def main():
    if MIN_Z < TAKEOFF_Z:
        raise ValueError(f"MIN_Z ({MIN_Z}) must be >= TAKEOFF_Z ({TAKEOFF_Z})")

    print("🔄 Initializing rclpy...")
    rclpy.init()

    print("📦 Loading DECK-GA paths...")
    raw_paths = load_paths(DATA_FILE, num_uavs=3)
    paths = [sanitize_and_scale(p) for p in raw_paths]

    # Sanity check: startpoints vs spawn layout
    check_startpoints_vs_spawn(paths)

    print("🚁 Creating DroneInterface objects...")
    drones = []
    for ns in ["drone0", "drone1", "drone2"]:
        drones.append(DroneInterface(drone_id=ns, verbose=False, use_sim_time=True))

    print("⏳ Waiting 5 seconds for behavior servers...")
    time.sleep(5.0)

    try:
        print(f"🛫 Taking off all drones to {TAKEOFF_Z:.1f} m...")
        takeoff_all(drones, height=TAKEOFF_Z)

        print("✈️ Executing DECK-GA paths (scaled, z-clamped)...")
        go_to_all(drones, paths)

        print("🧊 Hovering 3 seconds, then landing...")
        time.sleep(3.0)
        land_all(drones)

    finally:
        print("🔌 Shutting down...")
        for d in drones:
            d.shutdown()
        rclpy.shutdown()

    print("✅ Mission complete.")


if __name__ == "__main__":
    main()
