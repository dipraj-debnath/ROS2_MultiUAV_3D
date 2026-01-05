#!/usr/bin/env python3
# deckga_as2_execute.py
#
# Executes DECK-GA + QuickNav paths in Aerostack2
# Now includes: waypoint printing for debugging

import os
import time
import pickle
from pathlib import Path

import rclpy
from as2_python_api.drone_interface import DroneInterface

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------

DATA_FILE = Path(__file__).resolve().parent / "data" / "deckga_quicknav_output.pkl"

DT = 2.5                  # Slower for safety
MIN_Z = 2.0
SCALE = 0.05              # Your recommended scale
FLIGHT_SPEED = 1.2        # Slower = safer

# -------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------

def load_paths(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    paths = data["quicknav_paths"]
    while len(paths) < 3:
        paths.append([])

    return paths[:3]


def sanitize_and_scale(path):
    out = []
    for p in path:
        x = float(p[0]) * SCALE
        y = float(p[1]) * SCALE
        z = float(p[2]) * SCALE

        if z < MIN_Z:
            z = MIN_Z

        out.append((x, y, z))
    return out


def takeoff_all(drones, height: float):
    for d in drones:
        d.offboard()
        d.arm()
        d.takeoff(height=height, wait=True)


def go_to_all(drones, paths):
    max_len = max(len(p) for p in paths)

    for k in range(max_len):

        for index, (d, p) in enumerate(zip(drones, paths)):

            if k < len(p):
                x, y, z = p[k]

                # -------- PRINT DEBUG WAYPOINT INFO -------- #
                print(f"[drone{index}] WP {k+1}/{len(p)} → (x={x:.2f}, y={y:.2f}, z={z:.2f})")

                # -------- SEND COMMAND -------- #
                d.go_to(
                    x=x,
                    y=y,
                    z=z,
                    speed=FLIGHT_SPEED,
                    frame_id="earth",
                    wait=False,
                )

        time.sleep(DT)


def land_all(drones):
    for d in drones:
        d.land(wait=True)
        d.disarm()

# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------

def main():
    print("🔄 Initializing rclpy...")
    rclpy.init()

    print("📦 Loading DECK-GA QuickNav paths...")
    raw_paths = load_paths(DATA_FILE)
    paths = [sanitize_and_scale(p) for p in raw_paths]

    print("🚁 Creating DroneInterface objects...")
    drones = []
    for ns in ["drone0", "drone1", "drone2"]:
        drones.append(DroneInterface(drone_id=ns, verbose=False, use_sim_time=True))

    print("⏳ Waiting 5 seconds for behavior servers...")
    time.sleep(5.0)

    print("🛫 Taking off all drones to 3.5 m...")
    takeoff_all(drones, height=3.5)

    print("✈️ Executing DECK-GA QuickNav paths (scaled)...")
    go_to_all(drones, paths)

    print("🧊 Hovering 3 seconds, then landing...")
    time.sleep(3.0)
    land_all(drones)

    print("🔌 Shutting down...")
    for d in drones:
        d.shutdown()

    rclpy.shutdown()
    print("✅ Mission complete.")


if __name__ == "__main__":
    main()
