#!/usr/bin/env python3
"""
deckga_execute_antarctica.py

DECK-GA execution for Antarctica world (Gazebo Harmonic + Aerostack2).

This script:
- Loads DECK-GA output PKL (deckga_paths + offset_used)
- Applies the Antarctica transform pipeline (auto/original/shifted handling + XY scaling/centering/clamp + Z mapping + z_base)
- Arms + offboards + takes off
- Executes multi-UAV waypoint stepping in parallel (go_to(wait=False)) with pacing sleeps
- Lands

Timing instrumentation (for paper/experiments):

A) Planned time (algorithmic / kinematic model):
    * Uses transformed waypoint path length in meters and constant speed model.
    * planned_time_uav_i = path_length_i / speed
    * planned_mission_time = max_i(planned_time_uav_i)  (parallel UAVs => makespan)

B) Executed time (system-level / wall-clock):
    * Measures real wall-clock time spent in: offboard+arm, takeoff, first-WP handshake,
      path execution loop, hover, landing.

C) Mission completion time (recommended metric for experiments):
    * "Mission complete" is defined as:
        time from first waypoint command to final waypoint command.
    * We report:
        - Per-UAV mission completion time (command-level): t_last_cmd - t_first_cmd
        - Mission makespan (parallel UAVs): max over UAVs

CSV logging (for plots):
- Writes a per-run summary CSV
- Writes per-UAV waypoint/segment CSVs with per-segment distance + command-level dt
  (WP1->WP2, WP2->WP3, ...)

Notes:
- Because go_to(wait=False), segment dt is command-level timing, not exact arrival timing.
  For exact arrival timing, we would need feedback/state subscriptions.
"""

import argparse
import csv
import inspect
import os
import pickle
import subprocess
import time
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

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


def safe_offboard_arm(drone: DroneInterface) -> None:
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


def safe_takeoff(drone: DroneInterface, height: float, wait: bool) -> None:
    try:
        drone.takeoff(height=float(height), wait=bool(wait))
    except TypeError:
        try:
            drone.takeoff(height=float(height))
        except Exception:
            pass
    except Exception:
        pass


def safe_land(drone: DroneInterface, wait: bool) -> None:
    try:
        drone.land(wait=bool(wait))
    except TypeError:
        try:
            drone.land()
        except Exception:
            pass
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
) -> bool:
    try:
        drone.go_to(x=x, y=y, z=z, speed=float(speed), frame_id=str(frame_id), wait=bool(wait))
        return True
    except TypeError:
        pass
    except Exception:
        return False

    try:
        drone.go_to(x, y, z, float(speed), str(frame_id), bool(wait))
        return True
    except Exception:
        return False


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

        if mode == "shifted":
            q[:, 0] -= offset_used[0]
            q[:, 1] -= offset_used[1]
            q[:, 2] -= offset_used[2]

        q[:, 0] = q[:, 0] * float(xy_scale) + float(xy_center_x)
        q[:, 1] = q[:, 1] * float(xy_scale) + float(xy_center_y)

        if bool(clamp_xy):
            q[:, 0] = np.clip(q[:, 0], float(x_min), float(x_max))
            q[:, 1] = np.clip(q[:, 1], float(y_min), float(y_max))

        rel_z = map_range(q[:, 2], float(zin_min), float(zin_max), float(zout_min), float(zout_max))
        q[:, 2] = rel_z + float(z_base)

        out.append(q)

    return out


def print_ranges(paths_m: List[np.ndarray]) -> Tuple[float, float, float, float, float, float]:
    if not paths_m:
        return (0, 0, 0, 0, 0, 0)
    xs = np.concatenate([p[:, 0] for p in paths_m if len(p) > 0])
    ys = np.concatenate([p[:, 1] for p in paths_m if len(p) > 0])
    zs = np.concatenate([p[:, 2] for p in paths_m if len(p) > 0])
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    zmin, zmax = float(zs.min()), float(zs.max())
    print(f"Final waypoint ranges (meters): x[{xmin:.2f},{xmax:.2f}] y[{ymin:.2f},{ymax:.2f}] z[{zmin:.2f},{zmax:.2f}]")
    return xmin, xmax, ymin, ymax, zmin, zmax


# ----------------------------
# Timing helpers
# ----------------------------
def path_length_m(path: np.ndarray) -> float:
    if path.size == 0 or len(path) < 2:
        return 0.0
    segs = path[1:, :] - path[:-1, :]
    return float(np.sum(np.linalg.norm(segs, axis=1)))


def planned_times(paths: List[np.ndarray], speed: float) -> Tuple[List[float], List[float], float]:
    v = max(float(speed), 1e-6)
    lengths = [path_length_m(p) for p in paths]
    times_s = [L / v for L in lengths]
    makespan = max(times_s) if times_s else 0.0
    return lengths, times_s, makespan


def _fmt_s(seconds: float) -> str:
    s = float(seconds)
    if s < 0:
        s = 0.0
    if s < 120.0:
        return f"{s:.2f} s"
    return f"{s/60.0:.2f} min"


# ----------------------------
# CSV logging helpers
# ----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, header: List[str], rows: List[List[Any]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ----------------------------
# Action availability wait
# ----------------------------
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


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--deckga_pkl", default="deckga_ros2/data/deckga_output_antarctica.pkl")
    ap.add_argument("--namespaces", default="drone0,drone1,drone2")
    ap.add_argument("--frame_id", default=DEFAULT_FRAME_ID)

    # Speed
    ap.add_argument("--speed", type=float, default=1.5)

    # Waits
    ap.add_argument("--wait_actions_s", type=float, default=60.0)
    ap.add_argument("--init_wait_s", type=float, default=5.0)
    ap.add_argument("--pre_arm_wait_s", type=float, default=0.0)
    ap.add_argument("--takeoff_settle_s", type=float, default=6.0)

    # Takeoff policy (keep your working behavior)
    ap.add_argument("--takeoff_margin", type=float, default=0.20, help="takeoff_abs_z = min_waypoint_z + margin")
    ap.add_argument("--takeoff_sequential", action="store_true")

    # First waypoint handshake
    ap.add_argument("--first_wp_wait", action="store_true")

    # Coord mode
    ap.add_argument("--coord_mode", default="auto", choices=["auto", "original", "shifted"])

    # XY mapping
    ap.add_argument("--xy_scale", type=float, default=1.0)
    ap.add_argument("--xy_center_x", type=float, default=0.0)
    ap.add_argument("--xy_center_y", type=float, default=0.0)

    # XY clamp
    ap.add_argument("--clamp_xy", action="store_true")
    ap.add_argument("--x_min", type=float, default=-30.0)
    ap.add_argument("--x_max", type=float, default=30.0)
    ap.add_argument("--y_min", type=float, default=-30.0)
    ap.add_argument("--y_max", type=float, default=30.0)

    # Z mapping
    ap.add_argument("--zin_min", type=float, default=33.34)
    ap.add_argument("--zin_max", type=float, default=39.30)
    ap.add_argument("--zout_min", type=float, default=1.14)
    ap.add_argument("--zout_max", type=float, default=7.10)
    ap.add_argument("--z_base", type=float, default=32.2)

    # CSV logger
    ap.add_argument("--log_dir", default="results_antarctica", help="directory to write CSV logs")
    ap.add_argument("--run_tag", default="antarctica", help="string tag included in output filenames")

    ap.add_argument("--use_sim_time", action="store_true")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    namespaces = [s.strip() for s in args.namespaces.split(",") if s.strip()]
    if not namespaces:
        raise ValueError("No namespaces provided")

    run_stamp = now_tag()
    ensure_dir(args.log_dir)

    print("Initializing rclpy...")
    rclpy.init()

    drones: List[DroneInterface] = []
    try:
        # 1) Wait for actions
        wait_for_actions(namespaces, args.wait_actions_s)

        # 2) Load + transform
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

        # 3) Print ranges + planned timing
        _, _, _, _, zmin, _ = print_ranges(paths_m)
        planned_lengths, planned_times_s, planned_makespan = planned_times(paths_m, args.speed)

        print("\n=== Planned timing (kinematic model) ===")
        print(f"Assumed constant speed: {float(args.speed):.3f} m/s")
        for i in range(len(paths_m)):
            print(f"UAV {i}: length={planned_lengths[i]:.3f} m, planned_time={planned_times_s[i]:.2f} s")
        print(f"Planned makespan (parallel UAVs): {planned_makespan:.2f} s")
        print("======================================\n")

        # 4) Takeoff target
        takeoff_abs_z = float(zmin) + float(args.takeoff_margin)
        print(f"Auto takeoff ABS z = min_waypoint_z + margin = {takeoff_abs_z:.2f}")

        # 5) Create interfaces
        print("Creating DroneInterface objects...")
        for ns in namespaces:
            drones.append(make_drone_interface(ns, use_sim_time=args.use_sim_time, verbose=args.verbose))

        print(f"Initial settle wait: {args.init_wait_s:.1f}s")
        time.sleep(args.init_wait_s)

        if args.pre_arm_wait_s > 0.0:
            print(f"Pre-arm warmup wait: {args.pre_arm_wait_s:.1f}s")
            time.sleep(args.pre_arm_wait_s)

        # --------------------------
        # Executed timing (wall-clock)
        # --------------------------
        t_exec_start = time.perf_counter()

        # Offboard + arm
        t_arm_start = time.perf_counter()
        print("Switching to offboard + arming...")
        for d in drones:
            safe_offboard_arm(d)
        t_arm_end = time.perf_counter()

        # Takeoff
        t_takeoff_start = time.perf_counter()
        if args.takeoff_sequential:
            print(f"Taking off SEQUENTIALLY to ABS z={takeoff_abs_z:.2f} ({args.frame_id} frame)...")
            for d in drones:
                safe_takeoff(d, height=takeoff_abs_z, wait=True)
        else:
            print(f"Taking off ALL drones in parallel to ABS z={takeoff_abs_z:.2f} ({args.frame_id} frame)...")
            for d in drones:
                safe_takeoff(d, height=takeoff_abs_z, wait=False)
        t_takeoff_end = time.perf_counter()

        time.sleep(args.takeoff_settle_s)

        # Re-assert offboard
        for d in drones:
            try:
                d.offboard()
            except Exception:
                pass

        # First waypoint handshake
        t_handshake_start = time.perf_counter()
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
        t_handshake_end = time.perf_counter()

        # Main execution: parallel stepping
        print("Executing DECK-GA paths (parallel waypoint stepping)...")
        max_len = max((len(p) for p in paths_m), default=0)

        last_sent: List[Optional[np.ndarray]] = [None] * len(drones)

        # Command-level mission completion (first cmd -> last cmd)
        first_wp_cmd_t: List[Optional[float]] = [None] * len(drones)
        last_wp_cmd_t: List[Optional[float]] = [None] * len(drones)

        # Per-UAV segment logs
        # Each UAV: list of dict rows
        seg_logs: List[List[Dict[str, Any]]] = [[] for _ in range(len(drones))]
        cum_dist: List[float] = [0.0 for _ in range(len(drones))]
        last_cmd_time: List[Optional[float]] = [None for _ in range(len(drones))]

        t_paths_start = time.perf_counter()

        for k in range(max_len):
            step_dists: List[float] = []

            for i, d in enumerate(drones):
                p = paths_m[i]
                if k >= len(p) or len(p) == 0:
                    continue

                x, y, z = map(float, p[k])
                print(f"[{namespaces[i]}] WP {k+1}/{len(p)} -> (x={x:.2f}, y={y:.2f}, z={z:.2f})")
                safe_go_to(d, x, y, z, args.speed, args.frame_id, wait=False)

                t_cmd_abs = time.perf_counter()
                t_cmd_rel = t_cmd_abs - t_paths_start

                # mission completion timestamps (command-level)
                if first_wp_cmd_t[i] is None:
                    first_wp_cmd_t[i] = t_cmd_abs
                last_wp_cmd_t[i] = t_cmd_abs

                # segment distance + dt (command-level)
                if last_sent[i] is None:
                    seg_dist = 0.0
                else:
                    seg_dist = float(np.linalg.norm(np.array([x, y, z], dtype=float) - last_sent[i]))

                if last_cmd_time[i] is None:
                    seg_dt = 0.0
                else:
                    seg_dt = float(t_cmd_abs - last_cmd_time[i])

                cum_dist[i] += seg_dist
                seg_speed_cmd = (seg_dist / seg_dt) if seg_dt > 1e-9 else 0.0

                seg_logs[i].append({
                    "wp_idx": int(k + 1),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "cmd_time_s": float(t_cmd_rel),
                    "seg_dist_m": float(seg_dist),
                    "seg_dt_s": float(seg_dt),
                    "seg_speed_cmd_mps": float(seg_speed_cmd),
                    "cum_dist_m": float(cum_dist[i]),
                })

                last_cmd_time[i] = t_cmd_abs
                last_sent[i] = np.array([x, y, z], dtype=float)
                step_dists.append(seg_dist)

            # pacing sleep proxy for longest step
            if step_dists:
                max_step = max(step_dists)
                sleep_t = max(0.30, float(max_step / max(args.speed, 1e-6)))
                time.sleep(sleep_t)

        t_paths_end = time.perf_counter()

        # Hover + land
        print("Hover 2s...")
        t_hover_start = time.perf_counter()
        time.sleep(2.0)
        t_hover_end = time.perf_counter()

        print("Landing all drones (non-blocking)...")
        t_land_start = time.perf_counter()
        for d in drones:
            safe_land(d, wait=False)
        time.sleep(3.0)
        t_land_end = time.perf_counter()

        t_exec_end = time.perf_counter()

        print("Done.")

        # --------------------------
        # Reporting (console)
        # --------------------------
        print("\n=== Executed timing (wall-clock) ===")
        print(f"Offboard+arm phase : {_fmt_s(t_arm_end - t_arm_start)}")
        print(f"Takeoff phase      : {_fmt_s(t_takeoff_end - t_takeoff_start)}")
        print(f"1st WP handshake   : {_fmt_s(t_handshake_end - t_handshake_start)}")
        print(f"Path execution     : {_fmt_s(t_paths_end - t_paths_start)}")
        print(f"Hover phase        : {_fmt_s(t_hover_end - t_hover_start)}")
        print(f"Landing phase      : {_fmt_s(t_land_end - t_land_start)}")
        print(f"TOTAL (arm->land)  : {_fmt_s(t_exec_end - t_exec_start)}")

        print("\n=== Mission completion time (command-level, no landing) ===")
        mission_times_s: List[float] = []
        for i in range(len(drones)):
            first = first_wp_cmd_t[i]
            last = last_wp_cmd_t[i]
            if first is None or last is None:
                t_m = 0.0
            else:
                t_m = float(last - first)
            mission_times_s.append(t_m)
            print(f"UAV {i}: {_fmt_s(t_m)}")
        mission_makespan = max(mission_times_s) if mission_times_s else 0.0
        print(f"Mission makespan (parallel UAVs): {_fmt_s(mission_makespan)}")
        print("=========================================================\n")

        # --------------------------
        # CSV logging
        # --------------------------
        # Summary CSV (one row per run)
        summary_path = os.path.join(args.log_dir, f"run_{args.run_tag}_{run_stamp}_summary.csv")
        summary_header = [
            "run_tag", "timestamp",
            "deckga_pkl",
            "num_uavs", "speed_mps",
            "takeoff_abs_z",
            "planned_makespan_s",
            "executed_total_arm_to_land_s",
            "executed_path_phase_s",
            "mission_makespan_cmd_s",
            "offboard_arm_s", "takeoff_s", "handshake_s", "hover_s", "landing_s",
        ]
        summary_row = [[
            str(args.run_tag),
            str(run_stamp),
            str(args.deckga_pkl),
            int(len(drones)),
            float(args.speed),
            float(takeoff_abs_z),
            float(planned_makespan),
            float(t_exec_end - t_exec_start),
            float(t_paths_end - t_paths_start),
            float(mission_makespan),
            float(t_arm_end - t_arm_start),
            float(t_takeoff_end - t_takeoff_start),
            float(t_handshake_end - t_handshake_start),
            float(t_hover_end - t_hover_start),
            float(t_land_end - t_land_start),
        ]]
        write_csv(summary_path, summary_header, summary_row)

        # Per-UAV segment CSVs
        for i in range(len(drones)):
            uav_path = os.path.join(args.log_dir, f"run_{args.run_tag}_{run_stamp}_uav{i}_segments.csv")
            header = ["wp_idx", "x", "y", "z", "cmd_time_s", "seg_dist_m", "seg_dt_s", "seg_speed_cmd_mps", "cum_dist_m"]
            rows: List[List[Any]] = []
            for r in seg_logs[i]:
                rows.append([
                    r["wp_idx"], r["x"], r["y"], r["z"],
                    r["cmd_time_s"], r["seg_dist_m"], r["seg_dt_s"], r["seg_speed_cmd_mps"], r["cum_dist_m"]
                ])
            write_csv(uav_path, header, rows)

        print(f"[LOG] Wrote summary CSV: {summary_path}")
        for i in range(len(drones)):
            print(f"[LOG] Wrote UAV{i} segments CSV: {os.path.join(args.log_dir, f'run_{args.run_tag}_{run_stamp}_uav{i}_segments.csv')}")

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
