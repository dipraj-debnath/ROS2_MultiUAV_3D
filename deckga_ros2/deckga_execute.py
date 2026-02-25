#!/usr/bin/env python3
"""
deckga_execute.py (default world)

Execution for Aerostack2 multi-UAV with:
- Default (wait=False) mode OR --ensure_reach (wait=True per waypoint)
- First-waypoint handshake option (like Antarctica)
- Wait-for-actions option (like Antarctica)
- CSV logging (summary + per-UAV segments)
- Distance reporting (waypoint-to-waypoint polyline distance in EXECUTED coords)

Landing stability:
- Landing is done NON-BLOCKING (wait=False) for all drones, then a settle sleep, then best-effort disarm.
  This prevents the common issue: one drone's LandBehavior hangs and the whole script blocks forever.

Transform:
- This version assumes you generated points already scaled if you want scaled execution.
- Defaults are IDENTITY transform (scale_xy=1, scale_z=1, z_offset=0, z_min=0).
"""

from __future__ import annotations

import argparse
import csv
import pickle
import subprocess
import time
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np

import rclpy
from as2_python_api.drone_interface import DroneInterface


# =========================
# Defaults
# =========================
DEFAULT_DECKGA_PKL = "/home/dipraj/Documents/GitHub/ROS2_MultiUAV_3D/deckga_ros2/data/deckga_output.pkl"

# DEFAULT = identity transform (recommended if you scale at generation time)
DEFAULT_COORD_MODE = "original"  # "original" | "shifted" | "auto"
DEFAULT_SCALE_XY = 1.0
DEFAULT_SCALE_Z = 1.0
DEFAULT_Z_OFFSET = 0.0
DEFAULT_Z_MIN = 0.0

DEFAULT_FRAME_ID = "earth"

DEFAULT_WAIT_ACTIONS_S = 60.0
DEFAULT_INIT_WAIT_S = 5.0
DEFAULT_PRE_ARM_WAIT_S = 0.0
DEFAULT_TAKEOFF_SETTLE_S = 6.0

DEFAULT_TAKEOFF_Z = 0.5
DEFAULT_SPEED = 1.0
DEFAULT_HOVER_S = 2.0
DEFAULT_LAND_SETTLE_S = 8.0  # time to let land complete after sending land(wait=False)

DEFAULT_LOG_DIR = "/home/dipraj/Documents/GitHub/ROS2_MultiUAV_3D/results_csv"
DEFAULT_RUN_TAG = "default_world"


# =========================
# Transform (shared)
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
    if offset_used is None or all_pts.shape[0] == 0:
        return False
    min_xy = all_pts[:, :2].min(axis=0)
    mean_xy = all_pts[:, :2].mean(axis=0)
    if min_xy[0] < -1e-6 or min_xy[1] < -1e-6:
        return False
    return (mean_xy[0] >= 0.20 * offset_used[0]) or (mean_xy[1] >= 0.20 * offset_used[1])


@dataclass(frozen=True)
class TransformConfig:
    coord_mode: str
    scale_xy: float
    scale_z: float
    z_offset: float
    z_min: float


def transform_paths(raw_paths: Sequence[Any], offset_used: Optional[np.ndarray], tf: TransformConfig) -> List[np.ndarray]:
    paths_np: List[np.ndarray] = []
    for p in raw_paths:
        arr = np.asarray(p, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Each path must be Nx3. Got shape {arr.shape}")
        paths_np.append(arr)

    all_pts = _stack_all_points(paths_np)

    mode = tf.coord_mode.lower().strip()
    if mode not in ("original", "shifted", "auto"):
        raise ValueError("coord_mode must be one of: original, shifted, auto")

    use_shift = False
    if mode == "shifted":
        use_shift = True
    elif mode == "auto":
        use_shift = _auto_is_shifted(all_pts, offset_used)
    else:
        use_shift = False

    out: List[np.ndarray] = []
    for p in paths_np:
        q = p.copy()

        # Unshift if needed
        if use_shift and offset_used is not None:
            q[:, 0] -= float(offset_used[0])
            q[:, 1] -= float(offset_used[1])
            q[:, 2] -= float(offset_used[2])

        # Scale
        q[:, 0] *= float(tf.scale_xy)
        q[:, 1] *= float(tf.scale_xy)
        q[:, 2] *= float(tf.scale_z)

        # Z offset + clamp
        q[:, 2] += float(tf.z_offset)
        q[:, 2] = np.maximum(q[:, 2], float(tf.z_min))

        out.append(q)

    return out


# =========================
# IO helpers
# =========================
def load_deckga_pkl(pkl_path: str) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    p = Path(pkl_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"DECK-GA output PKL not found: {p}")

    with p.open("rb") as f:
        data = pickle.load(f)

    if "deckga_paths" not in data:
        raise KeyError(f"'deckga_paths' missing. PKL keys: {list(data.keys())}")

    paths = [np.asarray(x, dtype=float) for x in data["deckga_paths"]]
    for i, arr in enumerate(paths):
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Path {i} has bad shape {arr.shape}; expected (N,3)")

    offset_used = None
    if "offset_used" in data:
        off = np.asarray(data["offset_used"], dtype=float).reshape(-1)
        if off.size >= 3:
            offset_used = off[:3].copy()

    return paths, offset_used


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_csv(path: str, header: List[str], rows: List[List[Any]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# =========================
# Distance + timing helpers
# =========================
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
    return f"{s / 60.0:.2f} min"


# =========================
# Action availability wait (like Antarctica)
# =========================
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


# =========================
# DroneInterface wrappers (compat)
# =========================
def make_drone_interface(ns: str, use_sim_time: bool, verbose: bool) -> DroneInterface:
    try:
        return DroneInterface(namespace=ns, use_sim_time=bool(use_sim_time), verbose=bool(verbose))
    except TypeError:
        return DroneInterface(ns)


def safe_offboard_arm(d: DroneInterface) -> None:
    di = cast(Any, d)
    try:
        di.offboard()
    except Exception:
        try:
            di.set_offboard_mode()
        except Exception:
            pass
    try:
        di.arm()
    except Exception:
        pass


def safe_takeoff(d: DroneInterface, height: float, wait: bool) -> None:
    di = cast(Any, d)
    try:
        di.takeoff(height=float(height), wait=bool(wait))
        return
    except TypeError:
        pass
    except Exception:
        return
    try:
        di.takeoff(height=float(height))
    except Exception:
        return


def safe_go_to(d: DroneInterface, x: float, y: float, z: float, speed: float, frame_id: str, wait: bool) -> bool:
    di = cast(Any, d)
    try:
        di.go_to(x=x, y=y, z=z, speed=float(speed), frame_id=str(frame_id), wait=bool(wait))
        return True
    except TypeError:
        pass
    except Exception:
        return False
    try:
        di.go_to(x, y, z, float(speed), str(frame_id), bool(wait))
        return True
    except Exception:
        return False


def safe_land(d: DroneInterface, wait: bool) -> None:
    di = cast(Any, d)
    try:
        di.land(wait=bool(wait))
        return
    except TypeError:
        pass
    except Exception:
        return
    try:
        di.land()
    except Exception:
        return


def safe_disarm(d: DroneInterface) -> None:
    di = cast(Any, d)
    try:
        di.disarm()
    except Exception:
        pass


# =========================
# ensure-reach worker
# =========================
def _uav_worker_wait_each_wp(
    ns: str,
    drone: DroneInterface,
    path: np.ndarray,
    speed: float,
    frame_id: str,
    wp_settle_s: float,
    t0: float,
    seg_logs_out: List[Dict[str, Any]],
) -> None:
    last_xyz = None
    last_t = None
    cum = 0.0

    for k in range(len(path)):
        x, y, z = map(float, path[k])
        print(f"[{ns}] WP {k+1}/{len(path)} -> (x={x:.2f}, y={y:.2f}, z={z:.2f}) [wait=True]")

        t_cmd = time.perf_counter()
        ok = safe_go_to(drone, x, y, z, speed, frame_id, wait=True)
        t_done = time.perf_counter()

        seg_dist = 0.0 if last_xyz is None else float(np.linalg.norm(np.array([x, y, z]) - last_xyz))
        seg_dt = 0.0 if last_t is None else float(t_done - last_t)

        cum += seg_dist
        seg_speed = (seg_dist / seg_dt) if seg_dt > 1e-9 else 0.0

        seg_logs_out.append(
            {
                "wp_idx": int(k + 1),
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "cmd_time_s": float(t_cmd - t0),
                "done_time_s": float(t_done - t0),
                "seg_dist_m": float(seg_dist),
                "seg_dt_s": float(seg_dt),
                "seg_speed_mps": float(seg_speed),
                "cum_dist_m": float(cum),
                "ok": bool(ok),
            }
        )

        last_xyz = np.array([x, y, z], dtype=float)
        last_t = t_done

        if wp_settle_s > 1e-6:
            time.sleep(float(wp_settle_s))


# =========================
# Main
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--deckga_pkl", default=DEFAULT_DECKGA_PKL)
    parser.add_argument("--num_uavs", type=int, default=3)
    parser.add_argument("--uav_prefix", default="drone")

    parser.add_argument("--frame_id", default=DEFAULT_FRAME_ID)
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)

    # Antarctica-like waits
    parser.add_argument("--wait_actions_s", type=float, default=DEFAULT_WAIT_ACTIONS_S)
    parser.add_argument("--init_wait_s", type=float, default=DEFAULT_INIT_WAIT_S)
    parser.add_argument("--pre_arm_wait_s", type=float, default=DEFAULT_PRE_ARM_WAIT_S)
    parser.add_argument("--takeoff_settle_s", type=float, default=DEFAULT_TAKEOFF_SETTLE_S)

    # Takeoff/hover/land
    parser.add_argument("--takeoff_z", type=float, default=DEFAULT_TAKEOFF_Z)
    parser.add_argument("--hover_s", type=float, default=DEFAULT_HOVER_S)
    parser.add_argument("--land_settle_s", type=float, default=DEFAULT_LAND_SETTLE_S)
    parser.add_argument("--takeoff_sequential", action="store_true")
    parser.add_argument("--first_wp_wait", action="store_true", help="Handshake to first waypoint with wait=True (recommended).")

    # Transform controls
    parser.add_argument("--coord_mode", default=DEFAULT_COORD_MODE, choices=["auto", "original", "shifted"])
    parser.add_argument("--scale_xy", type=float, default=DEFAULT_SCALE_XY)
    parser.add_argument("--scale_z", type=float, default=DEFAULT_SCALE_Z)
    parser.add_argument("--z_offset", type=float, default=DEFAULT_Z_OFFSET)
    parser.add_argument("--z_min", type=float, default=DEFAULT_Z_MIN)

    # Logging
    parser.add_argument("--log_dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--run_tag", default=DEFAULT_RUN_TAG)

    # ROS node options
    parser.add_argument("--use_sim_time", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    # Execution modes
    parser.add_argument("--ensure_reach", action="store_true",
                        help="Guarantee each waypoint is reached: per-UAV threads, go_to(wait=True) per waypoint.")
    parser.add_argument("--wp_settle_s", type=float, default=0.0,
                        help="Extra settle sleep after each reached waypoint (only with --ensure_reach).")

    args = parser.parse_args()

    ensure_dir(args.log_dir)
    run_stamp = now_tag()

    namespaces = [f"{args.uav_prefix}{i}" for i in range(int(args.num_uavs))]

    print("Initializing rclpy...")
    rclpy.init()

    drones: List[DroneInterface] = []
    seg_logs: List[List[Dict[str, Any]]] = []

    try:
        wait_for_actions(namespaces, args.wait_actions_s)

        print(f"Loading DECK-GA paths: {args.deckga_pkl}")
        raw_paths, offset_used = load_deckga_pkl(args.deckga_pkl)

        raw_paths = raw_paths[: len(namespaces)]
        tf = TransformConfig(
            coord_mode=str(args.coord_mode),
            scale_xy=float(args.scale_xy),
            scale_z=float(args.scale_z),
            z_offset=float(args.z_offset),
            z_min=float(args.z_min),
        )

        paths_m = transform_paths(raw_paths, offset_used, tf)

        planned_lengths, planned_times_s, planned_makespan = planned_times(paths_m, args.speed)
        print("\n=== Planned timing (kinematic model) ===")
        print(f"Assumed constant speed: {float(args.speed):.3f} m/s")
        for i in range(len(paths_m)):
            print(f"UAV {i}: length={planned_lengths[i]:.3f} m, planned_time={planned_times_s[i]:.2f} s")
        print(f"Planned mission makespan (parallel UAVs): {planned_makespan:.2f} s")
        print("=======================================\n")

        print("Creating DroneInterface objects...")
        for ns in namespaces:
            drones.append(make_drone_interface(ns, use_sim_time=args.use_sim_time, verbose=args.verbose))

        print(f"Initial settle wait: {float(args.init_wait_s):.1f}s")
        time.sleep(float(args.init_wait_s))

        if float(args.pre_arm_wait_s) > 0.0:
            print(f"Pre-arm warmup wait: {float(args.pre_arm_wait_s):.1f}s")
            time.sleep(float(args.pre_arm_wait_s))

        t_exec_start = time.perf_counter()

        # Arm + Offboard
        t_arm_start = time.perf_counter()
        print("Arming + switching to offboard for all drones...")
        for d in drones:
            safe_offboard_arm(d)
        t_arm_end = time.perf_counter()

        # Takeoff
        t_takeoff_start = time.perf_counter()
        if args.takeoff_sequential:
            print(f"Taking off SEQUENTIALLY to {float(args.takeoff_z):.2f} m (wait=True)...")
            for d in drones:
                safe_takeoff(d, height=float(args.takeoff_z), wait=True)
        else:
            print(f"Taking off ALL drones to {float(args.takeoff_z):.2f} m (wait=True)...")
            for d in drones:
                safe_takeoff(d, height=float(args.takeoff_z), wait=True)
        t_takeoff_end = time.perf_counter()

        print(f"Takeoff settle wait: {float(args.takeoff_settle_s):.1f}s")
        time.sleep(float(args.takeoff_settle_s))

        # First waypoint handshake (recommended for crisp tracking)
        t_handshake_start = time.perf_counter()
        if args.first_wp_wait:
            print("First-waypoint handshake (wait=True per drone)...")
            for i, d in enumerate(drones):
                p = paths_m[i]
                if len(p) == 0:
                    continue
                x, y, z = map(float, p[0])
                print(f"[{namespaces[i]}] FIRST WP (blocking) -> (x={x:.2f}, y={y:.2f}, z={z:.2f})")
                safe_go_to(d, x, y, z, float(args.speed), str(args.frame_id), wait=True)
            time.sleep(1.0)
        t_handshake_end = time.perf_counter()

        # Execute paths
        t_paths_start = time.perf_counter()
        mission_times_s: List[float] = [0.0 for _ in drones]

        if args.ensure_reach:
            print("Executing paths with --ensure_reach (per-UAV threads, wait=True per WP)...")
            t0 = time.perf_counter()

            seg_logs = [[] for _ in range(len(drones))]
            threads: List[threading.Thread] = []

            for i, d in enumerate(drones):
                p = paths_m[i]
                th = threading.Thread(
                    target=_uav_worker_wait_each_wp,
                    args=(namespaces[i], d, p, float(args.speed), str(args.frame_id), float(args.wp_settle_s), t0, seg_logs[i]),
                    daemon=True,
                )
                threads.append(th)
                th.start()

            for th in threads:
                th.join()

            t_paths_end = time.perf_counter()

            for i in range(len(drones)):
                if seg_logs[i]:
                    mission_times_s[i] = float(seg_logs[i][-1]["done_time_s"] - seg_logs[i][0]["cmd_time_s"])
        else:
            # Legacy: wait=False stepping with pacing
            seg_logs = [[] for _ in range(len(drones))]
            last_sent: List[Optional[np.ndarray]] = [None] * len(drones)
            last_cmd_time: List[Optional[float]] = [None] * len(drones)
            first_cmd_abs: List[Optional[float]] = [None] * len(drones)
            last_cmd_abs: List[Optional[float]] = [None] * len(drones)
            cum_dist: List[float] = [0.0] * len(drones)

            max_len = max((len(p) for p in paths_m), default=0)
            print("Executing paths (wait=False) with pacing...")
            for k in range(max_len):
                step_dists: List[float] = []
                for i, d in enumerate(drones):
                    p = paths_m[i]
                    if k >= len(p) or len(p) == 0:
                        continue

                    x, y, z = map(float, p[k])
                    print(f"[{namespaces[i]}] WP {k+1}/{len(p)} -> (x={x:.2f}, y={y:.2f}, z={z:.2f})")
                    safe_go_to(d, x, y, z, float(args.speed), str(args.frame_id), wait=False)

                    t_cmd_abs = time.perf_counter()
                    if first_cmd_abs[i] is None:
                        first_cmd_abs[i] = t_cmd_abs
                    last_cmd_abs[i] = t_cmd_abs

                    seg_dist = 0.0 if last_sent[i] is None else float(np.linalg.norm(np.array([x, y, z]) - last_sent[i]))
                    seg_dt = 0.0 if last_cmd_time[i] is None else float(t_cmd_abs - last_cmd_time[i])

                    cum_dist[i] += seg_dist
                    seg_speed_cmd = (seg_dist / seg_dt) if seg_dt > 1e-9 else 0.0

                    seg_logs[i].append({
                        "wp_idx": int(k + 1),
                        "x": float(x), "y": float(y), "z": float(z),
                        "cmd_time_s": float(t_cmd_abs - t_paths_start),
                        "seg_dist_m": float(seg_dist),
                        "seg_dt_s": float(seg_dt),
                        "seg_speed_cmd_mps": float(seg_speed_cmd),
                        "cum_dist_m": float(cum_dist[i]),
                    })

                    last_sent[i] = np.array([x, y, z], dtype=float)
                    last_cmd_time[i] = t_cmd_abs
                    step_dists.append(seg_dist)

                if step_dists:
                    max_step = max(step_dists)
                    sleep_t = max(0.30, float(max_step / max(float(args.speed), 1e-6)))
                    time.sleep(sleep_t)

            t_paths_end = time.perf_counter()

            for i in range(len(drones)):
                if first_cmd_abs[i] is not None and last_cmd_abs[i] is not None:
                    mission_times_s[i] = float(last_cmd_abs[i] - first_cmd_abs[i])

        # Hover + land (NON-BLOCKING land)
        print(f"Hovering {float(args.hover_s):.2f} seconds, then landing...")
        t_hover_start = time.perf_counter()
        time.sleep(float(args.hover_s))
        t_hover_end = time.perf_counter()

        t_land_start = time.perf_counter()
        for d in drones:
            safe_land(d, wait=False)
        time.sleep(float(args.land_settle_s))
        for d in drones:
            safe_disarm(d)
        t_land_end = time.perf_counter()

        t_exec_end = time.perf_counter()

        # Reporting
        print("\n=== Executed timing (wall-clock) ===")
        print(f"Offboard+arm phase : {_fmt_s(t_arm_end - t_arm_start)}")
        print(f"Takeoff phase      : {_fmt_s(t_takeoff_end - t_takeoff_start)}")
        print(f"Handshake phase    : {_fmt_s(t_handshake_end - t_handshake_start)}")
        print(f"Path execution     : {_fmt_s(t_paths_end - t_paths_start)}")
        print(f"Hover phase        : {_fmt_s(t_hover_end - t_hover_start)}")
        print(f"Landing phase      : {_fmt_s(t_land_end - t_land_start)}")
        print(f"TOTAL (arm->land)  : {_fmt_s(t_exec_end - t_exec_start)}")

        print("\n=== Mission completion time ===")
        for i, tm in enumerate(mission_times_s):
            print(f"UAV {i}: {_fmt_s(tm)}")
        mission_makespan = max(mission_times_s) if mission_times_s else 0.0
        print(f"Mission makespan (parallel UAVs): {_fmt_s(mission_makespan)}")
        print("=========================================================\n")

        # Executed distances (waypoint-to-waypoint)
        executed_uav_dists_m: List[float] = []
        for i in range(len(drones)):
            if not seg_logs or i >= len(seg_logs) or len(seg_logs[i]) == 0:
                executed_uav_dists_m.append(0.0)
            else:
                executed_uav_dists_m.append(float(seg_logs[i][-1].get("cum_dist_m", 0.0)))
        executed_total_dist_m = float(np.sum(executed_uav_dists_m))

        print("\n=== Executed (waypoint-to-waypoint) flying distances ===")
        for i, d_m in enumerate(executed_uav_dists_m):
            print(f"UAV {i}: {d_m:.3f} m")
        print(f"TOTAL (sum UAV0..N): {executed_total_dist_m:.3f} m")
        print("=========================================================\n")

        # CSV logging
        summary_path = str(Path(args.log_dir) / f"run_{args.run_tag}_{run_stamp}_summary.csv")
        d0 = executed_uav_dists_m[0] if len(executed_uav_dists_m) > 0 else 0.0
        d1 = executed_uav_dists_m[1] if len(executed_uav_dists_m) > 1 else 0.0
        d2 = executed_uav_dists_m[2] if len(executed_uav_dists_m) > 2 else 0.0

        summary_header = [
            "run_tag", "timestamp",
            "deckga_pkl",
            "num_uavs", "speed_mps",
            "takeoff_z_m",
            "planned_makespan_s",
            "executed_total_arm_to_land_s",
            "executed_path_phase_s",
            "mission_makespan_s",
            "offboard_arm_s", "takeoff_s", "handshake_s", "hover_s", "landing_s",
            "ensure_reach", "wp_settle_s", "land_settle_s",
            "executed_uav0_dist_m", "executed_uav1_dist_m", "executed_uav2_dist_m", "executed_total_dist_m",
        ]

        summary_row = [[
            str(args.run_tag),
            str(run_stamp),
            str(args.deckga_pkl),
            int(len(drones)),
            float(args.speed),
            float(args.takeoff_z),
            float(planned_makespan),
            float(t_exec_end - t_exec_start),
            float(t_paths_end - t_paths_start),
            float(mission_makespan),
            float(t_arm_end - t_arm_start),
            float(t_takeoff_end - t_takeoff_start),
            float(t_handshake_end - t_handshake_start),
            float(t_hover_end - t_hover_start),
            float(t_land_end - t_land_start),
            bool(args.ensure_reach),
            float(args.wp_settle_s),
            float(args.land_settle_s),
            float(d0), float(d1), float(d2), float(executed_total_dist_m),
        ]]

        write_csv(summary_path, summary_header, summary_row)
        print(f"[LOG] Wrote summary CSV: {summary_path}")

        for i in range(len(drones)):
            uav_path = str(Path(args.log_dir) / f"run_{args.run_tag}_{run_stamp}_uav{i}_segments.csv")
            if args.ensure_reach:
                header = ["wp_idx", "x", "y", "z", "cmd_time_s", "done_time_s", "seg_dist_m", "seg_dt_s",
                          "seg_speed_mps", "cum_dist_m", "ok"]
                rows = [[
                    r["wp_idx"], r["x"], r["y"], r["z"],
                    r["cmd_time_s"], r["done_time_s"],
                    r["seg_dist_m"], r["seg_dt_s"], r["seg_speed_mps"], r["cum_dist_m"], r["ok"]
                ] for r in seg_logs[i]]
            else:
                header = ["wp_idx", "x", "y", "z", "cmd_time_s", "seg_dist_m", "seg_dt_s", "seg_speed_cmd_mps",
                          "cum_dist_m"]
                rows = [[
                    r["wp_idx"], r["x"], r["y"], r["z"],
                    r["cmd_time_s"], r["seg_dist_m"], r["seg_dt_s"], r["seg_speed_cmd_mps"], r["cum_dist_m"]
                ] for r in seg_logs[i]]

            write_csv(uav_path, header, rows)
            print(f"[LOG] Wrote UAV{i} segments CSV: {uav_path}")

    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl+C). Attempting SAFE non-blocking landing for all drones...")
        try:
            for d in drones:
                safe_land(d, wait=False)
            time.sleep(3.0)
            for d in drones:
                safe_disarm(d)
        except Exception:
            pass

    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()