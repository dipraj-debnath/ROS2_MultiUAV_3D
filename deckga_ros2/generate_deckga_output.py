#!/usr/bin/env python3
"""
Generate DECK_GA-only output for Aerostack execution.

- Loads waypoints (Nx3) from a .pkl (numpy array)
- Shifts all points (and start points) into ++ quadrant if needed
- Runs DECK_GA = DCKmeans clustering + GA 3D path planning per cluster
- Shifts optimized routes back to original coordinates
- Saves: deckga_ros2/data/deckga_output.pkl with key: 'deckga_paths'
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

# Ensure repo root is on sys.path so imports work regardless of where script is run from
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DCKmeans import manual_kmeans_clustering
from GA_path_planning import ga_3d_pathplanning


def calculate_path_distance(path: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def shift_to_positive(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Shift points into the ++ quadrant so that min(x),min(y),min(z) are >= 0.
    Returns (shifted_points, offset).
    """
    mins = np.min(points, axis=0)
    offset = np.where(mins < 0.0, -mins, 0.0)
    return points + offset, offset


def parse_start_points(s: str, num_uavs: int) -> np.ndarray:
    """
    Format: "x,y,z;x,y,z;..."
    """
    sp = []
    for item in s.split(";"):
        xyz = [float(v) for v in item.split(",")]
        if len(xyz) != 3:
            raise ValueError("Each start point must be x,y,z")
        sp.append(xyz)
    arr = np.asarray(sp, dtype=float)
    if arr.shape != (num_uavs, 3):
        raise ValueError(f"start_points must be shape ({num_uavs},3), got {arr.shape}")
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points_pkl", default=str(REPO_ROOT / "points.pkl"))
    ap.add_argument("--num_uavs", type=int, default=3)
    ap.add_argument(
        "--start_points",
        default="10,0,0;50,20,0;90,0,0",
        help='Semicolon-separated: "x,y,z;x,y,z;..." (must match num_uavs)',
    )
    ap.add_argument("--out_pkl", default=str(REPO_ROOT / "deckga_ros2" / "data" / "deckga_output.pkl"))
    args = ap.parse_args()

    points_path = Path(args.points_pkl)
    out_path = Path(args.out_pkl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with points_path.open("rb") as f:
        points = np.asarray(pickle.load(f), dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be Nx3, got {points.shape}")

    start_points = parse_start_points(args.start_points, args.num_uavs)

    # Shift all to ++ quadrant for DECK_GA (handles --, -+, +-, ++ inputs)
    all_in = np.vstack([start_points, points])
    all_shifted, offset = shift_to_positive(all_in)
    start_shifted = all_shifted[: args.num_uavs]
    points_shifted = all_shifted[args.num_uavs :]

    # DECK_GA pipeline
    clusters, centroids = manual_kmeans_clustering(points_shifted, args.num_uavs)

    clusters_with_start = []
    for i, cluster in enumerate(clusters):
        cluster = np.asarray(cluster, dtype=float)
        clusters_with_start.append(np.vstack([start_shifted[i], cluster]))

    raw_lengths = [calculate_path_distance(c) for c in clusters_with_start]

    optimized_paths_shifted = []
    dega_lengths = []
    for cluster_points in clusters_with_start:
        opt = ga_3d_pathplanning(cluster_points)
        optimized_paths_shifted.append(opt)
        dega_lengths.append(calculate_path_distance(opt))

    # Shift back to original coordinates for ROS2/Aerostack execution
    deckga_paths = [p - offset for p in optimized_paths_shifted]
    optimized_paths = [p - offset for p in optimized_paths_shifted]

    result = {
        "deckga_paths": deckga_paths,         # consumed by deckga_execute.py
        "optimized_paths": optimized_paths,   # for analysis
        "raw_lengths": raw_lengths,
        "dega_lengths": dega_lengths,
        "final_lengths": dega_lengths,
        "centroids": centroids,
        "offset_used": offset,
        "num_uavs": args.num_uavs,
        "num_points": int(points.shape[0]),
    }

    with out_path.open("wb") as f:
        pickle.dump(result, f)

    print("Saved:", out_path)
    print("Keys:", sorted(result.keys()))
    print("Offset used:", offset)


if __name__ == "__main__":
    main()
