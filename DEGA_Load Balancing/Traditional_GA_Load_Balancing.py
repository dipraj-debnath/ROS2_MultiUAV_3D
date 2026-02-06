#!/usr/bin/env python3
"""
Traditional GA baseline: Load-Balancing clustering + Traditional GA routing.

DECK_GA.py-style behavior:
- CLI: --points_pkl, --num_uavs, --start_points, --out_pkl, --save_fig_dir
- Loads Nx3 waypoints from .pkl
- Clusters via load_balancing_clustering()
- Prepends per-UAV start point, runs Traditional GA per cluster
- Ensures closed tour
- Saves output with SAME downstream key: deckga_paths
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from Load_Balancing import load_balancing_clustering          # :contentReference[oaicite:2]{index=2}
from Traditional_GA import basic_ga_path_planning              # :contentReference[oaicite:3]{index=3}


# ----------------------------
# Helpers (DECK-style)
# ----------------------------
def load_points(pkl_path: Path) -> np.ndarray:
    with pkl_path.open("rb") as f:
        arr = pickle.load(f)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Points must be Nx3. Got shape={arr.shape} from {pkl_path}")
    return arr


def parse_start_points(s: str, num_uavs: int) -> np.ndarray:
    """
    Format: "x,y,z;x,y,z;..."
    """
    sp = []
    for item in s.split(";"):
        xyz = [float(v) for v in item.split(",")]
        if len(xyz) != 3:
            raise ValueError('Each start point must be "x,y,z"')
        sp.append(xyz)
    arr = np.asarray(sp, dtype=float)
    if arr.shape != (num_uavs, 3):
        raise ValueError(f"start_points must be shape ({num_uavs},3), got {arr.shape}")
    return arr


def shift_to_positive(all_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Shift all points so min(x), min(y), min(z) become >= 0.
    Returns: (shifted_points, offset_used)
    """
    mins = np.min(all_points, axis=0)
    offset = np.where(mins < 0.0, -mins, 0.0)
    return all_points + offset, offset


def ensure_closed_tour(path: np.ndarray) -> np.ndarray:
    """
    Ensure the path returns to start (first point == last point).
    """
    path = np.asarray(path, dtype=float)
    if len(path) == 0:
        return path
    if not np.allclose(path[0], path[-1]):
        path = np.vstack([path, path[0]])
    return path


def calculate_path_distance(path: np.ndarray) -> float:
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def maybe_save_fig(fig, save_dir: Path | None, name: str):
    if save_dir is None:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"{name}.png", bbox_inches="tight", dpi=300)


# ----------------------------
# Main
# ----------------------------
def main():
    # repo root: .../ROS2_MultiUAV_3D
    repo_root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser()
    ap.add_argument("--points_pkl", required=True, help="Path to Nx3 points .pkl")
    ap.add_argument("--out_pkl", required=True, help="Output .pkl path (planner output)")
    ap.add_argument("--num_uavs", type=int, default=3)
    ap.add_argument(
        "--start_points",
        required=True,
        help='Semicolon-separated: "x,y,z;x,y,z;..." (must match num_uavs)',
    )

    # Traditional GA controls (explicit for benchmarking)
    ap.add_argument("--population_size", type=int, default=50)
    ap.add_argument("--num_iterations", type=int, default=25000)

    ap.add_argument("--save_fig_dir", default=None, help="If set, saves figures (PNG) into this directory")
    ap.add_argument("--no_plot", action="store_true", help="Disable plotting")
    ap.add_argument("--plot_cluster", action="store_true", help="Plot clustering (optional)")

    args = ap.parse_args()

    points_path = (repo_root / args.points_pkl).resolve() if not Path(args.points_pkl).is_absolute() else Path(args.points_pkl)
    out_path = (repo_root / args.out_pkl).resolve() if not Path(args.out_pkl).is_absolute() else Path(args.out_pkl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dir = (repo_root / args.save_fig_dir).resolve() if args.save_fig_dir else None

    # Step 1: Load points
    points = load_points(points_path)
    num_points = len(points)
    num_uavs = args.num_uavs
    start_points = parse_start_points(args.start_points, num_uavs)

    print("\n--- INPUT ---")
    print("Points file:", points_path)
    print("Num points:", num_points)
    print("Num UAVs  :", num_uavs)
    print("Start pts :", start_points)
    print("Traditional GA params:", {"population_size": args.population_size, "num_iterations": args.num_iterations})

    # DECK-style shifting: operate in non-negative space, then shift back
    all_in = np.vstack([start_points, points])
    all_shifted, offset_used = shift_to_positive(all_in)
    start_shifted = all_shifted[:num_uavs]
    points_shifted = all_shifted[num_uavs:]

    # Step 2: Load Balancing clustering
    t0 = time.time()
    clusters_dict, centroids = load_balancing_clustering(points_shifted, num_uavs)  # :contentReference[oaicite:4]{index=4}
    cluster_time = time.time() - t0

    # Convert dict -> list in UAV index order
    clusters = []
    for i in range(num_uavs):
        pts_i = np.asarray(clusters_dict.get(i, []), dtype=float)
        if pts_i.size == 0:
            pts_i = np.zeros((0, 3), dtype=float)
        clusters.append(pts_i)

    centroids = np.asarray(centroids, dtype=float)

    print("\n--- Load Balancing Output ---")
    for i, c in enumerate(clusters):
        print(f"Cluster {i} size: {len(c)}")
    print("Centroids:\n", centroids)
    print(f"Load Balancing Time: {cluster_time:.3f} s")

    # Optional cluster plot
    if (not args.no_plot) and args.plot_cluster:
        fig_cluster = plt.figure()
        axc = fig_cluster.add_subplot(111, projection="3d")
        axc.set_title("Load Balancing Clustering (shifted space)")
        colors = cm.rainbow(np.linspace(0, 1, num_uavs))
        for i, cluster in enumerate(clusters):
            if len(cluster) > 0:
                axc.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], color=colors[i], label=f"Cluster {i+1}")
        axc.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c="black", s=100, marker="x", label="Centroids")
        axc.set_xlabel("X"); axc.set_ylabel("Y"); axc.set_zlabel("Z")
        axc.legend()
        maybe_save_fig(fig_cluster, save_dir, "traditional_ga_load_balancing_clustering")
        plt.show()

    # Step 3: Prepend each UAV start point
    clusters_with_start = [np.vstack([start_shifted[i], clusters[i]]) for i in range(num_uavs)]
    raw_lengths = [calculate_path_distance(c) for c in clusters_with_start]

    print("\n--- Raw Lengths (start + assigned points, shifted) ---")
    for i, d in enumerate(raw_lengths):
        print(f"UAV {i}: {d:.3f}")
    print("Total raw:", float(np.sum(raw_lengths)))

    # Step 4: Traditional GA routing per UAV cluster
    optimized_paths_shifted = []
    ga_lengths = []

    t1 = time.time()
    for i, cluster_points in enumerate(clusters_with_start):
        print(f"\n--- Running Traditional GA for UAV {i} ---")

        # If only start point exists, path is just [start, start]
        if len(cluster_points) <= 1:
            print(f"Cluster {i} has no assigned waypoints. Using trivial tour.")
            trivial = ensure_closed_tour(cluster_points)
            optimized_paths_shifted.append(trivial)
            ga_lengths.append(calculate_path_distance(trivial))
            continue

        # Make population size safe if cluster is small
        pop_size = min(args.population_size, len(cluster_points))

        opt = basic_ga_path_planning(
            cluster_points,
            population_size=pop_size,
            num_iterations=args.num_iterations,
        )  # :contentReference[oaicite:5]{index=5}
        opt = ensure_closed_tour(opt)

        optimized_paths_shifted.append(opt)
        ga_len = calculate_path_distance(opt)
        ga_lengths.append(ga_len)

        print(f"UAV {i} Optimized Length: {ga_len:.3f}")

    ga_time = time.time() - t1

    print("\n--- Optimized Lengths (Traditional GA, shifted) ---")
    for i, d in enumerate(ga_lengths):
        print(f"UAV {i}: {d:.3f}")
    print("Total optimized:", float(np.sum(ga_lengths)))

    # Step 5: Shift paths back to ORIGINAL coordinate space
    deckga_paths = [np.asarray(p, dtype=float) - offset_used for p in optimized_paths_shifted]

    # Step 6: Save output (keep same key for downstream reuse)
    out = {
        "algo": "TraditionalGA_LoadBalancing",
        "deckga_paths": deckga_paths,
        "raw_lengths": raw_lengths,
        "ga_lengths": ga_lengths,
        "final_lengths": ga_lengths,
        "centroids": centroids,           # shifted-space centroids (analysis)
        "offset_used": offset_used,
        "num_uavs": num_uavs,
        "num_points": int(num_points),
        "points_file": str(points_path),
        "cluster_time_s": float(cluster_time),
        "ga_time_s": float(ga_time),
        "total_time_s": float(cluster_time + ga_time),
        "ga_params": {
            "population_size": int(args.population_size),
            "num_iterations": int(args.num_iterations),
        },
    }

    with out_path.open("wb") as f:
        pickle.dump(out, f)

    print("\n--- Saved Output ---")
    print("Output pkl:", out_path)
    print("Keys:", sorted(out.keys()))

    # Step 7: Plot combined optimized paths (ORIGINAL space)
    if not args.no_plot:
        fig = plt.figure(figsize=(4.8, 3.4), dpi=200)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Optimised UAV Paths (Load Balancing + Traditional GA)", fontsize=10)

        colors = cm.rainbow(np.linspace(0, 1, num_uavs))
        for i, path in enumerate(deckga_paths):
            path = np.asarray(path, dtype=float)
            if len(path) == 0:
                continue
            ax.plot(path[:, 0], path[:, 1], path[:, 2], "*-", label=f"UAV {i+1}", color=colors[i])
            ax.scatter(path[0, 0], path[0, 1], path[0, 2], c="red", s=50, marker="o")

        ax.set_xlabel("X", fontsize=9)
        ax.set_ylabel("Y", fontsize=9)
        ax.set_zlabel("Z", fontsize=9)
        maybe_save_fig(fig, save_dir, "traditional_ga_load_balancing_optimized_paths_combined")
        plt.show()

    print("\n--- Timing Summary ---")
    print(f"Load Balancing Time : {cluster_time:.3f} s")
    print(f"Traditional GA Time : {ga_time:.3f} s")
    print(f"Total Time          : {(cluster_time + ga_time):.3f} s")


if __name__ == "__main__":
    main()
