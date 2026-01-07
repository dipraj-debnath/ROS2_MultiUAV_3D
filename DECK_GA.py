#!/usr/bin/env python3
"""
DECK-GA (DCKmeans + GA) planner + plotting + ROS2 output writer.

What this script does:
1) Loads 3D waypoints from a .pkl (Nx3).
2) Runs DCKmeans clustering into num_uavs clusters.
3) Runs GA per cluster to produce optimized closed tours (return-to-start).
4) Saves a ROS2/Aerostack-consumable output pickle (deckga_output.pkl):
   - deckga_paths (list of Nx3 paths, in ORIGINAL coordinates)
   - raw_lengths, dega_lengths, centroids, offset_used, etc.
5) Always shows the final combined optimized-path plot (unless --no_plot).
6) Optionally saves figures if --save_fig_dir is provided.

Run examples:
- Use latest offline dataset:
  python3 DECK_GA.py --points_pkl data/points/points_current.pkl --num_uavs 3

- Save figures:
  python3 DECK_GA.py --points_pkl data/points/points_current.pkl --save_fig_dir results/figs

- Write output explicitly:
  python3 DECK_GA.py --points_pkl data/points/points_current.pkl --out_pkl deckga_ros2/data/deckga_output.pkl
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from DCKmeans import manual_kmeans_clustering
from GA_path_planning import ga_3d_pathplanning

matplotlib.rc("font", family="sans-serif")


# ----------------------------
# Helpers
# ----------------------------
def load_points(pkl_path: Path) -> np.ndarray:
    with pkl_path.open("rb") as f:
        arr = pickle.load(f)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Points must be Nx3. Got shape={arr.shape} from {pkl_path}")
    return arr


def calculate_path_distance(path: np.ndarray) -> float:
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


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


def maybe_save_fig(fig, save_dir: Path | None, name: str):
    if save_dir is None:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"{name}.png", bbox_inches="tight", dpi=300)


# ----------------------------
# Main
# ----------------------------
def main():
    repo_root = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser()
    ap.add_argument("--points_pkl", default="data/points/points_current.pkl")
    ap.add_argument("--out_pkl", default="deckga_ros2/data/deckga_output.pkl")
    ap.add_argument("--num_uavs", type=int, default=3)

    ap.add_argument(
        "--start_points",
        default="10,0,0;50,20,0;90,0,0",
        help='Semicolon-separated: "x,y,z;x,y,z;..." (must match num_uavs)',
    )

    ap.add_argument("--no_plot", action="store_true", help="Disable showing plots")
    ap.add_argument("--save_fig_dir", default=None, help="If set, saves figures (PNG) into this directory")

    # Optional plotting controls (safe defaults: show only final combined plot)
    ap.add_argument("--plot_kmeans", action="store_true", help="Show KMeans clustering plot (optional)")
    ap.add_argument("--plot_per_uav", action="store_true", help="Show individual UAV path plots (optional)")

    args = ap.parse_args()

    points_path = (repo_root / args.points_pkl).resolve() if not Path(args.points_pkl).is_absolute() else Path(args.points_pkl)
    out_path = (repo_root / args.out_pkl).resolve() if not Path(args.out_pkl).is_absolute() else Path(args.out_pkl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dir = None
    if args.save_fig_dir:
        save_dir = (repo_root / args.save_fig_dir).resolve()

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

    # IMPORTANT: shift into positive space for algorithm stability, then shift back later
    all_in = np.vstack([start_points, points])
    all_shifted, offset_used = shift_to_positive(all_in)
    start_shifted = all_shifted[:num_uavs]
    points_shifted = all_shifted[num_uavs:]

    # Step 2: Run KMeans (DCKmeans)
    start_time_kmeans = time.time()
    clusters, centroids = manual_kmeans_clustering(points_shifted, num_uavs)
    elapsed_time_kmeans = time.time() - start_time_kmeans

    print("\n--- DCKmeans Output ---")
    print("Centroids:\n", centroids)
    print(f"DCKmeans completed in {elapsed_time_kmeans:.3f} s")

    # Optional: KMeans plot
    if (not args.no_plot) and args.plot_kmeans:
        fig_kmeans = plt.figure()
        ax_kmeans = fig_kmeans.add_subplot(111, projection="3d")
        ax_kmeans.set_title("DCKmeans Clustering (shifted space)")
        colors = cm.rainbow(np.linspace(0, 1, num_uavs))
        for i, cluster in enumerate(clusters):
            cp = np.asarray(cluster, dtype=float)
            ax_kmeans.scatter(cp[:, 0], cp[:, 1], cp[:, 2], color=colors[i], label=f"Cluster {i+1}")
        ax_kmeans.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c="black", s=100, marker="x", label="Centroids")
        ax_kmeans.set_xlabel("X")
        ax_kmeans.set_ylabel("Y")
        ax_kmeans.set_zlabel("Z")
        ax_kmeans.legend()
        maybe_save_fig(fig_kmeans, save_dir, "kmeans_clustering")
        plt.show()

    # Step 3: Add each UAV start point to its cluster
    clusters_with_start = []
    for i, cluster in enumerate(clusters):
        cluster_arr = np.asarray(cluster, dtype=float)
        clusters_with_start.append(np.vstack([start_shifted[i], cluster_arr]))

    raw_lengths = [calculate_path_distance(c) for c in clusters_with_start]
    print("\n--- Raw Path Lengths (before GA, shifted) ---")
    for i, d in enumerate(raw_lengths):
        print(f"UAV {i}: {d:.3f}")
    print("Total raw:", float(np.sum(raw_lengths)))

    # Step 4: GA optimization per UAV cluster
    optimized_paths_shifted = []
    dega_lengths = []
    start_time_ga = time.time()

    for i, cluster_points in enumerate(clusters_with_start):
        print(f"\n--- GA Input for UAV {i} (shifted) ---")
        print(cluster_points)

        opt = ga_3d_pathplanning(cluster_points)
        opt = ensure_closed_tour(opt)

        optimized_paths_shifted.append(opt)
        dega_len = calculate_path_distance(opt)
        dega_lengths.append(dega_len)

        print(f"--- GA Output for UAV {i} (shifted) ---")
        print(opt)
        print(f"Length: {dega_len:.3f}")

    elapsed_time_ga = time.time() - start_time_ga
    print(f"\nGA completed in {elapsed_time_ga:.3f} s")

    print("\n--- Optimized Path Lengths (after GA, shifted) ---")
    for i, d in enumerate(dega_lengths):
        print(f"UAV {i}: {d:.3f}")
    print("Total optimized:", float(np.sum(dega_lengths)))

    improvement = np.asarray(raw_lengths) - np.asarray(dega_lengths)
    print("\n--- Improvement (raw - optimized) ---")
    for i, imp in enumerate(improvement):
        print(f"UAV {i}: {imp:.3f}")
    print("GA improved all UAVs?", bool(np.all(np.asarray(dega_lengths) <= np.asarray(raw_lengths) + 1e-9)))

    # Step 5: Shift optimized paths back to ORIGINAL coordinate space
    deckga_paths = [np.asarray(p, dtype=float) - offset_used for p in optimized_paths_shifted]

    # Step 6: Save ROS2/Aerostack output file
    out = {
        "deckga_paths": deckga_paths,   # consumed by deckga_execute.py
        "raw_lengths": raw_lengths,
        "dega_lengths": dega_lengths,
        "final_lengths": dega_lengths,
        "centroids": centroids,         # shifted-space centroids (fine for analysis)
        "offset_used": offset_used,
        "num_uavs": num_uavs,
        "num_points": int(num_points),
        "points_file": str(points_path),
    }

    with out_path.open("wb") as f:
        pickle.dump(out, f)

    print("\n--- Saved Output ---")
    print("Output pkl:", out_path)
    print("Keys:", sorted(out.keys()))
    print("Offset used:", offset_used)

    # Step 7: Plot at least the final combined optimized paths (ORIGINAL space)
    if not args.no_plot:
        fig_combined = plt.figure(figsize=(4.5, 3.2), dpi=200)
        ax = fig_combined.add_subplot(111, projection="3d")
        ax.set_title("Optimised UAV Paths (DECK-GA)", fontsize=10)

        colors = cm.rainbow(np.linspace(0, 1, len(deckga_paths)))
        for i, path in enumerate(deckga_paths):
            path = np.asarray(path, dtype=float)
            ax.plot(path[:, 0], path[:, 1], path[:, 2], "*-", label=f"UAV {i+1}", color=colors[i])
            ax.scatter(path[0, 0], path[0, 1], path[0, 2], c="red", s=60, marker="o")

        ax.set_xlabel("X", fontsize=9)
        ax.set_ylabel("Y", fontsize=9)
        ax.set_zlabel("Z", fontsize=9)
        # ax.legend()  # optional
        maybe_save_fig(fig_combined, save_dir, "deckga_optimized_paths_combined")
        plt.show()

        # Optional: per-UAV plots
        if args.plot_per_uav:
            for i, path in enumerate(deckga_paths):
                fig = plt.figure()
                ax2 = fig.add_subplot(111, projection="3d")
                ax2.plot(path[:, 0], path[:, 1], path[:, 2], "*-", label=f"UAV {i+1}")
                ax2.scatter(path[0, 0], path[0, 1], path[0, 2], c="red", s=100, marker="o", label="Start")
                ax2.set_title(f"Optimized Path UAV {i+1}")
                ax2.set_xlabel("X")
                ax2.set_ylabel("Y")
                ax2.set_zlabel("Z")
                ax2.legend()
                maybe_save_fig(fig, save_dir, f"deckga_uav{i+1}_path")
                plt.show()

    # Summary timing
    print("\n--- Summary ---")
    print(f"DCKmeans Time: {elapsed_time_kmeans:.3f} s")
    print(f"GA Time     : {elapsed_time_ga:.3f} s")
    print(f"Total Time  : {(elapsed_time_kmeans + elapsed_time_ga):.3f} s")


if __name__ == "__main__":
    main()
