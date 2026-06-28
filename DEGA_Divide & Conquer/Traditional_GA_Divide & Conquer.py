#!/usr/bin/env python3
"""
Traditional GA baseline: Divide-and-Conquer clustering + Traditional GA routing.

DECK_GA.py-style behavior:
- CLI: --points_pkl, --num_uavs, --start_points, --out_pkl, --save_fig_dir
- Loads Nx3 waypoints from .pkl
- Clusters via divide_and_conquer_clustering()
- Prepends per-UAV start point, runs Traditional GA per cluster
- Forces the returned GA tour to START at the start point (important for ROS execution)
- Ensures closed tour (last == first)
- Saves output with SAME downstream key: deckga_paths

Notes:
- Some clustering implementations return clusters as:
    (A) list of arrays of points (Nx3), OR
    (B) list/array of integer labels per point, OR
    (C) list of lists of indices
  This script normalizes all of these to per-UAV Nx3 point arrays.
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from Divide_And_Conquer import divide_and_conquer_clustering
from Traditional_GA import basic_ga_path_planning


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


def rotate_route_to_start(path: np.ndarray, start_pt: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    """
    Rotate a cyclic tour so that the first point is start_pt.
    If start_pt is missing (shouldn't happen if we prepended it), we insert it.
    """
    path = np.asarray(path, dtype=float)
    if len(path) == 0:
        return path

    closed = np.allclose(path[0], path[-1], atol=atol)
    if closed:
        core = path[:-1]
    else:
        core = path

    idxs = np.where(np.all(np.isclose(core, start_pt, atol=atol), axis=1))[0]
    if len(idxs) == 0:
        # GA returned a route without the explicit start point -> enforce it
        core = np.vstack([start_pt, core])
        return ensure_closed_tour(core)

    k = int(idxs[0])
    core = np.vstack([core[k:], core[:k]])
    return ensure_closed_tour(core)


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


def _is_intlike_array(a: np.ndarray) -> bool:
    if a.size == 0:
        return False
    if np.issubdtype(a.dtype, np.integer):
        return True
    if np.issubdtype(a.dtype, np.floating):
        return np.all(np.isclose(a, np.round(a)))
    return False


def normalize_clusters(
    clusters_raw,
    points_shifted: np.ndarray,
    num_uavs: int,
) -> list[np.ndarray]:
    """
    Normalize clustering output to: list of length num_uavs, each is (Ni,3) float array.

    Supports:
    - labels array of shape (N,) with integer cluster ids
    - list of index lists/arrays
    - list of point arrays (Ni,3)
    - dict-like {cluster_id: points/indices}
    """
    # Case: dict mapping
    if isinstance(clusters_raw, dict):
        out = []
        for i in range(num_uavs):
            v = clusters_raw.get(i, [])
            out.append(_cluster_item_to_points(v, points_shifted))
        return out

    arr = np.asarray(clusters_raw, dtype=object)

    # Case: labels vector
    if arr.ndim == 1 and len(arr) == len(points_shifted):
        # Could be labels, could be list-of-clusters. Disambiguate:
        if _is_intlike_array(np.asarray(arr, dtype=float)):
            labels = np.asarray(arr, dtype=int)
            out = []
            for i in range(num_uavs):
                idx = np.where(labels == i)[0]
                out.append(np.asarray(points_shifted[idx], dtype=float))
            return out

    # Case: list-like of clusters
    if hasattr(clusters_raw, "__len__"):
        out = []
        for i in range(num_uavs):
            if i < len(clusters_raw):
                out.append(_cluster_item_to_points(clusters_raw[i], points_shifted))
            else:
                out.append(np.zeros((0, 3), dtype=float))
        return out

    raise TypeError(f"Unrecognized clusters output type: {type(clusters_raw)}")


def _cluster_item_to_points(item, points_shifted: np.ndarray) -> np.ndarray:
    """
    Convert one cluster 'item' into (Ni,3) points.
    item can be:
    - (Ni,3) points
    - list/array of indices
    - empty / scalar
    """
    if item is None:
        return np.zeros((0, 3), dtype=float)

    a = np.asarray(item)

    # Empty
    if a.size == 0:
        return np.zeros((0, 3), dtype=float)

    # Already points (Ni,3)
    if a.ndim == 2 and a.shape[1] == 3:
        return np.asarray(a, dtype=float)

    # Indices list
    if a.ndim == 1 and _is_intlike_array(a.astype(float, copy=False)):
        idx = np.asarray(a, dtype=int)
        if idx.size == 0:
            return np.zeros((0, 3), dtype=float)
        return np.asarray(points_shifted[idx], dtype=float)

    # Scalar (bad/degenerate)
    if a.ndim == 0:
        # treat as single index if int-like
        if np.isfinite(float(a)) and np.isclose(float(a), round(float(a))):
            idx = int(round(float(a)))
            if 0 <= idx < len(points_shifted):
                return np.asarray(points_shifted[[idx]], dtype=float)
        return np.zeros((0, 3), dtype=float)

    # Fallback: cannot interpret
    raise TypeError(f"Cannot interpret cluster item with shape={a.shape} dtype={a.dtype}")


# ----------------------------
# Main
# ----------------------------
def main():
    repo_root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser()
    ap.add_argument("--points_pkl", required=True, help="Path to Nx3 points .pkl")
    ap.add_argument("--out_pkl", required=True, help="Output .pkl path (planner output)")
    ap.add_argument("--num_uavs", type=int, required=True)
    ap.add_argument(
        "--start_points",
        required=True,
        help='Semicolon-separated: "x,y,z;x,y,z;..." (must match num_uavs)',
    )

    # Traditional GA controls (kept explicit for benchmarking)
    ap.add_argument("--population_size", type=int, default=50)
    ap.add_argument("--num_iterations", type=int, default=50000)

    ap.add_argument("--save_fig_dir", default=None, help="If set, saves figures (PNG) into this directory")
    ap.add_argument("--no_plot", action="store_true", help="Disable plotting")
    ap.add_argument("--plot_cluster", action="store_true", help="Plot clustering (optional)")
    args = ap.parse_args()

    points_path = (repo_root / args.points_pkl).resolve() if not Path(args.points_pkl).is_absolute() else Path(args.points_pkl).resolve()
    out_path = (repo_root / args.out_pkl).resolve() if not Path(args.out_pkl).is_absolute() else Path(args.out_pkl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dir = (repo_root / args.save_fig_dir).resolve() if args.save_fig_dir else None

    # Step 1: Load points
    points = load_points(points_path)
    num_points = len(points)
    num_uavs = int(args.num_uavs)
    start_points = parse_start_points(args.start_points, num_uavs)

    print("\n--- INPUT ---")
    print("Points file:", str(points_path))
    print("Num points:", num_points)
    print("Num UAVs  :", num_uavs)
    print("Start pts :", start_points)
    print("Traditional GA params:", {"population_size": args.population_size, "num_iterations": args.num_iterations})

    # Shift points + starts together to keep consistent offsets
    all_in = np.vstack([points, start_points])
    all_shifted, offset_used = shift_to_positive(all_in)
    points_shifted = all_shifted[:len(points)]
    start_shifted = all_shifted[len(points):]

    # Step 2: Divide-and-Conquer clustering
    t0 = time.time()
    clusters_raw, centroids = divide_and_conquer_clustering(points_shifted, num_uavs)
    cluster_time = time.time() - t0

    clusters = normalize_clusters(clusters_raw, points_shifted, num_uavs)
    centroids = np.asarray(centroids, dtype=float)

    print("\n--- Divide & Conquer Output ---")
    for i, c in enumerate(clusters):
        print(f"Cluster {i} size:", int(len(c)))
    if centroids.size != 0:
        print("Centroids:\n", centroids)
    print(f"Divide & Conquer Time: {cluster_time:.3f} s")

    # Optional cluster plot (shifted space)
    if (not args.no_plot) and args.plot_cluster and save_dir is not None:
        fig_cluster = plt.figure(figsize=(4.5, 3.2), dpi=200)
        axc = fig_cluster.add_subplot(111, projection="3d")
        axc.set_title("Divide & Conquer Clusters (Traditional GA baseline)", fontsize=10)
        colors = cm.rainbow(np.linspace(0, 1, num_uavs))
        for i, c in enumerate(clusters):
            if len(c) == 0:
                continue
            c = np.asarray(c, dtype=float)
            axc.scatter(c[:, 0], c[:, 1], c[:, 2], s=10, color=colors[i], label=f"UAV {i}")
        if centroids.size != 0 and centroids.ndim == 2 and centroids.shape[1] == 3:
            axc.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=40, marker="x", label="Centroids")
        axc.set_xlabel("X"); axc.set_ylabel("Y"); axc.set_zlabel("Z")
        axc.legend()
        maybe_save_fig(fig_cluster, save_dir, "traditional_ga_divide_conquer_clustering")
        plt.close(fig_cluster)

    # Step 3: Prepend start point per UAV
    clusters_with_start = [np.vstack([start_shifted[i], np.asarray(clusters[i], dtype=float)]) for i in range(num_uavs)]
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

        if len(cluster_points) <= 1:
            print(f"Cluster {i} has no assigned waypoints. Using trivial tour.")
            trivial = ensure_closed_tour(cluster_points)
            optimized_paths_shifted.append(trivial)
            ga_lengths.append(calculate_path_distance(trivial))
            continue

        pop_size = min(args.population_size, len(cluster_points))

        opt = basic_ga_path_planning(
            cluster_points,
            population_size=pop_size,
            num_iterations=args.num_iterations,
        )

        # CRITICAL: force route to start at the start point (for ROS execution)
        opt = rotate_route_to_start(opt, start_shifted[i], atol=1e-6)

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

    # Step 6: Save output
    out = {
        "algo": "TraditionalGA_DivideConquer",
        "deckga_paths": deckga_paths,
        "raw_lengths": raw_lengths,
        "ga_lengths": ga_lengths,
        "final_lengths": ga_lengths,
        "centroids": centroids,
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
    print("Offset used:", offset_used)

    # Optional combined plot (original space)
    if (not args.no_plot) and save_dir is not None:
        fig_combined = plt.figure(figsize=(4.5, 3.2), dpi=200)
        ax = fig_combined.add_subplot(111, projection="3d")
        ax.set_title("Optimised UAV Paths (Divide & Conquer + Traditional GA)", fontsize=10)

        colors = cm.rainbow(np.linspace(0, 1, len(deckga_paths)))
        for i, path in enumerate(deckga_paths):
            path = np.asarray(path, dtype=float)
            if len(path) == 0:
                continue
            ax.plot(path[:, 0], path[:, 1], path[:, 2], "*-", label=f"UAV {i}", color=colors[i])
            ax.scatter(path[0, 0], path[0, 1], path[0, 2], c="red", s=60, marker="o")

        ax.set_xlabel("X", fontsize=9)
        ax.set_ylabel("Y", fontsize=9)
        ax.set_zlabel("Z", fontsize=9)
        maybe_save_fig(fig_combined, save_dir, "traditional_ga_divide_conquer_optimized_paths_combined")
        plt.close(fig_combined)

    print("\n--- Timing Summary ---")
    print(f"Divide & Conquer Time : {cluster_time:.3f} s")
    print(f"Traditional GA Time   : {ga_time:.3f} s")
    print(f"Total Time            : {(cluster_time + ga_time):.3f} s")


if __name__ == "__main__":
    main()
