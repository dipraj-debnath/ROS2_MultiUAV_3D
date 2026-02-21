#!/usr/bin/env python3
"""
Traditional GA baseline: Load-Balancing clustering + Traditional GA routing.

DECK_GA.py-style behavior:
- CLI: --points_pkl, --num_uavs, --start_points, --out_pkl, --save_fig_dir
- Loads Nx3 waypoints from .pkl
- Clusters via load_balancing_clustering()
- Prepends per-UAV start point, runs Traditional GA per cluster
- Forces the returned GA tour to START at the start point (important for ROS execution)
- Ensures closed tour (last == first)
- Saves output with SAME downstream key: deckga_paths

IMPORTANT:
Your Load_Balancing.py returns:
  (uav_assignments_dict, centroids)
where uav_assignments_dict[uav] is a python list of points.
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from Load_Balancing import load_balancing_clustering
from Traditional_GA import basic_ga_path_planning


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


def ensure_closed_tour(path: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    """
    Ensure the path returns to start (first point == last point).
    """
    path = np.asarray(path, dtype=float)
    if len(path) == 0:
        return path
    if not np.allclose(path[0], path[-1], atol=atol):
        path = np.vstack([path, path[0]])
    return path


def _find_point_index(path: np.ndarray, pt: np.ndarray, atol: float = 1e-6) -> int | None:
    """
    Find index of pt in path using isclose tolerance.
    Returns None if not found.
    """
    path = np.asarray(path, dtype=float)
    pt = np.asarray(pt, dtype=float)
    if len(path) == 0:
        return None
    idxs = np.where(np.all(np.isclose(path, pt, atol=atol), axis=1))[0]
    if len(idxs) == 0:
        return None
    return int(idxs[0])


def force_start_and_close(path: np.ndarray, start_pt: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    """
    Make sure:
    - start_pt is the FIRST element
    - tour is CLOSED (last == first)

    Works even if GA output doesn't include start_pt.
    """
    path = np.asarray(path, dtype=float)
    start_pt = np.asarray(start_pt, dtype=float)

    if len(path) == 0:
        return ensure_closed_tour(np.asarray([start_pt], dtype=float), atol=atol)

    # If closed, remove last temporarily for rotation/insertion
    closed = np.allclose(path[0], path[-1], atol=atol)
    if closed and len(path) > 1:
        path = path[:-1]

    k = _find_point_index(path, start_pt, atol=atol)
    if k is None:
        # GA didn't include start point -> insert it at the beginning
        path = np.vstack([start_pt, path])
    else:
        # Rotate cyclic tour so it starts at start_pt
        path = np.vstack([path[k:], path[:k]])

    return ensure_closed_tour(path, atol=atol)


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


def normalize_lb_output(lb_out, pts_shifted: np.ndarray, num_uavs: int):
    """
    Your Load_Balancing.py returns:
        (uav_assignments_dict, centroids)

    where:
        uav_assignments_dict[uav] = [pt0, pt1, ...]  (python lists of 3D points)
        centroids = np.ndarray (num_uavs, 3)

    This converts to:
        clusters: list[np.ndarray] where clusters[i] is (Mi,3)
        centroids: np.ndarray (num_uavs,3)
    """
    if not isinstance(lb_out, tuple) or len(lb_out) != 2:
        raise TypeError(
            f"Expected (assignments_dict, centroids) from load_balancing_clustering, got type={type(lb_out)}"
        )

    assignments, centroids = lb_out

    if not isinstance(assignments, dict):
        raise TypeError(
            f"Expected assignments as dict from Load_Balancing.py, got type={type(assignments)}"
        )

    centroids = np.asarray(centroids, dtype=float)
    if centroids.shape != (num_uavs, 3):
        # fallback: still allow but warn shape
        print(f"[WARN] centroids shape unexpected: {centroids.shape}, expected ({num_uavs},3)")

    clusters = []
    for i in range(num_uavs):
        pts_list = assignments.get(i, [])
        if pts_list is None:
            pts_list = []
        arr = np.asarray(pts_list, dtype=float)
        if arr.size == 0:
            arr = np.zeros((0, 3), dtype=float)
        else:
            arr = arr.reshape((-1, 3))
        clusters.append(arr)

    return clusters, centroids


# ----------------------------
# Main
# ----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--points_pkl", required=True, help="Nx3 points pkl (waypoints)")
    p.add_argument("--num_uavs", type=int, required=True)
    p.add_argument("--start_points", required=True, help='Format: "x,y,z;x,y,z;..."')
    p.add_argument("--out_pkl", required=True, help="Output pkl path")
    p.add_argument("--save_fig_dir", default=None, help="Directory to save figures (optional)")
    p.add_argument("--population_size", type=int, default=50)
    p.add_argument("--num_iterations", type=int, default=25000)
    p.add_argument("--no_plot", action="store_true")
    args = p.parse_args()

    points_path = Path(args.points_pkl).expanduser().resolve()
    out_path = Path(args.out_pkl).expanduser().resolve()
    save_dir = Path(args.save_fig_dir).expanduser().resolve() if args.save_fig_dir else None

    num_uavs = int(args.num_uavs)
    points = load_points(points_path)
    num_points = len(points)
    start_points = parse_start_points(args.start_points, num_uavs)

    print("\n--- INPUT ---")
    print("Points file:", str(points_path))
    print("Num points:", num_points)
    print("Num UAVs  :", num_uavs)
    print("Start pts :\n", start_points)
    print("Traditional GA params:", {"population_size": args.population_size, "num_iterations": args.num_iterations})

    # Shift points + starts together to keep consistent offsets
    all_for_shift = np.vstack([points, start_points])
    all_shifted, offset_used = shift_to_positive(all_for_shift)
    pts_shifted = all_shifted[:len(points)]
    start_shifted = all_shifted[len(points):]

    # Step 1: Load Balancing clustering
    t0 = time.time()
    lb_out = load_balancing_clustering(pts_shifted, num_uavs)
    clusters, centroids = normalize_lb_output(lb_out, pts_shifted, num_uavs)
    cluster_time = time.time() - t0

    print("\n--- Load Balancing Output ---")
    for i, c in enumerate(clusters):
        print(f"Cluster {i} size:", int(c.shape[0]))
    print("Centroids:\n", np.asarray(centroids))
    print(f"Load Balancing Time: {cluster_time:.3f} s")

    # Optional clustering plot (shifted)
    if (not args.no_plot) and save_dir is not None:
        fig_cluster = plt.figure(figsize=(4.5, 3.2), dpi=200)
        axc = fig_cluster.add_subplot(111, projection="3d")
        axc.set_title("Load Balancing Clusters (Traditional GA baseline)", fontsize=10)
        colors = cm.rainbow(np.linspace(0, 1, num_uavs))
        for i, c in enumerate(clusters):
            if c.shape[0] > 0:
                axc.scatter(c[:, 0], c[:, 1], c[:, 2], s=10, color=colors[i], label=f"UAV {i}")
        if np.asarray(centroids).shape == (num_uavs, 3):
            axc.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=40, marker="x", label="Centroids")
        axc.set_xlabel("X"); axc.set_ylabel("Y"); axc.set_zlabel("Z")
        axc.legend()
        maybe_save_fig(fig_cluster, save_dir, "traditional_ga_load_balancing_clustering")
        plt.close(fig_cluster)

    # Step 2: Prepend start point per UAV (cluster may be empty)
    clusters_with_start = []
    for i in range(num_uavs):
        c = np.asarray(clusters[i], dtype=float).reshape((-1, 3))
        if c.shape[0] == 0:
            clusters_with_start.append(np.asarray([start_shifted[i]], dtype=float))
        else:
            clusters_with_start.append(np.vstack([start_shifted[i], c]))

    # Raw lengths: closed “start -> points -> start”
    raw_lengths = []
    for i, c in enumerate(clusters_with_start):
        raw_lengths.append(calculate_path_distance(ensure_closed_tour(c)))

    print("\n--- Raw Lengths (start + assigned points, shifted, closed) ---")
    for i, d in enumerate(raw_lengths):
        print(f"UAV {i}: {d:.3f}")
    print("Total raw:", float(np.sum(raw_lengths)))

    # Step 3: Traditional GA routing per UAV cluster
    optimized_paths_shifted = []
    ga_lengths = []

    t1 = time.time()
    for i, cluster_points in enumerate(clusters_with_start):
        print(f"\n--- Running Traditional GA for UAV {i} ---")

        # If only start exists
        if len(cluster_points) <= 1:
            trivial = ensure_closed_tour(cluster_points)
            optimized_paths_shifted.append(trivial)
            ga_lengths.append(calculate_path_distance(trivial))
            print(f"UAV {i} has no assigned waypoints. Length: {ga_lengths[-1]:.3f}")
            continue

        pop_size = min(args.population_size, len(cluster_points))

        opt = basic_ga_path_planning(
            cluster_points,
            population_size=pop_size,
            num_iterations=args.num_iterations,
        )

        # CRITICAL: force correct start + close tour (for ROS execution)
        opt = force_start_and_close(opt, start_shifted[i], atol=1e-6)

        optimized_paths_shifted.append(opt)
        ga_len = calculate_path_distance(opt)
        ga_lengths.append(ga_len)

        print(f"UAV {i} Optimized Length: {ga_len:.3f}")

    ga_time = time.time() - t1

    print("\n--- Optimized Lengths (Traditional GA, shifted) ---")
    for i, d in enumerate(ga_lengths):
        print(f"UAV {i}: {d:.3f}")
    print("Total optimized:", float(np.sum(ga_lengths)))

    # Step 4: Shift paths back to ORIGINAL coordinate space
    deckga_paths = [np.asarray(p, dtype=float) - offset_used for p in optimized_paths_shifted]

    # Sanity
    print("\n--- Path start/stop sanity (original space) ---")
    for i, P in enumerate(deckga_paths):
        P = np.asarray(P, dtype=float)
        print(f"UAV{i}: n={len(P)} first={P[0]} last={P[-1]}")

    # Step 5: Save output
    out = {
        "algo": "TraditionalGA_LoadBalancing",
        "deckga_paths": deckga_paths,
        "raw_lengths": raw_lengths,
        "ga_lengths": ga_lengths,
        "final_lengths": ga_lengths,
        "centroids": np.asarray(centroids, dtype=float),
        "offset_used": np.asarray(offset_used, dtype=float),
        "num_uavs": int(num_uavs),
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
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
        ax.set_title("Optimised UAV Paths (Load Balancing + Traditional GA)", fontsize=10)

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
        maybe_save_fig(fig_combined, save_dir, "traditional_ga_load_balancing_optimized_paths_combined")
        plt.close(fig_combined)

    print("\n--- Timing Summary ---")
    print(f"Load Balancing Time : {cluster_time:.3f} s")
    print(f"Traditional GA Time : {ga_time:.3f} s")
    print(f"Total Time          : {(cluster_time + ga_time):.3f} s")


if __name__ == "__main__":
    main()
