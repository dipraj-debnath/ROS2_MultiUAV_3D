#!/usr/bin/env python3
"""
Generate synthetic 3D waypoints for DECK_GA experiments.

- Supports any coordinate range, including negative quadrants.
- Saves an Nx3 numpy array to a .pkl file.

Example:
  python3 data/points/generate_points_xyz.py --n 30 --x -100 100 --y -100 100 --z 0 50 --out data/points/points_30_anyquad.pkl
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="Number of waypoints")
    ap.add_argument("--x", type=float, nargs=2, default=[0, 100], metavar=("XMIN", "XMAX"))
    ap.add_argument("--y", type=float, nargs=2, default=[0, 100], metavar=("YMIN", "YMAX"))
    ap.add_argument("--z", type=float, nargs=2, default=[0, 100], metavar=("ZMIN", "ZMAX"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/points/points.pkl", help="Output .pkl path")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    xs = rng.uniform(args.x[0], args.x[1], size=args.n)
    ys = rng.uniform(args.y[0], args.y[1], size=args.n)
    zs = rng.uniform(args.z[0], args.z[1], size=args.n)

    points = np.vstack([xs, ys, zs]).T.astype(float)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(points, f)

    print("Saved:", out_path)
    print("Shape:", points.shape)
    print(
        "Ranges:",
        f"x[{points[:,0].min():.3f},{points[:,0].max():.3f}]",
        f"y[{points[:,1].min():.3f},{points[:,1].max():.3f}]",
        f"z[{points[:,2].min():.3f},{points[:,2].max():.3f}]",
    )


if __name__ == "__main__":
    main()
