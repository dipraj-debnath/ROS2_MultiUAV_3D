#!/usr/bin/env python3
"""
Generate synthetic 3D waypoints for DECK_GA experiments.

- Supports any coordinate range, including negative quadrants.
- Saves an Nx3 numpy array to a .pkl file.
- ALSO saves a .txt file with the same points (for reproducibility / inspection).

Example:
  python3 data/points/generate_points_xyz.py \
    --n 30 --x -100 100 --y -100 100 --z 10 100 --seed 11 \
    --out data/points/points_seed11_n30_z10_100.pkl \
    --out_txt data/points/points_seed11_n30_z10_100.txt
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


def _default_txt_path(out_pkl: Path) -> Path:
    # Replace .pkl with .txt (or append .txt if no suffix)
    if out_pkl.suffix.lower() == ".pkl":
        return out_pkl.with_suffix(".txt")
    return out_pkl.with_name(out_pkl.name + ".txt")


def _save_txt(points: np.ndarray, txt_path: Path, fmt: str, precision: int) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt not in ("csv", "space"):
        raise ValueError("--txt_format must be 'csv' or 'space'")

    sep = "," if fmt == "csv" else " "
    format_str = f"%.{int(precision)}f"

    header = "x,y,z" if fmt == "csv" else "x y z"
    np.savetxt(
        txt_path,
        points,
        fmt=format_str,
        delimiter=sep,
        header=header,
        comments="",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="Number of waypoints")
    ap.add_argument("--x", type=float, nargs=2, default=[0, 100], metavar=("XMIN", "XMAX"))
    ap.add_argument("--y", type=float, nargs=2, default=[0, 100], metavar=("YMIN", "YMAX"))
    ap.add_argument("--z", type=float, nargs=2, default=[0, 100], metavar=("ZMIN", "ZMAX"))
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--out", default="data/points/points.pkl", help="Output .pkl path")

    # NEW: TXT output
    ap.add_argument("--out_txt", default=None, help="Optional output .txt path. Default: same as --out with .txt suffix.")
    ap.add_argument("--txt_format", choices=["csv", "space"], default="csv", help="TXT delimiter style")
    ap.add_argument("--precision", type=int, default=2, help="Decimal places in TXT")

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    xs = rng.uniform(args.x[0], args.x[1], size=args.n)
    ys = rng.uniform(args.y[0], args.y[1], size=args.n)
    zs = rng.uniform(args.z[0], args.z[1], size=args.n)

    points = np.vstack([xs, ys, zs]).T.astype(float)

    out_pkl = Path(args.out)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as f:
        pickle.dump(points, f)

    out_txt = Path(args.out_txt) if args.out_txt else _default_txt_path(out_pkl)
    _save_txt(points, out_txt, fmt=args.txt_format, precision=args.precision)

    print("Saved PKL:", out_pkl)
    print("Saved TXT:", out_txt)
    print("Shape:", points.shape)
    print(
        "Ranges:",
        f"x[{points[:,0].min():.3f},{points[:,0].max():.3f}]",
        f"y[{points[:,1].min():.3f},{points[:,1].max():.3f}]",
        f"z[{points[:,2].min():.3f},{points[:,2].max():.3f}]",
    )


if __name__ == "__main__":
    main()
