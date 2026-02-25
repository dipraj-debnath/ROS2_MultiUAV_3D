#!/usr/bin/env python3
"""
generate_points_xyz.py

Generate Nx3 random points and save:
  (A) RAW (unscaled) points  -> PKL + TXT
  (B) SCALED points          -> PKL + TXT   (optional; enabled when scale != 1)

This lets you:
- fly/visualize using the scaled dataset everywhere (planner + RViz + execute)
- still keep raw dataset for reporting "original coordinate" stats if needed

Scaling:
  x,y scaled by --scale_xy
  z   scaled by --scale_z

Outputs:
- --out (PKL)           : SCALED if scale != 1 else RAW
- --out_txt (TXT)       : same dataset as --out
- --out_raw_pkl (PKL)   : RAW (always written if provided; auto if scale != 1)
- --out_raw_txt (TXT)   : RAW (always written if provided; auto if scale != 1)

TXT formats:
- csv: header "x,y,z" then rows
- space: "x y z" rows (no header)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _default_txt_path(pkl_path: Path) -> Path:
    return pkl_path.with_suffix(".txt")


def _suffix_path(p: Path, suffix: str) -> Path:
    # points_seed11_n30_z10_100.pkl -> points_seed11_n30_z10_100_raw.pkl
    return p.with_name(p.stem + suffix + p.suffix)


def _save_pkl(points: np.ndarray, out_pkl: Path) -> None:
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as f:
        pickle.dump(points, f)


def _save_txt(points: np.ndarray, out_txt: Path, fmt: str = "csv", precision: int = 2) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points, dtype=float)

    if fmt == "csv":
        header = "x,y,z"
        np.savetxt(out_txt, pts, delimiter=",", header=header, comments="", fmt=f"%.{precision}f")
    elif fmt == "space":
        np.savetxt(out_txt, pts, delimiter=" ", fmt=f"%.{precision}f")
    else:
        raise ValueError("txt_format must be one of: csv, space")


def _ranges(points: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    p = np.asarray(points, dtype=float)
    return float(p[:, 0].min()), float(p[:, 0].max()), float(p[:, 1].min()), float(p[:, 1].max()), float(p[:, 2].min()), float(p[:, 2].max())


def main() -> None:
    ap = argparse.ArgumentParser(allow_abbrev=False)

    ap.add_argument("--n", type=int, required=True, help="Number of points")
    ap.add_argument("--x", type=float, nargs=2, required=True, metavar=("XMIN", "XMAX"))
    ap.add_argument("--y", type=float, nargs=2, required=True, metavar=("YMIN", "YMAX"))
    ap.add_argument("--z", type=float, nargs=2, required=True, metavar=("ZMIN", "ZMAX"))
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--out", required=True, help="Output PKL path. Will store SCALED if scale != 1 else RAW.")
    ap.add_argument("--out_txt", default=None, help="Optional output TXT path for same dataset as --out.")
    ap.add_argument("--txt_format", default="csv", choices=["csv", "space"])
    ap.add_argument("--precision", type=int, default=2)

    # NEW: scaling applied at generation time
    ap.add_argument("--scale_xy", type=float, default=1.0, help="Scale factor for x,y (default 1.0)")
    ap.add_argument("--scale_z", type=float, default=1.0, help="Scale factor for z (default 1.0)")

    # NEW: keep raw outputs too (recommended if scaling != 1)
    ap.add_argument("--out_raw_pkl", default=None, help="Optional RAW (unscaled) PKL path.")
    ap.add_argument("--out_raw_txt", default=None, help="Optional RAW (unscaled) TXT path.")

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    n = int(args.n)
    xmin, xmax = float(args.x[0]), float(args.x[1])
    ymin, ymax = float(args.y[0]), float(args.y[1])
    zmin, zmax = float(args.z[0]), float(args.z[1])

    points_raw = np.column_stack([
        rng.uniform(xmin, xmax, size=n),
        rng.uniform(ymin, ymax, size=n),
        rng.uniform(zmin, zmax, size=n),
    ]).astype(float)

    scale_xy = float(args.scale_xy)
    scale_z = float(args.scale_z)

    do_scale = (abs(scale_xy - 1.0) > 1e-12) or (abs(scale_z - 1.0) > 1e-12)

    out_pkl = Path(args.out).expanduser()
    out_txt = Path(args.out_txt).expanduser() if args.out_txt else _default_txt_path(out_pkl)

    # Auto raw outputs if scaling is enabled and user did not specify
    out_raw_pkl: Optional[Path] = Path(args.out_raw_pkl).expanduser() if args.out_raw_pkl else None
    out_raw_txt: Optional[Path] = Path(args.out_raw_txt).expanduser() if args.out_raw_txt else None

    if do_scale:
        if out_raw_pkl is None:
            out_raw_pkl = _suffix_path(out_pkl, "_raw")
        if out_raw_txt is None:
            out_raw_txt = _default_txt_path(out_raw_pkl)

    # Save RAW (if requested / auto-enabled)
    if out_raw_pkl is not None:
        _save_pkl(points_raw, out_raw_pkl)
        _save_txt(points_raw, out_raw_txt if out_raw_txt else _default_txt_path(out_raw_pkl),
                  fmt=args.txt_format, precision=args.precision)

    # Prepare dataset for --out (scaled if enabled else raw)
    if do_scale:
        points_out = points_raw.copy()
        points_out[:, 0] *= scale_xy
        points_out[:, 1] *= scale_xy
        points_out[:, 2] *= scale_z
    else:
        points_out = points_raw

    _save_pkl(points_out, out_pkl)
    _save_txt(points_out, out_txt, fmt=args.txt_format, precision=args.precision)

    # Prints
    print("Saved PKL:", str(out_pkl))
    print("Saved TXT:", str(out_txt))
    if out_raw_pkl is not None:
        print("Saved RAW PKL:", str(out_raw_pkl))
        if out_raw_txt is not None:
            print("Saved RAW TXT:", str(out_raw_txt))

    print("Shape:", points_out.shape)
    rx0, rx1, ry0, ry1, rz0, rz1 = _ranges(points_out)
    print(f"Ranges: x[{rx0:.3f},{rx1:.3f}] y[{ry0:.3f},{ry1:.3f}] z[{rz0:.3f},{rz1:.3f}]")

    if do_scale:
        print(f"Scale applied: scale_xy={scale_xy} scale_z={scale_z}")
        # Helpful for later: you can convert scaled distance back to original by /scale_xy (if uniform)
        if abs(scale_xy - scale_z) < 1e-12:
            print(f"Uniform scale => original_distance ≈ scaled_distance / {scale_xy}")


if __name__ == "__main__":
    main()