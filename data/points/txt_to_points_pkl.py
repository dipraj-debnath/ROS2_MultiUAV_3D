#!/usr/bin/env python3
import argparse
import numpy as np
import pickle
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_txt", required=True, help="Input txt with columns: x y z (header allowed)")
    ap.add_argument("--out_pkl", required=True, help="Output pkl path to store Nx3 float array")
    args = ap.parse_args()

    in_path = Path(args.in_txt)
    out_path = Path(args.out_pkl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load: allow header, allow whitespace, ignore blank lines
    rows = []
    with in_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("x"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) != 3:
                raise ValueError(f"Bad line (expected 3 values): {line}")
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])

    pts = np.array(rows, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Bad shape: {pts.shape}, expected (N,3)")

    with out_path.open("wb") as f:
        pickle.dump(pts, f)

    print(f"Saved: {out_path}")
    print(f"Shape: {pts.shape}")
    print(f"Ranges: x[{pts[:,0].min():.3f},{pts[:,0].max():.3f}] "
          f"y[{pts[:,1].min():.3f},{pts[:,1].max():.3f}] "
          f"z[{pts[:,2].min():.3f},{pts[:,2].max():.3f}]")

if __name__ == "__main__":
    main()
