#!/usr/bin/env python3
"""
Offline DECK_GA runner:
- (Optionally) generates random points (any quadrant)
- Runs DECK_GA and writes deckga_ros2/data/deckga_output.pkl

This does NOT launch Aerostack. After this runs:
- Launch Aerostack (Terminal 1/2)
- Execute: python3 deckga_ros2/deckga_execute.py
"""

import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_uavs", type=int, default=3)
    ap.add_argument("--n_points", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--x", type=float, nargs=2, default=[-100, 100])
    ap.add_argument("--y", type=float, nargs=2, default=[-100, 100])
    ap.add_argument("--z", type=float, nargs=2, default=[0, 50])

    ap.add_argument("--points_pkl", default=str(REPO_ROOT / "data" / "points" / "points_current.pkl"))
    ap.add_argument("--no_generate", action="store_true", help="Skip point generation and use existing points_pkl")
    args = ap.parse_args()

    points_pkl = Path(args.points_pkl)
    points_pkl.parent.mkdir(parents=True, exist_ok=True)

    if not args.no_generate:
        cmd_gen = [
            "python3",
            str(REPO_ROOT / "data" / "points" / "generate_points_xyz.py"),
            "--n", str(args.n_points),
            "--x", str(args.x[0]), str(args.x[1]),
            "--y", str(args.y[0]), str(args.y[1]),
            "--z", str(args.z[0]), str(args.z[1]),
            "--seed", str(args.seed),
            "--out", str(points_pkl),
        ]
        print("Running:", " ".join(cmd_gen))
        subprocess.check_call(cmd_gen)

    cmd_plan = [
        "python3",
        str(REPO_ROOT / "deckga_ros2" / "generate_deckga_output.py"),
        "--points_pkl", str(points_pkl),
        "--num_uavs", str(args.num_uavs),
    ]
    print("Running:", " ".join(cmd_plan))
    subprocess.check_call(cmd_plan)

    print("Done. Output written to deckga_ros2/data/deckga_output.pkl")


if __name__ == "__main__":
    main()
