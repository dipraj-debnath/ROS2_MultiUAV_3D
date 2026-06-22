#!/usr/bin/env python3
import argparse
import csv
import glob
import math
import os

parser = argparse.ArgumentParser(
    description="Build a summary table for a DECK-GA experiment configuration."
)
parser.add_argument("waypoints", type=int, help="Number of waypoints (e.g. 30)")
parser.add_argument("uavs",      type=int, help="Number of UAVs (e.g. 2)")
args = parser.parse_args()

WPTS = args.waypoints
UAVS = args.uavs
NUM_RUNS    = 10
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_antarctica_csv")
OUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_csv")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE    = os.path.join(OUT_DIR, f"summary_table_uav{UAVS}_{WPTS}pts.txt")

DECKGA_COLS  = ["planned_makespan_s", "mission_makespan_s",
                "planned_total_dist_m", "executed_total_dist_m"]
PAIRWISE_COL = "overall_min_pairwise_m"
ALL_COLS     = ["Run"] + DECKGA_COLS + [PAIRWISE_COL]


def find_csv(pattern):
    matches = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
    return matches[-1] if matches else None  # latest timestamp wins


def read_cell(filepath, col):
    if filepath is None:
        return None
    with open(filepath, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    return rows[-1].get(col)


def fmt(val):
    if val is None:
        return "MISSING"
    try:
        return f"{float(val):.3f}"
    except (ValueError, TypeError):
        return "MISSING"


rows = []
for n in range(1, NUM_RUNS + 1):
    deckga_file = find_csv(
        f"*ex{n}_{WPTS}pts_uav{UAVS}_deckga_run_{n}*summary.csv"
    )
    pairwise_file = find_csv(
        f"*ex{n}_{WPTS}pts_uav{UAVS}_pairwise_distance_run_{n}*summary.csv"
    )

    row = {"Run": str(n)}
    for col in DECKGA_COLS:
        row[col] = read_cell(deckga_file, col)
    row[PAIRWISE_COL] = read_cell(pairwise_file, PAIRWISE_COL)
    rows.append(row)


def stats(col):
    vals = []
    for r in rows:
        v = r.get(col)
        if v is not None:
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                pass
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    std  = math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))
    return mean, std


col_widths = {}
for col in ALL_COLS:
    col_widths[col] = max(
        len(col),
        max(len(fmt(r.get(col)) if col != "Run" else r["Run"]) for r in rows),
        10,
    )


def make_row(cells):
    return "  ".join(str(v).rjust(col_widths[c]) for c, v in zip(ALL_COLS, cells))


header  = make_row(ALL_COLS)
divider = "  ".join("-" * col_widths[c] for c in ALL_COLS)

lines = [header, divider]
for r in rows:
    cells = [r["Run"]] + [fmt(r.get(c)) for c in DECKGA_COLS] + [fmt(r.get(PAIRWISE_COL))]
    lines.append(make_row(cells))

lines.append("")
mean_cells = ["mean"]
std_cells  = ["std"]
for col in DECKGA_COLS + [PAIRWISE_COL]:
    mean, std = stats(col)
    mean_cells.append(fmt(mean))
    std_cells.append(fmt(std))
lines.append(make_row(mean_cells))
lines.append(make_row(std_cells))

table = "\n".join(lines)

print(table)

with open(OUT_FILE, "w") as fh:
    fh.write(table + "\n")

print(f"\nSaved to: {OUT_FILE}")
