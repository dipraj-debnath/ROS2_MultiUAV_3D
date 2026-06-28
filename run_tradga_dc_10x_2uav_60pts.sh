#!/usr/bin/env bash
set -euo pipefail

TOTAL=10
RESULTS="results_csv/trad_ga_dc_2uavs_60_points_10_runs_results.txt"
START_POINTS="7.00,4.24,32.31;4.50,4.24,32.09"
POINTS_PKL="data/points/antarctica_aspa135_m3_60_points.pkl"
OUT_DIR="deckga_ros2/data"
FIG_DIR="results_antarctica"

mkdir -p results_csv "$OUT_DIR" "$FIG_DIR"

{
    echo "Traditional GA D&C | 2 UAVs | 60 Waypoints | 10 Runs Results"
    echo "antarctica_aspa135_m3 | start_points: 7.00,4.24,32.31 ; 4.50,4.24,32.09"
    echo "==========================================================================="
    echo ""
} > "$RESULTS"

for i in $(seq 1 $TOTAL); do
    echo "=== Running Traditional GA D&C 60pts 2-UAV Run $i/$TOTAL ==="

    OUT=$(python3 "DEGA_Divide & Conquer/Traditional_GA_Divide & Conquer.py" \
        --points_pkl "$POINTS_PKL" \
        --num_uavs 2 \
        --start_points="$START_POINTS" \
        --out_pkl "$OUT_DIR/traditional_ga_divide_conquer_output_antarctica_60_uav2_ex${i}.pkl" \
        --save_fig_dir "$FIG_DIR" \
        --population_size 50 \
        --num_iterations 25000 2>&1)

    echo "$OUT"

    {
        echo "=== Run $i / $TOTAL ==="
        echo "$OUT" | grep -E \
            "^--- (Raw Lengths|Optimized Lengths|Timing Summary)|^UAV [0-9]+: [0-9]|^Total (raw|optimized):|^Divide & Conquer Time :|^Traditional GA Time|^Total Time"
        echo ""
    } >> "$RESULTS"

    echo "=== Run $i complete ==="
done

python3 - >> "$RESULTS" << 'PY'
import pickle
import numpy as np
from pathlib import Path

out_dir = Path("deckga_ros2/data")
totals, ga_times, per_uav = [], [], {}

for i in range(1, 11):
    pkl_path = out_dir / f"traditional_ga_divide_conquer_output_antarctica_60_uav2_ex{i}.pkl"
    if not pkl_path.exists():
        print(f"WARNING: {pkl_path.name} not found — skipping.")
        continue
    with pkl_path.open("rb") as f:
        d = pickle.load(f)
    lengths = d.get("ga_lengths", d.get("final_lengths", []))
    totals.append(float(np.sum(lengths)))
    ga_times.append(float(d.get("ga_time_s", 0.0)))
    for j, l in enumerate(lengths):
        per_uav.setdefault(j, []).append(float(l))

print("===========================================================================")
print("AGGREGATE SUMMARY (10 runs)")
if totals:
    mn, mx, mean = min(totals), max(totals), float(np.mean(totals))
    if mx - mn < 0.001:
        print(f"Total optimized path length : {mean:.3f} (identical across all runs)")
    else:
        print(f"Total optimized path length : min={mn:.3f}  max={mx:.3f}  mean={mean:.3f}")
if ga_times:
    print(f"GA Time range               : {min(ga_times):.3f} s – {max(ga_times):.3f} s")
    print(f"GA Time mean                : {float(np.mean(ga_times)):.3f} s")
uav_means = "  ".join(f"UAV{j}={float(np.mean(v)):.3f}" for j, v in sorted(per_uav.items()))
print(f"Per-UAV mean optimized path : {uav_means}")
PY

echo ""
echo "Results saved to $RESULTS"
echo "All $TOTAL runs finished."
