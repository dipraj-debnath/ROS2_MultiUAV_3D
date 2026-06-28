#!/usr/bin/env bash
set -euo pipefail

TOTAL=10

for i in $(seq 1 $TOTAL); do
    echo "=== Running DECK_GA 60pts Run $i/10 ==="
    python3 DECK_GA.py \
        --points_pkl "data/points/antarctica_aspa135_m3_60_points.pkl" \
        --num_uavs 2 \
        --start_points="7.00,4.24,32.31;4.50,4.24,32.09" \
        --out_pkl "deckga_ros2/data/deckga_output_antarctica_60_uav2_ex${i}.pkl" \
        --save_fig_dir "results_antarctica"
    echo "=== Run $i complete ==="
done

echo "All $TOTAL runs finished."
