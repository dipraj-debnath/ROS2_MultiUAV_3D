#!/usr/bin/env bash
set -euo pipefail

TOTAL=10

for i in $(seq 1 $TOTAL); do
    echo "=== Running DECK_GA 120pts 3-UAV Run $i/10 ==="
    python3 DECK_GA.py \
        --points_pkl "data/points/antarctica_aspa135_m3_120_points.pkl" \
        --num_uavs 3 \
        --start_points="7.00,6.00,32.09;5.50,4.00,32.20;4.00,8.00,32.31" \
        --out_pkl "deckga_ros2/data/deckga_output_antarctica_120_uav3_ex${i}.pkl" \
        --save_fig_dir "results_antarctica"
    echo "=== Run $i complete ==="
done

echo "All $TOTAL runs finished."
