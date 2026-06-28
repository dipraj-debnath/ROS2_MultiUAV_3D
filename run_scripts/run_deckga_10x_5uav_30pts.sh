#!/usr/bin/env bash
set -euo pipefail

TOTAL=10

for i in $(seq 1 $TOTAL); do
    echo "=== Running DECK_GA 30pts 5-UAV Run $i/10 ==="
    python3 DECK_GA.py \
        --points_pkl "data/points/antarctica_aspa135_m3_30_points.pkl" \
        --num_uavs 5 \
        --start_points="4.50,6.00,32.80;8.00,6.00,32.80;4.60,2.50,32.80;8.00,2.50,32.20;2.50,3.50,33.20" \
        --out_pkl "deckga_ros2/data/deckga_output_antarctica_30_uav5_ex${i}.pkl" \
        --save_fig_dir "results_antarctica"
    echo "=== Run $i complete ==="
done

echo "All $TOTAL runs finished."
