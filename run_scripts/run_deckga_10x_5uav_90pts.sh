#!/usr/bin/env bash
set -euo pipefail

TOTAL=10

for i in $(seq 1 $TOTAL); do
    echo "=== Running DECK_GA 90pts 5-UAV Run $i/10 ==="
    python3 DECK_GA.py \
        --points_pkl "data/points/antarctica_aspa135_m3_90_points.pkl" \
        --num_uavs 5 \
        --start_points="4.50,2.50,32.90;8.00,8.00,32.70;4.50,8.00,32.80;8.00,4.50,32.60;6.00,5.00,32.99" \
        --out_pkl "deckga_ros2/data/deckga_output_antarctica_90_uav5_ex${i}.pkl" \
        --save_fig_dir "results_antarctica"
    echo "=== Run $i complete ==="
done

echo "All $TOTAL runs finished."
