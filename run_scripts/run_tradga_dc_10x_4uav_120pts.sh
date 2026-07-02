#!/usr/bin/env bash
set -euo pipefail
REPO="$HOME/Documents/GitHub/ROS2_MultiUAV_3D"
cd "$REPO"
for i in $(seq 1 10); do
    python3 "DEGA_Divide & Conquer/Traditional_GA_Divide & Conquer.py" \
        --points_pkl "data/points/antarctica_aspa135_m3_120_points.pkl" \
        --num_uavs 4 \
        --start_points="7.00,8.00,32.09;4.00,4.00,32.50;4.00,8.00,32.31;7.00,4.00,32.50" \
        --out_pkl "deckga_ros2/data/traditional_ga_divide_conquer_output_antarctica_120_uav4_ex${i}.pkl" \
        --save_fig_dir "results_antarctica" \
        --population_size 50 \
        --num_iterations 25000
done
