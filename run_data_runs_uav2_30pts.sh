#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_data_runs_uav2_30pts.sh <start_run> <end_run>
# Example: ./run_data_runs_uav2_30pts.sh 1 10
START_RUN=${1:?Usage: $0 <start_run> <end_run>}
END_RUN=${2:?Usage: $0 <start_run> <end_run>}

REPO="$HOME/Documents/GitHub/ROS2_MultiUAV_3D"
PROJECT_GAZEBO="$REPO/aerostack_examples/02_examples_gazebo_project/project_gazebo"
ROS2_SETUP="/opt/ros/humble/setup.bash"
AS2_SETUP="$HOME/as2_harmonic_ws/install/setup.bash"
ANTARCTICA_ENV="$PROJECT_GAZEBO/setup_antarctica_env.bash"

# Source ROS2 + AS2 into this shell so ros2 CLI works for action polling
# ROS2 setup scripts reference unset vars internally, so disable -u around them
# shellcheck disable=SC1090
set +u
source "$ROS2_SETUP"
source "$AS2_SETUP"
set -u

REQUIRED_ACTIONS=(
    "/drone0/TakeoffBehavior"
    "/drone0/GoToBehavior"
    "/drone0/LandBehavior"
    "/drone1/TakeoffBehavior"
    "/drone1/GoToBehavior"
    "/drone1/LandBehavior"
)

# Poll ros2 action list until all required actions appear or timeout
wait_for_actions() {
    local timeout_s=240
    local interval=5
    local elapsed=0
    echo "[wait_for_actions] Polling for drone actions (timeout ${timeout_s}s)..."
    while true; do
        local action_list
        action_list=$(ros2 action list 2>/dev/null || true)
        local all_ok=true
        for action in "${REQUIRED_ACTIONS[@]}"; do
            if ! grep -q "$action" <<< "$action_list"; then
                all_ok=false
                break
            fi
        done
        if $all_ok; then
            echo "[wait_for_actions] All actions available (${elapsed}s elapsed)."
            return 0
        fi
        if (( elapsed >= timeout_s )); then
            echo "[WARN] Timed out waiting for actions after ${timeout_s}s. Proceeding anyway."
            return 0
        fi
        sleep $interval
        elapsed=$(( elapsed + interval ))
        echo "  ... still waiting (${elapsed}s elapsed)"
    done
}

# Terminal 10 full cleanup
do_cleanup() {
    local n=$1
    echo "--- Cleanup (Terminal 10) for run $n ---"
    (cd "$PROJECT_GAZEBO" && ./stop.bash) 2>/dev/null || true
    pkill -u "$USER" -f "deckga_execute_antarctica.py"    2>/dev/null || true
    pkill -u "$USER" -f "rviz_paths_node_antarctica.py"   2>/dev/null || true
    pkill -u "$USER" -f "drone_pairwise_distance_logger.py" 2>/dev/null || true
    pkill -u "$USER" -f rviz2                             2>/dev/null || true
    pkill -u "$USER" -f "gz sim"                          2>/dev/null || true
    pkill -u "$USER" -f gzserver                          2>/dev/null || true
    pkill -u "$USER" -f gzclient                          2>/dev/null || true
    ros2 daemon stop  2>/dev/null || true
    sleep 3
    ros2 daemon start 2>/dev/null || true
    rm -rf /tmp/ros*  2>/dev/null || true
    echo "--- Cleanup done ---"
}

# Kill Gazebo and the pairwise logger if the script exits for any reason
trap 'echo "[trap] Script exiting — running emergency cleanup..."; do_cleanup "trap"' EXIT INT TERM

cd "$REPO"

for N in $(seq "$START_RUN" "$END_RUN"); do

    echo ""
    echo "=========================================="
    echo "===== STARTING RUN $N ====="
    echo "=========================================="
    echo ""

    # ------------------------------------------------------------------
    # Step 1: Launch Gazebo + Aerostack2 in background
    # ------------------------------------------------------------------
    echo "[run $N] Launching Gazebo + Aerostack2..."
    (
        set +eu  # tmux attach-session will fail in non-interactive shell — that is OK
                 # also disable -u so ROS2 setup scripts don't trip on unbound vars
        # shellcheck disable=SC1090
        source "$ROS2_SETUP"
        source "$AS2_SETUP"
        source "$ANTARCTICA_ENV"
        cd "$PROJECT_GAZEBO"
        ./launch_as2.bash -m
    ) &
    echo "[run $N] launch_as2.bash started in background. Sleeping 30s for tmux sessions to stabilise..."
    sleep 30

    # ------------------------------------------------------------------
    # Step 2: Wait until drone actions are advertised
    # ------------------------------------------------------------------
    wait_for_actions

    # ------------------------------------------------------------------
    # Step 3: Start pairwise distance logger in background
    # ------------------------------------------------------------------
    echo "[run $N] Starting pairwise distance logger..."
    python3 deckga_ros2/drone_pairwise_distance_logger.py \
        --topics /drone0/ground_truth/pose,/drone1/ground_truth/pose \
        --names drone0,drone1 \
        --msg_type pose_stamped \
        --rate_hz 10 \
        --print_every_s 1.0 \
        --log_dir results_antarctica_csv \
        --run_tag "ex${N}_30pts_uav2_pairwise_distance_run_${N}" &
    LOGGER_PID=$!
    echo "[run $N] Pairwise logger started (PID $LOGGER_PID). Sleeping 3s..."
    sleep 3

    # ------------------------------------------------------------------
    # Step 4: Run DECK-GA executor (foreground — blocks until mission done)
    # ------------------------------------------------------------------
    echo "[run $N] Running DECK-GA executor (ex${N} PKL)..."
    python3 deckga_ros2/deckga_execute_antarctica.py \
        --deckga_pkl "deckga_ros2/data/deckga_output_antarctica_30_uav2_ex${N}.pkl" \
        --namespaces drone0,drone1 \
        --coord_mode auto \
        --xy_scale 1.0 --xy_center_x 0.0 --xy_center_y 0.0 \
        --zin_min 32.34 --zin_max 41.30 \
        --zout_min 3.80 --zout_max 6.80 \
        --z_base 32.2 \
        --clamp_xy --x_min -40 --x_max 40 --y_min -40 --y_max 40 \
        --takeoff_margin 0.10 \
        --pre_arm_wait_s 5 \
        --speed 1.9 \
        --wait_actions_s 10 \
        --init_wait_s 5 \
        --takeoff_settle_s 1 \
        --takeoff_sequential \
        --first_wp_wait \
        --ensure_reach \
        --wp_settle_s 0.3 \
        --log_dir results_antarctica_csv \
        --run_tag "ex${N}_30pts_uav2_deckga_run_${N}"
    echo "[run $N] Executor finished."

    # ------------------------------------------------------------------
    # Step 5: SIGINT pairwise logger so it flushes its summary CSV
    # ------------------------------------------------------------------
    echo "[run $N] Sending SIGINT to pairwise logger (PID $LOGGER_PID) to flush CSV..."
    kill -SIGINT "$LOGGER_PID" 2>/dev/null || true
    sleep 5   # Give logger time to write files before cleanup kills it

    # ------------------------------------------------------------------
    # Step 6: Full Terminal 10 cleanup — kill Gazebo, rviz, reset daemon
    # ------------------------------------------------------------------
    do_cleanup "$N"

    # ------------------------------------------------------------------
    # Step 7: Settle before next run
    # ------------------------------------------------------------------
    echo "[run $N] Settling 10s before next run..."
    sleep 10

    echo ""
    echo "=========================================="
    echo "===== RUN $N COMPLETE ====="
    echo "=========================================="
    echo ""

done

echo "All runs from $START_RUN to $END_RUN complete."
