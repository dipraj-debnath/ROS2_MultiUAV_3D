#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_data_runs_trad_ga_dc_uav2_30pts.sh <start_run> <end_run>
# Example: ./run_data_runs_trad_ga_dc_uav2_30pts.sh 1 10
START_RUN=${1:?Usage: $0 <start_run> <end_run>}
END_RUN=${2:?Usage: $0 <start_run> <end_run>}

REPO="$HOME/Documents/GitHub/ROS2_MultiUAV_3D"
PROJECT_GAZEBO="$REPO/aerostack_examples/02_examples_gazebo_project/project_gazebo"
ROS2_SETUP="/opt/ros/humble/setup.bash"
AS2_SETUP="$HOME/as2_harmonic_ws/install/setup.bash"
ANTARCTICA_ENV="$PROJECT_GAZEBO/setup_antarctica_env.bash"

TMUX_SESSIONS=("drone0" "drone1")

set +u
source "$ROS2_SETUP"
source "$AS2_SETUP"
set -u

LIVE_POSE_TOPICS=(
    "/drone0/ground_truth/pose"
    "/drone1/ground_truth/pose"
)

REQUIRED_ACTIONS=(
    "/drone0/TakeoffBehavior"
    "/drone0/GoToBehavior"
    "/drone0/LandBehavior"
    "/drone0/FollowPathBehavior"
    "/drone1/TakeoffBehavior"
    "/drone1/GoToBehavior"
    "/drone1/LandBehavior"
    "/drone1/FollowPathBehavior"
)

wait_for_ready() {
    local timeout_s=240
    local interval=5
    local elapsed=0
    local echo_timeout=4
    echo "[wait_for_ready] Waiting for live poses + all actions (timeout ${timeout_s}s)..."
    while true; do
        local all_ok=true
        local reason=""

        for topic in "${LIVE_POSE_TOPICS[@]}"; do
            if ! timeout "$echo_timeout" ros2 topic echo --once "$topic" \
                    > /dev/null 2>&1; then
                all_ok=false
                reason="no live message on $topic"
                break
            fi
        done

        if $all_ok; then
            local action_list
            action_list=$(ros2 action list 2>/dev/null || true)
            for action in "${REQUIRED_ACTIONS[@]}"; do
                if ! grep -q "$action" <<< "$action_list"; then
                    all_ok=false
                    reason="action not yet available: $action"
                    break
                fi
            done
        fi

        if $all_ok; then
            echo "[wait_for_ready] Drones LIVE — poses flowing and all actions up (${elapsed}s elapsed)."
            return 0
        fi

        if (( elapsed >= timeout_s )); then
            echo "[WARN] Timed out after ${timeout_s}s. Last reason: $reason. Proceeding anyway."
            return 0
        fi

        sleep $interval
        elapsed=$(( elapsed + interval ))
        echo "  ... not ready yet [${reason}] (${elapsed}s elapsed)"
    done
}

wait_until_clean() {
    local timeout_s=30
    local interval=2
    local elapsed=0
    echo "[wait_until_clean] Waiting for all Gazebo/drone processes to die (up to ${timeout_s}s)..."
    while true; do
        local found=false
        if pgrep -u "$USER" -f "gz sim"            > /dev/null 2>&1; then found=true; fi
        if pgrep -u "$USER" -x "gzserver"          > /dev/null 2>&1; then found=true; fi
        if pgrep -u "$USER" -x "gzclient"          > /dev/null 2>&1; then found=true; fi
        if pgrep -u "$USER" -f "gazebo"            > /dev/null 2>&1; then found=true; fi
        if pgrep -u "$USER" -f "drone._interface"  > /dev/null 2>&1; then found=true; fi
        if pgrep -u "$USER" -f "micro.xrce\|MicroXRCE\|uxr_agent" > /dev/null 2>&1; then found=true; fi

        if ! $found; then
            echo "[wait_until_clean] Environment is clean (${elapsed}s elapsed)."
            return 0
        fi
        if (( elapsed >= timeout_s )); then
            echo "[WARN] Processes still alive after ${timeout_s}s — continuing anyway."
            return 0
        fi
        sleep $interval
        elapsed=$(( elapsed + interval ))
        echo "  ... still waiting for processes to die (${elapsed}s)"
    done
}

do_cleanup() {
    local n=$1
    echo "--- Cleanup for run $n ---"

    (cd "$PROJECT_GAZEBO" && ./stop.bash) 2>/dev/null || true

    for session in "${TMUX_SESSIONS[@]}"; do
        tmux kill-session -t "$session" 2>/dev/null || true
    done
    tmux kill-server 2>/dev/null || true

    pkill -u "$USER" -f "deckga_execute_antarctica.py"      2>/dev/null || true
    pkill -u "$USER" -f "rviz_paths_node_antarctica.py"     2>/dev/null || true
    pkill -u "$USER" -f "drone_pairwise_distance_logger.py" 2>/dev/null || true

    pkill -u "$USER" -f rviz2 2>/dev/null || true

    pkill -u "$USER" -f "gz sim"   2>/dev/null || true
    pkill -u "$USER" -x gzserver   2>/dev/null || true
    pkill -u "$USER" -x gzclient   2>/dev/null || true
    pkill -u "$USER" -f gazebo     2>/dev/null || true

    pkill -u "$USER" -f "as2_"         2>/dev/null || true
    pkill -u "$USER" -f "aerostack"    2>/dev/null || true

    pkill -u "$USER" -f "MicroXRCEAgent" 2>/dev/null || true
    pkill -u "$USER" -f "micro.xrce"     2>/dev/null || true
    pkill -u "$USER" -f "uxr_agent"      2>/dev/null || true

    ros2 daemon stop  2>/dev/null || true
    sleep 2
    ros2 daemon start 2>/dev/null || true
    rm -rf /tmp/ros*  2>/dev/null || true
    rm -rf /dev/shm/fastrtps_* /dev/shm/fastdds_* 2>/dev/null || true

    wait_until_clean

    echo "[cleanup] Settling 15s..."
    sleep 15
    echo "--- Cleanup done ---"
}

trap 'echo "[trap] Script exiting — running emergency cleanup..."; do_cleanup "trap"' EXIT INT TERM

cd "$REPO"

for N in $(seq "$START_RUN" "$END_RUN"); do

    echo ""
    echo "=========================================="
    echo "===== STARTING RUN $N ====="
    echo "=========================================="
    echo ""

    wait_until_clean

    # Step 1: Launch Gazebo + Aerostack2 (world_swarm_2.yaml is already correct)
    echo "[run $N] Launching Gazebo + Aerostack2 (2-UAV config)..."
    (
        set +eu
        source "$ROS2_SETUP"
        source "$AS2_SETUP"
        source "$ANTARCTICA_ENV"
        cd "$PROJECT_GAZEBO"
        ./launch_as2.bash -m
    ) &
    echo "[run $N] launch_as2.bash started in background. Sleeping 45s for tmux sessions to stabilise..."
    sleep 45

    # Step 2: Wait until drones are ready
    wait_for_ready

    # Step 3: Start pairwise distance logger
    echo "[run $N] Starting pairwise distance logger..."
    python3 deckga_ros2/drone_pairwise_distance_logger.py \
        --topics /drone0/ground_truth/pose,/drone1/ground_truth/pose \
        --names drone0,drone1 \
        --msg_type pose_stamped \
        --rate_hz 10 \
        --print_every_s 1.0 \
        --log_dir results_antarctica_csv \
        --run_tag "ex${N}_30pts_uav2_trad_ga_dc_pairwise_run_${N}" &
    LOGGER_PID=$!
    echo "[run $N] Pairwise logger started (PID $LOGGER_PID). Sleeping 3s..."
    sleep 3

    # Step 4: Run executor (foreground — blocks until mission done)
    echo "[run $N] Running executor (Traditional GA D&C, ex${N} PKL)..."
    set +e
    python3 deckga_ros2/deckga_execute_antarctica.py \
        --deckga_pkl "deckga_ros2/data/traditional_ga_divide_conquer_output_antarctica_30_uav2_ex${N}.pkl" \
        --namespaces drone0,drone1 \
        --coord_mode auto \
        --xy_scale 1.0 --xy_center_x 0.0 --xy_center_y 0.0 \
        --zin_min 32.09 --zin_max 39.98 \
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
        --run_tag "ex${N}_30pts_uav2_trad_ga_dc_run_${N}"
    EXEC_EXIT=$?
    set -e

    if (( EXEC_EXIT != 0 )); then
        echo ""
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "WARNING: RUN $N MAY HAVE FAILED — VERIFY"
        echo "Executor exited with code $EXEC_EXIT"
        echo "Check results_antarctica_csv/ for run $N CSVs."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo ""
    else
        echo "[run $N] Executor finished successfully (exit 0)."
    fi

    # Step 5: SIGINT pairwise logger to flush summary CSV
    echo "[run $N] Sending SIGINT to pairwise logger (PID $LOGGER_PID) to flush CSV..."
    kill -SIGINT "$LOGGER_PID" 2>/dev/null || true
    sleep 5

    # Step 6: Full cleanup
    do_cleanup "$N"

    echo ""
    echo "=========================================="
    echo "===== RUN $N COMPLETE ====="
    echo "=========================================="
    echo ""

done

echo "All runs from $START_RUN to $END_RUN complete."
