#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_data_runs_classical_kmeans_dega_uav4_90pts.sh <start_run> <end_run>
START_RUN=${1:?Usage: $0 <start_run> <end_run>}
END_RUN=${2:?Usage: $0 <start_run> <end_run>}

REPO="$HOME/Documents/GitHub/ROS2_MultiUAV_3D"
PROJECT_GAZEBO="$REPO/aerostack_examples/02_examples_gazebo_project/project_gazebo"
ROS2_SETUP="/opt/ros/humble/setup.bash"
AS2_SETUP="$HOME/as2_harmonic_ws/install/setup.bash"
ANTARCTICA_ENV="$PROJECT_GAZEBO/setup_antarctica_env.bash"

TMUX_SESSIONS=("drone0" "drone1" "drone2" "drone3")

set +u
source "$ROS2_SETUP"
source "$AS2_SETUP"
set -u

LIVE_POSE_TOPICS=(
    "/drone0/ground_truth/pose"
    "/drone1/ground_truth/pose"
    "/drone2/ground_truth/pose"
    "/drone3/ground_truth/pose"
)

REQUIRED_ACTIONS=(
    "/drone0/TakeoffBehavior" "/drone0/GoToBehavior" "/drone0/LandBehavior" "/drone0/FollowPathBehavior"
    "/drone1/TakeoffBehavior" "/drone1/GoToBehavior" "/drone1/LandBehavior" "/drone1/FollowPathBehavior"
    "/drone2/TakeoffBehavior" "/drone2/GoToBehavior" "/drone2/LandBehavior" "/drone2/FollowPathBehavior"
    "/drone3/TakeoffBehavior" "/drone3/GoToBehavior" "/drone3/LandBehavior" "/drone3/FollowPathBehavior"
)

wait_for_ready() {
    local timeout_s=240 interval=5 elapsed=0 echo_timeout=4
    echo "[wait_for_ready] Waiting for live poses + all actions (timeout ${timeout_s}s)..."
    while true; do
        local all_ok=true reason=""
        for topic in "${LIVE_POSE_TOPICS[@]}"; do
            if ! timeout "$echo_timeout" ros2 topic echo --once "$topic" > /dev/null 2>&1; then
                all_ok=false; reason="no live message on $topic"; break
            fi
        done
        if $all_ok; then
            local action_list
            action_list=$(ros2 action list 2>/dev/null || true)
            for action in "${REQUIRED_ACTIONS[@]}"; do
                if ! grep -q "$action" <<< "$action_list"; then
                    all_ok=false; reason="action not yet available: $action"; break
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
        sleep $interval; elapsed=$(( elapsed + interval ))
        echo "  ... not ready yet [${reason}] (${elapsed}s elapsed)"
    done
}

wait_until_clean() {
    local timeout_s=30 interval=2 elapsed=0
    echo "[wait_until_clean] Waiting for all Gazebo/drone processes to die (up to ${timeout_s}s)..."
    while true; do
        local found=false
        pgrep -u "$USER" -f "gz sim"           > /dev/null 2>&1 && found=true
        pgrep -u "$USER" -x "gzserver"         > /dev/null 2>&1 && found=true
        pgrep -u "$USER" -x "gzclient"         > /dev/null 2>&1 && found=true
        pgrep -u "$USER" -f "gazebo"           > /dev/null 2>&1 && found=true
        pgrep -u "$USER" -f "drone._interface" > /dev/null 2>&1 && found=true
        pgrep -u "$USER" -f "micro.xrce\|MicroXRCE\|uxr_agent" > /dev/null 2>&1 && found=true
        if ! $found; then
            echo "[wait_until_clean] Environment is clean (${elapsed}s elapsed)."; return 0
        fi
        if (( elapsed >= timeout_s )); then
            echo "[WARN] Processes still alive after ${timeout_s}s — continuing anyway."; return 0
        fi
        sleep $interval; elapsed=$(( elapsed + interval ))
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
    pkill -u "$USER" -f rviz2      2>/dev/null || true
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

    echo "[run $N] Writing 4-UAV world config..."
    cat > "$PROJECT_GAZEBO/config/world_swarm_4.yaml" << 'YAML'
world_name: "aspa135_m3"
origin:
    latitude: -66.28223056
    longitude: 110.53892500
    altitude: 30.19

drones:
  - model_type: "quadrotor_base"
    model_name: "drone0"
    flight_time: 60
    xyz:
      - 4.00
      - 4.00
      - 32.50
    payload:
      - model_type: "gps"
        model_name: "gps"

  - model_type: "quadrotor_base"
    model_name: "drone1"
    flight_time: 60
    xyz:
      - 7.00
      - 8.00
      - 32.09
    payload:
      - model_type: "gps"
        model_name: "gps"

  - model_type: "quadrotor_base"
    model_name: "drone2"
    flight_time: 60
    xyz:
      - 4.00
      - 8.00
      - 32.31
    payload:
      - model_type: "gps"
        model_name: "gps"

  - model_type: "quadrotor_base"
    model_name: "drone3"
    flight_time: 60
    xyz:
      - 7.00
      - 4.00
      - 32.00
    payload:
      - model_type: "gps"
        model_name: "gps"
YAML
    cp "$PROJECT_GAZEBO/config/world_swarm_4.yaml" "$PROJECT_GAZEBO/config/world_swarm.yaml"
    grep "model_name" "$PROJECT_GAZEBO/config/world_swarm.yaml"

    echo "[run $N] Launching Gazebo + Aerostack2 (4-UAV config)..."
    (
        set +eu
        source "$ROS2_SETUP"
        source "$AS2_SETUP"
        source "$ANTARCTICA_ENV"
        cd "$PROJECT_GAZEBO"
        ./launch_as2.bash -m
    ) &
    echo "[run $N] Sleeping 45s for boot..."
    sleep 45

    wait_for_ready

    echo "[run $N] Starting pairwise distance logger..."
    python3 deckga_ros2/drone_pairwise_distance_logger.py \
        --topics /drone0/ground_truth/pose,/drone1/ground_truth/pose,/drone2/ground_truth/pose,/drone3/ground_truth/pose \
        --names drone0,drone1,drone2,drone3 \
        --msg_type pose_stamped \
        --rate_hz 10 \
        --print_every_s 1.0 \
        --log_dir results_antarctica_csv \
        --run_tag "ex${N}_90pts_uav4_classical_kmeans_dega_pairwise_run_${N}" &
    LOGGER_PID=$!
    echo "[run $N] Logger PID $LOGGER_PID. Sleeping 3s..."
    sleep 3

    echo "[run $N] Running executor..."
    set +e
    python3 deckga_ros2/deckga_execute_antarctica.py \
        --deckga_pkl "deckga_ros2/data/classical_kmeans_dega_output_antarctica_90_uav4_ex${N}.pkl" \
        --namespaces drone0,drone1,drone2,drone3 \
        --coord_mode auto \
        --xy_scale 1.0 --xy_center_x 0.0 --xy_center_y 0.0 \
        --zin_min 32.09 --zin_max 39.99 \
        --zout_min 3.80 --zout_max 6.80 \
        --z_base 32.00 \
        --clamp_xy --x_min -40 --x_max 40 --y_min -40 --y_max 40 \
        --takeoff_margin 0.10 \
        --pre_arm_wait_s 5 \
        --speed 1.9 \
        --wait_actions_s 10 \
        --init_wait_s 5 \
        --takeoff_settle_s 3 \
        --takeoff_sequential \
        --first_wp_wait \
        --ensure_reach \
        --wp_settle_s 0.3 \
        --log_dir results_antarctica_csv \
        --run_tag "ex${N}_90pts_uav4_classical_kmeans_dega_run_${N}"
    EXEC_EXIT=$?
    set -e

    if (( EXEC_EXIT != 0 )); then
        echo "WARNING: RUN $N MAY HAVE FAILED — executor exited with code $EXEC_EXIT"
    else
        echo "[run $N] Executor finished successfully."
    fi

    kill -SIGINT "$LOGGER_PID" 2>/dev/null || true
    sleep 5

    do_cleanup "$N"

    echo ""
    echo "===== RUN $N COMPLETE ====="
    echo ""

done

echo "All runs from $START_RUN to $END_RUN complete."
