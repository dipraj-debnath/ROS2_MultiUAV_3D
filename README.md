# ROS2_MultiUAV_3D

Multi-UAV 3D mission planning and execution in **ROS2 Humble** / **Aerostack2** / **Gazebo** using **DECK-GA**
(**DCKmeans + Distance Efficient Genetic Algorithm (DEGA)** path planning).

This repository supports:
- Offline planning (DECK-GA and baselines)
- ROS2/Aerostack2 execution in Gazebo
- RViz visualization of planned paths
- Antarctica world execution workflow (ASPA135 terrain models)

---

## Repository Structure

- `aerostack_examples/`  
  Aerostack2 / Gazebo project launch files and configs (multi-UAV Gazebo project)

- `deckga_ros2/`  
  ROS2/Aerostack-facing Python scripts:
  - `rviz_paths_node.py` : publishes planned paths to RViz markers (`/deckga/markers`)
  - `deckga_execute.py` : executes the planned path through Aerostack2 actions
  - `data/` : planner output inputs for ROS2 execution (e.g., `deckga_output.pkl`)

- Offline planning core:
  - `DECK_GA.py`, `DCKmeans.py`, `GA_path_planning.py`

- Baselines (added):
  - `DEGA_Divide & Conquer/`
  - `DEGA_Load Balancing/`

- `data/points/`  
  Waypoint generator + datasets (`.pkl`) and conversion utilities

- `results/`  
  Planner-generated figures/logs (created at runtime)

---

## Prerequisites

You need:

- **Ubuntu 22.04**
- **ROS 2 Humble** installed
- **Aerostack2** installed and built in an overlay workspace (recommended path: `~/as2_harmonic_ws`)
  - Official documentation and installation guide: https://aerostack2.github.io/index.html
  - Typical workflow:
    ```bash
    mkdir -p ~/as2_harmonic_ws/src
    cd ~/as2_harmonic_ws/src
    # Follow Aerostack2 docs for the correct repository + dependencies
    # then build:
    cd ~/as2_harmonic_ws
    source /opt/ros/humble/setup.bash
    colcon build --symlink-install
    ```
- **Gazebo** (as required by the included Aerostack2 example project)
- **Python 3** + common scientific packages (e.g., `numpy`)

### Git LFS (required for Antarctica assets)
This repo uses **Git LFS** for large Gazebo model assets (e.g., terrain meshes/maps).

After cloning:
```bash
sudo apt update
sudo apt install git-lfs
git lfs install
git lfs pull
Environment Setup (IMPORTANT)
For all terminals in this repo, use:

source /opt/ros/humble/setup.bash
source ~/as2_harmonic_ws/install/setup.bash
If you are running the Antarctica project under project_gazebo, you may also need:

source ./setup_antarctica_env.bash
(from inside the project_gazebo directory)

Coordinate Transform (Important)
Both RViz visualization and Aerostack execution apply the same transform (defaults inside both scripts):

coord_mode = original

scale_xy = 0.05

scale_z = 0.05

z_offset = 2.5

z_min = 3.0 (safety clamp)

Meaning: the planner can operate in a larger coordinate range, while the executed/visualised path is scaled down
for a safe Gazebo/Aerostack2 flight envelope.

For the Antarctica world workflow, coordinate mapping parameters may differ.
See ANTARCTICA_EXPERIMENT.md.

Quickstart: Run the Full Experiment (Generic World)
This section runs the full pipeline using randomly generated 3D points (offline planning + Gazebo + RViz + execution).

Terminal 0 — Generate points + Run DECK_GA
cd ~/Documents/GitHub/ROS2_MultiUAV_3D
source /opt/ros/humble/setup.bash
source ~/as2_harmonic_ws/install/setup.bash

SEED=11
N=30
DATASET="data/points/points_seed${SEED}_n${N}_z10_100.pkl"

python3 data/points/generate_points_xyz.py \
  --n ${N} \
  --x -100 100 \
  --y -100 100 \
  --z 10 100 \
  --seed ${SEED} \
  --out "${DATASET}"

python3 DECK_GA.py \
  --points_pkl "${DATASET}" \
  --num_uavs 3 \
  --start_points="-40,0,10;40,0,10;0,0,10" \
  --out_pkl deckga_ros2/data/deckga_output.pkl \
  --save_fig_dir results
This produces:

Waypoints dataset: data/points/points_seed11_n30_z10_100.pkl

Planner output: deckga_ros2/data/deckga_output.pkl

Figures/logs: results/

Terminal 1 — Start Gazebo + Aerostack2 (multirotor)
cd ~/Documents/GitHub/ROS2_MultiUAV_3D/aerostack_examples/02_examples_gazebo_project/project_gazebo
source /opt/ros/humble/setup.bash
source ~/as2_harmonic_ws/install/setup.bash

./launch_as2.bash -m
Leave it running.

Terminal 2 — Start Ground Station + RViz (+ keyboard teleop)
cd ~/Documents/GitHub/ROS2_MultiUAV_3D/aerostack_examples/02_examples_gazebo_project/project_gazebo
source /opt/ros/humble/setup.bash
source ~/as2_harmonic_ws/install/setup.bash

./launch_ground_station.bash -m -t -v
Leave it running (RViz should open).

Terminal 3 — Quick verification (nodes + marker topic)
Run this before Terminal 4, then repeat after Terminal 4 to confirm /deckga/markers appears.

cd ~/Documents/GitHub/ROS2_MultiUAV_3D/aerostack_examples/02_examples_gazebo_project/project_gazebo
source /opt/ros/humble/setup.bash
source ~/as2_harmonic_ws/install/setup.bash

echo "=== Nodes (filtered) ==="
ros2 node list | egrep "drone|ground|rviz|gazebo|deckga" | head -n 120

echo "=== Marker topic (should appear after Terminal 4 starts) ==="
ros2 topic list | grep -E "/deckga/markers" || true
Terminal 4 — Publish DECK-GA paths to RViz (markers)
cd ~/Documents/GitHub/ROS2_MultiUAV_3D/aerostack_examples/02_examples_gazebo_project/project_gazebo
source /opt/ros/humble/setup.bash
source ~/as2_harmonic_ws/install/setup.bash

python3 ~/Documents/GitHub/ROS2_MultiUAV_3D/deckga_ros2/rviz_paths_node.py
After starting this, return to Terminal 3 and re-run the topic check; you should now see /deckga/markers.

Terminal 5 — Execute (flies the same transformed path)
cd ~/Documents/GitHub/ROS2_MultiUAV_3D/aerostack_examples/02_examples_gazebo_project/project_gazebo
source /opt/ros/humble/setup.bash
source ~/as2_harmonic_ws/install/setup.bash

python3 ~/Documents/GitHub/ROS2_MultiUAV_3D/deckga_ros2/deckga_execute.py \
  --deckga_pkl ~/Documents/GitHub/ROS2_MultiUAV_3D/deckga_ros2/data/deckga_output.pkl
Timing outputs (printed by deckga_execute.py):

Planned timing (kinematic model): per-UAV path length / speed and mission makespan

Executed timing (wall-clock): arming, takeoff, path loop, hover, landing

Mission completion time: first waypoint command → final waypoint command, and makespan

Terminal 6 — Clean stop (end experiment)
cd ~/Documents/GitHub/ROS2_MultiUAV_3D/aerostack_examples/02_examples_gazebo_project/project_gazebo
./stop.bash || true

pkill -u "$USER" -f deckga_execute.py || true
pkill -u "$USER" -f rviz_paths_node.py || true
pkill -u "$USER" -f rviz2 || true
pkill -u "$USER" -f gzserver || true
pkill -u "$USER" -f gzclient || true

ros2 daemon stop || true
ros2 daemon start || true
Antarctica (ASPA135) Experiment
For the full terminal-by-terminal Antarctica reproduction (waypoints TXT→PKL, DECK-GA planning, Gazebo launch,
RViz publish, execution, logs), see:

ANTARCTICA_EXPERIMENT.md (branch: antarctica)

This workflow uses the aspa135_mini3 terrain model and thermal overlay stored via Git LFS.

Notes on Waypoints Collection (Gazebo-only)
In the Antarctica Gazebo project, you can collect waypoints by manually flying and sampling:

/droneX/ground_truth/pose (Gazebo truth)

/droneX/self_localization/pose (estimated pose)

This allows generating mission-specific waypoints without PX4/QGroundControl.
