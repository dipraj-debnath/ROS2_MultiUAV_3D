# ROS2 Multi-UAV 3D — Antarctica DECK_GA Experiments

This branch contains the Antarctica multi-UAV mission-planning and execution experiments for the **DECK-GA** framework using **ROS 2 Humble**, **Aerostack2**, **Gazebo**, and **RViz**.

The experiments are conducted on the `aspa135_m3` Antarctica terrain world and evaluate multi-UAV path planning for different mission sizes and swarm sizes:

- Waypoints: **30, 60, 90, 120**
- UAVs: **2, 3, 4, 5**
- Planner: **DECK-GA**
- Benchmark planners:
  - **Traditional GA + Divide & Conquer**
  - **Traditional GA + Load Balancing** if enabled/used in further benchmarking

The repository includes terminal command files so the experiments can be reproduced without rewriting commands manually.

---

## 1. Repository Purpose

This repository supports:

1. Offline multi-UAV 3D path planning.
2. DECK-GA planning for Antarctica waypoint datasets.
3. ROS 2 / Aerostack2 execution in Gazebo.
4. RViz visualization of planned UAV paths.
5. Pairwise distance logging for collision-distance analysis.
6. Benchmarking against Traditional GA variants.

DECK-GA combines:

- **DCKmeans** for distance-aware clustering.
- **DEGA** for distance-efficient genetic-algorithm path planning.
- Multi-UAV route generation for mTSP-style inspection missions.

The Antarctica experiment represents a simulated UAV inspection scenario over Antarctic terrain, with waypoint altitudes selected to keep UAVs safely above the surface while avoiding unrealistically high flight.

---

## 2. Branch

Use the `antarctica` branch:

```bash
cd ~/Documents/GitHub/ROS2_MultiUAV_3D

git checkout antarctica
git pull origin antarctica
3. System Requirements

The experiments were run using:

Ubuntu 22.04
ROS 2 Humble
Aerostack2
Gazebo
RViz2
Python 3
Git LFS

Recommended Aerostack2 workspace:

~/as2_harmonic_ws

Every terminal must source the ROS and Aerostack2 environments:

source /opt/ros/humble/setup.bash
source ~/as2_harmonic_ws/install/setup.bash

For the Antarctica Gazebo project, also source:

source ./setup_antarctica_env.bash

from inside:

aerostack_examples/02_examples_gazebo_project/project_gazebo
4. Git LFS Requirement

The Antarctica Gazebo world uses large terrain assets. After cloning the repository, run:

sudo apt update
sudo apt install git-lfs
git lfs install
git lfs pull
5. Main Repository Structure
ROS2_MultiUAV_3D/
│
├── DECK_GA.py
├── DCKmeans.py
├── GA_path_planning.py
│
├── DEGA_Divide & Conquer/
│   └── Traditional_GA_Divide & Conquer.py
│
├── DEGA_Load Balancing/
│   └── Traditional_GA_Load_Balancing.py
│
├── data/
│   └── points/
│       ├── antarctica_aspa135_m3_30_points.txt
│       ├── antarctica_aspa135_m3_60_points.txt
│       ├── antarctica_aspa135_m3_90_points.txt
│       └── antarctica_aspa135_m3_120_points.txt
│
├── deckga_ros2/
│   ├── deckga_execute_antarctica.py
│   ├── rviz_paths_node_antarctica.py
│   ├── drone_pairwise_distance_logger.py
│   └── data/
│
├── aerostack_examples/
│   └── 02_examples_gazebo_project/
│       └── project_gazebo/
│           ├── config/
│           │   ├── world_swarm_2.yaml
│           │   ├── world_swarm_3.yaml
│           │   ├── world_swarm_4.yaml
│           │   ├── world_swarm_5.yaml
│           │   └── world_swarm.yaml
│           └── config_ground_station/
│
├── experiment_commands/
│   └── antarctica_deckga/
│
├── results_antarctica/
│
└── results_antarctica_csv/
6. Antarctica World

The Antarctica world used in these experiments is:

world_name: "aspa135_m3"
origin:
    latitude: -66.28223056
    longitude: 110.53892500
    altitude: 30.19

The active swarm file used by Aerostack2 is:

aerostack_examples/02_examples_gazebo_project/project_gazebo/config/world_swarm.yaml

For each experiment, copy the correct swarm configuration into the active file.

Example for 5 UAVs:

cd ~/Documents/GitHub/ROS2_MultiUAV_3D/aerostack_examples/02_examples_gazebo_project/project_gazebo

cp config/world_swarm_5.yaml config/world_swarm.yaml

cat config/world_swarm.yaml
7. UAV Start Configurations
2 UAVs
drone0 = 7.00, 4.24, 32.31
drone1 = 5.00, 4.24, 32.09
3 UAVs
drone0 = 7.00, 5.50, 32.31
drone1 = 5.50, 4.24, 32.09
drone2 = 5.50, 6.30, 32.20
4 UAVs
drone0 = 7.00, 6.10, 32.31
drone1 = 5.00, 4.24, 32.09
drone2 = 6.00, 5.50, 32.20
drone3 = 7.50, 4.54, 32.20
5 UAVs
drone0 = 7.50, 5.80, 32.80
drone1 = 4.50, 2.80, 32.80
drone2 = 4.50, 5.80, 32.80
drone3 = 7.50, 2.80, 32.80
drone4 = 6.00, 4.30, 33.20

These start positions are intentionally separated to reduce path overlap and improve takeoff stability in Gazebo/RViz.

8. Altitude Mapping Strategy

The DECK-GA and benchmark planner outputs use raw Antarctica waypoint altitudes. During RViz visualization and Aerostack2 execution, the scripts remap the raw z-values to a safer flight range.

The stable execution range used in the final experiments is approximately:

Final flight z range: 36.00 m to 39.00 m
Takeoff target: approximately 36.10 m

For 2, 3, and 4 UAV experiments, the lowest depot altitude is usually:

zin_min = 32.09

For 5 UAV experiments, the lowest depot altitude is usually:

zin_min = 32.80

The standard execution mapping is:

--zout_min 3.80
--zout_max 6.80
--z_base 32.2

This gives:

z_base + zout_min = 36.00 m
z_base + zout_max = 39.00 m

The takeoff command uses:

--takeoff_margin 0.10

Therefore the takeoff target is normally:

36.10 m
9. Standard Flight Parameters

The final successful experiments used the following stable execution parameters:

--takeoff_margin 0.10
--pre_arm_wait_s 5
--speed 1.5
--wait_actions_s 30
--init_wait_s 5
--takeoff_settle_s 3
--takeoff_sequential
--first_wp_wait
--ensure_reach
--wp_settle_s 0.3

These settings were selected because they provided stable takeoff, reliable waypoint tracking, and successful completion across 2, 3, 4, and 5 UAV configurations.

10. DECK-GA Experiment Matrix

The completed DECK-GA experiments cover:

Waypoints	2 UAVs	3 UAVs	4 UAVs	5 UAVs
30	Completed	Completed	Completed	Completed
60	Completed	Completed	Completed	Completed
90	Completed	Completed	Completed	Completed
120	Completed	Completed	Completed	Completed

The command files are stored in:

experiment_commands/antarctica_deckga/

Example DECK-GA command files:

antarctica_2uavs_30_waypoints_deckga_terminal_command.txt
antarctica_2uavs_60_waypoints_deckga_terminal_command.txt
antarctica_2uavs_90_waypoints_deckga_terminal_command.txt
antarctica_2uavs_120_waypoints_deckga_terminal_command.txt

antarctica_3uavs_30_waypoints_deckga_terminal_command.txt
antarctica_3uavs_60_waypoints_deckga_terminal_command.txt
antarctica_3uavs_90_waypoints_deckga_terminal_command.txt
antarctica_3uavs_120_waypoints_deckga_terminal_command.txt

antarctica_4uavs_30_waypoints_deckga_terminal_command.txt
antarctica_4uavs_60_waypoints_deckga_terminal_command.txt
antarctica_4uavs_90_waypoints_deckga_terminal_command.txt
antarctica_4uavs_120_waypoints_deckga_terminal_command.txt

antarctica_5uavs_30_waypoints_deckga_terminal_command.txt
antarctica_5uavs_60_waypoints_deckga_terminal_command.txt
antarctica_5uavs_90_waypoints_deckga_terminal_command.txt
antarctica_5uavs_120_waypoints_deckga_terminal_command.txt
11. Benchmark Experiment Matrix

Traditional GA + Divide & Conquer benchmark command files are also included.

Waypoints	2 UAVs	3 UAVs	4 UAVs	5 UAVs
30	Completed	Completed	Completed	Completed
60	Completed	Completed	Completed	Completed
90	Completed	Completed	Completed	Completed
120	Completed	Completed	Completed	Completed

Benchmark command files:

traditional_ga_divide_conquer_2uavs_30_waypoints_terminal_command.txt
traditional_ga_divide_conquer_2uavs_60_waypoints_terminal_command.txt
traditional_ga_divide_conquer_2uavs_90_waypoints_terminal_command.txt
traditional_ga_divide_conquer_2uavs_120_waypoints_terminal_command.txt

traditional_ga_divide_conquer_3uavs_30_waypoints_terminal_command.txt
traditional_ga_divide_conquer_3uavs_60_waypoints_terminal_command.txt
traditional_ga_divide_conquer_3uavs_90_waypoints_terminal_command.txt
traditional_ga_divide_conquer_3uavs_120_waypoints_terminal_command.txt

traditional_ga_divide_conquer_4uavs_30_waypoints_terminal_command.txt
traditional_ga_divide_conquer_4uavs_60_waypoints_terminal_command.txt
traditional_ga_divide_conquer_4uavs_90_waypoints_terminal_command.txt
traditional_ga_divide_conquer_4uavs_120_waypoints_terminal_command.txt

traditional_ga_divide_conquer_5uavs_30_waypoints_terminal_command.txt
traditional_ga_divide_conquer_5uavs_60_waypoints_terminal_command.txt
traditional_ga_divide_conquer_5uavs_90_waypoints_terminal_command.txt
traditional_ga_divide_conquer_5uavs_120_waypoints_terminal_command.txt
12. General Experiment Workflow

Each experiment follows the same structure.

Step 1 — Clean ROS
cd ~/Documents/GitHub/ROS2_MultiUAV_3D

source /opt/ros/humble/setup.bash
source ~/as2_harmonic_ws/install/setup.bash

ros2 daemon stop || true
rm -rf /dev/shm/fastrtps_* /dev/shm/fastdds_* 2>/dev/null || true
ros2 daemon start
Step 2 — Activate the correct swarm file

Example for 4 UAVs:

cd ~/Documents/GitHub/ROS2_MultiUAV_3D/aerostack_examples/02_examples_gazebo_project/project_gazebo

cp config/world_swarm_4.yaml config/world_swarm.yaml

cat config/world_swarm.yaml
Step 3 — Convert waypoint TXT to PKL

Example for 90 waypoints:

cd ~/Documents/GitHub/ROS2_MultiUAV_3D

python3 - << 'PY'
import numpy as np
import pickle
from pathlib import Path

txt_in = Path("data/points/antarctica_aspa135_m3_90_points.txt")
pkl_out = Path("data/points/antarctica_aspa135_m3_90_points.pkl")

pts = np.genfromtxt(txt_in, dtype=float, names=True, delimiter=None)
xyz = np.vstack([pts["x"], pts["y"], pts["z"]]).T

print("Shape:", xyz.shape)
print("Ranges:", xyz.min(axis=0), xyz.max(axis=0))

with pkl_out.open("wb") as f:
    pickle.dump(xyz, f)

print("Saved:", pkl_out)
PY
Step 4 — Run planner

Example DECK-GA:

python3 DECK_GA.py \
  --points_pkl "data/points/antarctica_aspa135_m3_90_points.pkl" \
  --num_uavs 4 \
  --start_points="7.00,6.10,32.31;5.00,4.24,32.09;6.00,5.50,32.20;7.50,4.54,32.20" \
  --out_pkl "deckga_ros2/data/deckga_output_antarctica_90_uav4.pkl" \
  --save_fig_dir "results_antarctica"

Example Traditional GA + Divide & Conquer:

python3 "DEGA_Divide & Conquer/Traditional_GA_Divide & Conquer.py" \
  --points_pkl "data/points/antarctica_aspa135_m3_90_points.pkl" \
  --num_uavs 4 \
  --start_points="7.00,6.10,32.31;5.00,4.24,32.09;6.00,5.50,32.20;7.50,4.54,32.20" \
  --out_pkl "deckga_ros2/data/traditional_ga_divide_conquer_output_antarctica_90_uav4.pkl" \
  --save_fig_dir "results_antarctica" \
  --population_size 50 \
  --num_iterations 25000
Step 5 — Launch Gazebo and Aerostack2
cd ~/Documents/GitHub/ROS2_MultiUAV_3D/aerostack_examples/02_examples_gazebo_project/project_gazebo

source /opt/ros/humble/setup.bash
source ~/as2_harmonic_ws/install/setup.bash
source ./setup_antarctica_env.bash

./launch_as2.bash -m
Step 6 — Launch RViz ground station
cd ~/Documents/GitHub/ROS2_MultiUAV_3D/aerostack_examples/02_examples_gazebo_project/project_gazebo

source /opt/ros/humble/setup.bash
source ~/as2_harmonic_ws/install/setup.bash
source ./setup_antarctica_env.bash

./launch_ground_station.bash -m -t -v
Step 7 — Publish paths in RViz

Example for a 4-UAV, 90-waypoint DECK-GA run:

cd ~/Documents/GitHub/ROS2_MultiUAV_3D

source /opt/ros/humble/setup.bash
source ~/as2_harmonic_ws/install/setup.bash

python3 deckga_ros2/rviz_paths_node_antarctica.py \
  --deckga_pkl deckga_ros2/data/deckga_output_antarctica_90_uav4.pkl \
  --num_uavs 4 \
  --coord_mode auto \
  --xy_scale 1.0 --xy_center_x 0.0 --xy_center_y 0.0 \
  --zin_min 32.09 --zin_max 39.99 \
  --zout_min 3.80 --zout_max 6.80 \
  --z_base 32.2 \
  --clamp_xy --x_min -40 --x_max 40 --y_min -40 --y_max 40 \
  --line_width 0.25 --wp_scale 0.45
Step 8 — Start pairwise distance logger

Example for 4 UAVs:

cd ~/Documents/GitHub/ROS2_MultiUAV_3D

source /opt/ros/humble/setup.bash
source ~/as2_harmonic_ws/install/setup.bash

python3 deckga_ros2/drone_pairwise_distance_logger.py \
  --topics /drone0/ground_truth/pose,/drone1/ground_truth/pose,/drone2/ground_truth/pose,/drone3/ground_truth/pose \
  --names drone0,drone1,drone2,drone3 \
  --msg_type pose_stamped \
  --rate_hz 10 \
  --print_every_s 1.0 \
  --log_dir results_antarctica_csv \
  --run_tag aspa135_90pts_uav4_pairwise_distance_test_run_1
Step 9 — Execute flight

Example for 4 UAVs, 90 waypoints:

cd ~/Documents/GitHub/ROS2_MultiUAV_3D

source /opt/ros/humble/setup.bash
source ~/as2_harmonic_ws/install/setup.bash

python3 deckga_ros2/deckga_execute_antarctica.py \
  --deckga_pkl deckga_ros2/data/deckga_output_antarctica_90_uav4.pkl \
  --namespaces drone0,drone1,drone2,drone3 \
  --coord_mode auto \
  --xy_scale 1.0 --xy_center_x 0.0 --xy_center_y 0.0 \
  --zin_min 32.09 --zin_max 39.99 \
  --zout_min 3.80 --zout_max 6.80 \
  --z_base 32.2 \
  --clamp_xy --x_min -40 --x_max 40 --y_min -40 --y_max 40 \
  --takeoff_margin 0.10 \
  --pre_arm_wait_s 5 \
  --speed 1.5 \
  --wait_actions_s 30 \
  --init_wait_s 5 \
  --takeoff_settle_s 3 \
  --takeoff_sequential \
  --first_wp_wait \
  --ensure_reach \
  --wp_settle_s 0.3 \
  --log_dir results_antarctica_csv \
  --run_tag aspa135_90pts_uav4_deckga_test_run_1
13. Results and Logs

Execution results are saved in:

results_antarctica_csv/

Typical output files include:

*_summary.csv
*_uav0_segments.csv
*_uav1_segments.csv
*_uav2_segments.csv
*_uav3_segments.csv
*_uav4_segments.csv
*_pairwise_distances.csv
*_pairwise_distance_summary.csv

These files support later analysis of:

planned path length
execution time
per-UAV route allocation
mission makespan
pairwise UAV separation
benchmark comparison
14. Clean Stop

After each experiment, stop Gazebo, RViz, ROS nodes, and background scripts:

cd ~/Documents/GitHub/ROS2_MultiUAV_3D/aerostack_examples/02_examples_gazebo_project/project_gazebo

./stop.bash || true

pkill -u "$USER" -f "deckga_execute_antarctica.py" || true
pkill -u "$USER" -f "rviz_paths_node_antarctica.py" || true
pkill -u "$USER" -f "drone_pairwise_distance_logger.py" || true
pkill -u "$USER" -f "rviz2" || true
pkill -u "$USER" -f "gz sim" || true
pkill -u "$USER" -f "gzserver" || true
pkill -u "$USER" -f "gzclient" || true

ros2 daemon stop || true
ros2 daemon start || true
15. Common Troubleshooting
FastDDS / shared memory error

If RViz path publishing or ROS 2 communication gives FastDDS shared memory errors, run:

ros2 daemon stop || true
rm -rf /dev/shm/fastrtps_* /dev/shm/fastdds_* 2>/dev/null || true
ros2 daemon start
RViz path publisher exits with ExternalShutdownException

This usually happens when ROS 2 or RViz is stopped while the node is spinning. It is not necessarily a planner failure. Restart the ROS daemon and rerun the RViz path publisher.

Drone does not appear in Gazebo

Check that the correct swarm file was copied:

cat aerostack_examples/02_examples_gazebo_project/project_gazebo/config/world_swarm.yaml

Then verify drone nodes:

ros2 node list | egrep "drone0|drone1|drone2|drone3|drone4"
Drone flies too low or too high

Check the altitude mapping:

--zin_min
--zin_max
--zout_min
--zout_max
--z_base

For the final stable Antarctica experiments, the recommended flight range is:

36.00 m to 39.00 m
16. Git Notes

Generated .pkl files and backup files may appear as untracked files. These are not always required for the repository because they can be regenerated from the TXT waypoint files and command files.

To stage only tracked modified/deleted files:

git add -u

To add specific command files safely:

git add "experiment_commands/antarctica_deckga/<filename>.txt"

Avoid using:

git add .

unless you intentionally want to add all untracked generated files.

17. Recommended Citation/Description for Paper

This branch implements a ROS 2/Aerostack2-based experimental validation workflow for multi-UAV 3D mission planning over an Antarctica terrain model. DECK-GA generates mTSP-style multi-UAV paths for varying waypoint densities and swarm sizes, while Gazebo and RViz are used for physics-based execution and visualization. Pairwise UAV separation and execution logs are collected to support quantitative comparison against traditional GA-based benchmark methods.

18. Status

Completed experiment set:

DECK-GA:
  30, 60, 90, 120 waypoints
  2, 3, 4, 5 UAVs

Traditional GA + Divide & Conquer:
  30, 60, 90, 120 waypoints
  2, 3, 4, 5 UAVs
