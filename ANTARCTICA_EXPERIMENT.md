# Antarctica Experiment (ASPA135_m3 / aspa135_mini3) — Reproducible Run Guide

This branch documents how to run the full experiment pipeline:
1) Convert selected Antarctica waypoints (TXT → PKL)
2) Run DECK-GA (produces mission paths per UAV)
3) Launch Aerostack2 + Gazebo (Antarctica world)
4) Visualize paths in RViz
5) Execute multi-UAV flight in simulation and log results

> Notes:
> - This repo uses **Git LFS** for large Gazebo assets (terrain/mesh).
> - You do **not** need PX4/QGC for this pipeline (Gazebo + Aerostack2 only).

---

## 0) Prerequisites

### A) System / tools
- Ubuntu 22.04
- ROS2 Humble
- Aerostack2 installed in `~/as2_harmonic_ws`
- Python3

### B) Git LFS (required)
This repo stores Antarctica models via Git LFS. After cloning:

```bash
sudo apt update
sudo apt install git-lfs
git lfs install
git lfs pull
1) Inputs / Outputs
Inputs
data/points/antarctica_aspa135_m3_points.txt
Your curated waypoint list (x,y,z).

Generated inputs
data/points/antarctica_aspa135_m3_points.pkl
Pickle file containing N×3 waypoint array.

DECK-GA output
deckga_ros2/data/deckga_output_antarctica.pkl
Multi-UAV planned paths for RViz + execution.

Results folders (generated)
results_antarctica/ (figures)

results_antarctica_csv/ (execution logs)

Tip: Keep generated outputs untracked (don’t commit).

2) Terminal-by-terminal Run Procedure
Terminal 1 — Convert permanent TXT → PKL (and sanity-check)
cd ~/Documents/GitHub/ROS2_MultiUAV_3D

# Confirm the txt exists
ls -la data/points/antarctica_aspa135_m3_points.txt
head -n 5 data/points/antarctica_aspa135_m3_points.txt
tail -n 5 data/points/antarctica_aspa135_m3_points.txt

# Convert TXT -> PKL
python3 data/points/txt_to_points_pkl.py \
  --in_txt data/points/antarctica_aspa135_m3_points.txt \
  --out_pkl data/points/antarctica_aspa135_m3_points.pkl

# Verify ranges
python3 - << 'PY'
import pickle, numpy as np
p="data/points/antarctica_aspa135_m3_points.pkl"
pts=np.array(pickle.load(open(p,"rb")))
print("shape:", pts.shape)
print("x:", pts[:,0].min(), pts[:,0].max())
print("y:", pts[:,1].min(), pts[:,1].max())
print("z:", pts[:,2].min(), pts[:,2].max())
PY
Terminal 2 — Run DECK_GA using Antarctica waypoint PKL
Use start_points that match your swarm spawn (from world_swarm.yaml):

drone0: (5.05, 3.24, 32.31)

drone1: (5.65, 6.10, 32.09)

drone2: (6.00, 4.50, 32.20)

cd ~/Documents/GitHub/ROS2_MultiUAV_3D

python3 DECK_GA.py \
  --points_pkl "data/points/antarctica_aspa135_m3_points.pkl" \
  --num_uavs 3 \
  --start_points="5.05,3.24,32.31;5.65,6.10,32.09;6.0,4.5,32.2" \
  --out_pkl "deckga_ros2/data/deckga_output_antarctica.pkl" \
  --save_fig_dir "results_antarctica"
Terminal 3 — Launch Aerostack2 + Gazebo swarm (Antarctica)
cd ~/Documents/GitHub/ROS2_MultiUAV_3D/aerostack_examples/02_examples_gazebo_project/project_gazebo

source /opt/ros/humble/setup.bash
source ~/as2_harmonic_ws/install/setup.bash
source ./setup_antarctica_env.bash

./launch_as2.bash -m
Terminal 4 — Ground station / RViz (3 drones + teleop)
cd ~/Documents/GitHub/ROS2_MultiUAV_3D/aerostack_examples/02_examples_gazebo_project/project_gazebo

source /opt/ros/humble/setup.bash
source ~/as2_harmonic_ws/install/setup.bash
source ./setup_antarctica_env.bash

./launch_ground_station.bash -m -t -v
Terminal 5 — Quick sanity: actions exist for all drones
source /opt/ros/humble/setup.bash
source ~/as2_harmonic_ws/install/setup.bash

ros2 action list | sort | egrep "/drone[0-2]/(TakeoffBehavior|GoToBehavior|LandBehavior|FollowPathBehavior)"
Terminal 6 — Publish DECK-GA paths in RViz
cd ~/Documents/GitHub/ROS2_MultiUAV_3D

source /opt/ros/humble/setup.bash
source ~/as2_harmonic_ws/install/setup.bash

python3 deckga_ros2/rviz_paths_node_antarctica.py \
  --deckga_pkl deckga_ros2/data/deckga_output_antarctica.pkl \
  --coord_mode auto \
  --xy_scale 1.0 --xy_center_x 0.0 --xy_center_y 0.0 \
  --zin_min 33.34 --zin_max 39.30 \
  --zout_min 1.14 --zout_max 7.10 \
  --z_base 32.2 \
  --clamp_xy --x_min -30 --x_max 30 --y_min -30 --y_max 30 \
  --line_width 0.25 --wp_scale 0.45
Z mapping parameters (important):

zin_min/zin_max: min/max Z values in waypoint input (terrain altitude band)

zout_min/zout_max: desired flight altitude band in Gazebo

z_base: base offset used by the coordinate mapping logic

Terminal 7 — Execute flight (stable sequencing + logging)
cd ~/Documents/GitHub/ROS2_MultiUAV_3D

source /opt/ros/humble/setup.bash
source ~/as2_harmonic_ws/install/setup.bash

python3 deckga_ros2/deckga_execute_antarctica.py \
  --deckga_pkl deckga_ros2/data/deckga_output_antarctica.pkl \
  --coord_mode auto \
  --xy_scale 1.0 --xy_center_x 0.0 --xy_center_y 0.0 \
  --zin_min 33.34 --zin_max 39.30 \
  --zout_min 1.14 --zout_max 7.10 \
  --z_base 32.2 \
  --clamp_xy --x_min -30 --x_max 30 --y_min -30 --y_max 30 \
  --takeoff_margin 0.20 \
  --pre_arm_wait_s 12 \
  --speed 1.5 \
  --wait_actions_s 60 \
  --init_wait_s 5 \
  --takeoff_settle_s 6 \
  --takeoff_sequential \
  --first_wp_wait \
  --log_dir results_antarctica_csv \
  --run_tag aspa135_trial1
3) Stop / Kill Commands (clean + safe)
A) Stop the Python nodes
In any running Python terminal: Ctrl+C

B) Stop Aerostack2 launch scripts (if provided)
cd ~/Documents/GitHub/ROS2_MultiUAV_3D/aerostack_examples/02_examples_gazebo_project/project_gazebo
./stop.bash || true
C) Kill only your experiment Python scripts
pkill -u "$USER" -f "deckga_execute_antarctica.py" || true
pkill -u "$USER" -f "rviz_paths_node_antarctica.py" || true
D) Close RViz + Gazebo
pkill -u "$USER" -f rviz2 || true
pkill -u "$USER" -f "gz sim" || true
pkill -u "$USER" -f gzserver || true
pkill -u "$USER" -f gzclient || true
E) Optional ROS2 discovery reset (only if discovery is broken)
ros2 daemon stop || true
ros2 daemon start || true
rm -rf /tmp/ros* || true
4) Common Issues / Notes
If drones “snap back” after takeoff during teleoperation:
open the Teleop UI → Behavior control and pause any active behavior that is overriding manual commands.

If RViz paths look vertically offset:
review zin_*, zout_*, and z_base mapping parameters.
