# ROS2_MultiUAV_3D

Multi-UAV 3D mission execution in ROS2 / Aerostack2 with DECK_GA (DCKmeans + Genetic Algorithm path planning).

Scope (ICUAS 2026):
- DECK_GA only (no QuickNav / no obstacle avoidance)
- Multi-UAV scaling experiments (3–6 UAVs)
- Waypoint set sizes (e.g., 30, 50, 100, 200)
- Benchmarking vs other planners in the same Aerostack2/Gazebo simulation environment
- 3D terrain is used for realistic altitude profiles (not for 3D obstacle avoidance)

Repository structure:
- aerostack_examples/ : Aerostack2 / Gazebo project launch files and configs
- deckga_ros2/        : Python scripts for loading planned routes and executing them in Aerostack2
- DECK_GA.py, DCKmeans.py, GA_path_planning.py : DECK_GA planning components (offline / research scripts)
