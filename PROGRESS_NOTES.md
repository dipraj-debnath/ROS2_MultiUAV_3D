# Experiment Progress Notes
Last updated: 2026-06-23

---

## 1. Experiment Status

### 2 UAVs — 30 Waypoints
| Stage | Status |
|---|---|
| DECK-GA planner 10 runs | COMPLETE — `results_csv/deck_ga_2uavs_30_points_10_runs_results.txt` |
| Gazebo flight 10 runs | COMPLETE — all CSVs in `results_antarctica_csv/` |
| Summary table | COMPLETE — `results_csv/summary_table_uav2_30pts.txt` |

Results: optimized total = 283.965 m, mission makespan mean = 128.4 s, min pairwise mean = 1.715 m.

### 2 UAVs — 60 Waypoints
| Stage | Status |
|---|---|
| DECK-GA planner 10 runs | COMPLETE — `results_csv/deck_ga_2uavs_60_points_10_runs_results.txt`, PKLs ex1–ex10 in `deckga_ros2/data/` |
| Gazebo flight 10 runs | COMPLETE — all CSVs in `results_antarctica_csv/` |
| Summary table | COMPLETE — `results_csv/summary_table_uav2_60pts.txt` |

Results: optimized total = 370.698 m, mission makespan mean = 164.4 s, min pairwise mean = 2.476 m.

### 2 UAVs — 90 Waypoints
| Stage | Status |
|---|---|
| DECK-GA planner 10 runs | COMPLETE — `results_csv/deck_ga_2uavs_90_points_10_runs_results.txt`, PKLs ex1–ex10 in `deckga_ros2/data/` |
| Gazebo flight 10 runs | COMPLETE — all CSVs in `results_antarctica_csv/` |
| Summary table | COMPLETE — `results_csv/summary_table_uav2_90pts.txt` |

Results: optimized total = 434.885 m, mission makespan mean = 211.6 s, min pairwise mean = 0.839 m.

### 2 UAVs — 120 Waypoints
| Stage | Status |
|---|---|
| DECK-GA planner 10 runs | COMPLETE — `results_csv/deck_ga_2uavs_120_points_10_runs_results.txt`, PKLs ex1–ex10 in `deckga_ros2/data/` |
| Gazebo flight 10 runs | COMPLETE — all CSVs in `results_antarctica_csv/` |
| Summary table | COMPLETE — `results_csv/summary_table_uav2_120pts.txt` |

Results: optimized total = 450.477 m, mission makespan mean = 268.7 s, min pairwise mean = 1.392 m.

### 3 UAVs — 30 / 60 / 90 / 120 Waypoints — NOT STARTED
### 4 UAVs — 30 / 60 / 90 / 120 Waypoints — NOT STARTED
### 5 UAVs — 30 / 60 / 90 / 120 Waypoints — NOT STARTED

---

## 2. Proven Pipeline (2-UAV, replicate for 3/4/5 UAVs)

| Step | Script / Command |
|---|---|
| Run DECK-GA planner 10x | `bash run_deckga_10x_{WPTS}pts.sh` |
| Run Gazebo flights 10x | `bash run_data_runs_uav{N}_{WPTS}pts.sh 1 10` |
| Generate summary table | `python3 make_summary_table.py {WPTS} {UAVS}` → saves to `results_csv/` |

---

## 3. Spawn / Start Points — 2-UAV Configs

```
drone0:  x=7.00,  y=4.24,  z=32.31
drone1:  x=4.50,  y=4.24,  z=32.09
--start_points="7.00,4.24,32.31;4.50,4.24,32.09"
```

---

## 4. CRITICAL: What Changes for 3 / 4 / 5 UAVs

These are NOT simple find-replace operations. Each must be handled carefully per UAV count.

### 4.1 Start Points — Read from the correct YAML first
Each UAV count has its own world swarm config:
- 3 UAVs → `project_gazebo/config/world_swarm_3.yaml`
- 4 UAVs → `project_gazebo/config/world_swarm_4.yaml`
- 5 UAVs → `project_gazebo/config/world_swarm_5.yaml`

**Before building any script for N UAVs:** read `world_swarm_N.yaml`, extract all spawn xyz coordinates, and use those exact values for `--start_points` and `--num_uavs N` in DECK_GA. Do NOT reuse the 2-UAV start points.

### 4.2 Pairwise Distance Logger
Must cover all drones. Examples:
- 3 UAVs: `--topics /drone0/ground_truth/pose,/drone1/ground_truth/pose,/drone2/ground_truth/pose --names drone0,drone1,drone2`
- 4 UAVs: add `/drone3/ground_truth/pose` and `drone3`
- 5 UAVs: add `/drone4/ground_truth/pose` and `drone4`

### 4.3 Flight Runner Script
All of these arrays must include every drone (drone0..droneN-1):
- `LIVE_POSE_TOPICS`
- `REQUIRED_ACTIONS` (TakeoffBehavior, GoToBehavior, LandBehavior, FollowPathBehavior for each drone)
- `TMUX_SESSIONS`
- `--namespaces` passed to the executor

### 4.4 z-range and Timing Flags
Verify per configuration from each command file Terminal 9. Do not assume they carry over from 2-UAV runs. Speed is fixed at **1.9 m/s** across all configurations.

---

## 5. First Step Tomorrow — 3 UAVs / 30 Waypoints

1. Read `project_gazebo/config/world_swarm_3.yaml` — extract all three drone spawn xyz coordinates.
2. Check the 3-UAV 30-waypoint command file — confirm `--start_points` and `--num_uavs 3` match the yaml.
3. Build `run_deckga_10x_3uav_30pts.sh` — adapt from the 2-UAV version with the correct start points and `--num_uavs 3`.
4. Run it to generate `deckga_output_antarctica_30_uav3_ex1.pkl` through `ex10.pkl`.
5. Build `run_data_runs_uav3_30pts.sh` — adapt from `run_data_runs_uav2_30pts.sh` updating all drone arrays (LIVE_POSE_TOPICS, REQUIRED_ACTIONS, TMUX_SESSIONS, --namespaces, pairwise logger topics/names).
6. Test with a single run: `bash run_data_runs_uav3_30pts.sh 1 1`
7. Verify run is GOOD (see Section 6), then batch runs 2–10.
8. Run `python3 make_summary_table.py 30 3` to generate the summary table.

---

## 6. Verification Habit — Per-Run Health Check

**GOOD run:** pairwise max climbs well past 40 m AND total executed distance ≈ planner's expected total.

**FAILED run:** pairwise max stays flat and low (< 5 m) — drones never separated, likely a takeoff/arming failure due to stale Gazebo state.

Quick check after each run:
```bash
python3 make_summary_table.py <waypoints> <uavs>
```

Or inspect the latest pairwise summary CSV directly:
```bash
ls -t results_antarctica_csv/*pairwise_distance_summary* | head -1 | xargs cat
```

If a run fails, close the terminal, open a fresh one, and re-run that single run number.
