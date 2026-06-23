# Experiment Progress Notes
Last updated: 2026-06-24

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

---

### 3 UAVs — 30 Waypoints
| Stage | Status |
|---|---|
| DECK-GA planner 10 runs | COMPLETE — `results_csv/deck_ga_3uavs_30_points_10_runs_results.txt`, PKLs ex1–ex10 in `deckga_ros2/data/` |
| Gazebo flight 10 runs | COMPLETE — all CSVs in `results_antarctica_csv/` |
| Summary table | COMPLETE — `results_csv/summary_table_uav3_30pts.txt` |

Start points: `4.00,8.00,32.31 ; 7.00,6.00,32.09 ; 5.50,4.00,32.20`
Results: planned total = 302.686 m (identical all runs), mission makespan mean = 86.580 s, min pairwise mean = 1.692 m.

### 3 UAVs — 60 Waypoints
| Stage | Status |
|---|---|
| DECK-GA planner 10 runs | COMPLETE — `results_csv/deck_ga_3uavs_60_points_10_runs_results.txt`, PKLs ex1–ex10 in `deckga_ros2/data/` |
| Gazebo flight 10 runs | COMPLETE — all CSVs in `results_antarctica_csv/` |
| Summary table | COMPLETE — `results_csv/summary_table_uav3_60pts.txt` |

Start points: `4.00,8.00,32.31 ; 7.00,6.00,32.09 ; 5.50,4.00,32.20`
Results: planned total mean = 338.703 m, mission makespan mean = 125.610 s, min pairwise mean = 1.559 m.

### 3 UAVs — 90 Waypoints
| Stage | Status |
|---|---|
| DECK-GA planner 10 runs | COMPLETE — `results_csv/deck_ga_3uavs_90_points_10_runs_results.txt`, PKLs ex1–ex10 in `deckga_ros2/data/` |
| Gazebo flight 10 runs | COMPLETE — all CSVs in `results_antarctica_csv/` |
| Summary table | COMPLETE — `results_csv/summary_table_uav3_90pts.txt` |

Start points: `5.50,4.00,32.20 ; 7.00,6.00,32.09 ; 4.00,8.00,32.31`
Results: planned total mean = 381.681 m, mission makespan mean = 156.213 s, min pairwise mean = 1.510 m.

### 3 UAVs — 120 Waypoints
| Stage | Status |
|---|---|
| DECK-GA planner 10 runs | COMPLETE — `results_csv/deck_ga_3uavs_120_points_10_runs_results.txt`, PKLs ex1–ex10 in `deckga_ros2/data/` |
| Gazebo flight 10 runs | COMPLETE — all CSVs in `results_antarctica_csv/` |
| Summary table | COMPLETE — `results_csv/summary_table_uav3_120pts.txt` |

Start points: `7.00,6.00,32.09 ; 5.50,4.00,32.20 ; 4.00,8.00,32.31`
Results: planned total mean = 438.334 m, mission makespan mean = 184.028 s, min pairwise mean = 1.952 m.

---

### 4 UAVs — 30 / 60 / 90 / 120 Waypoints — NOT STARTED
### 5 UAVs — 30 / 60 / 90 / 120 Waypoints — NOT STARTED

---

## 2. Overall Completion Tracker

```
2-UAV:  30 ✓   60 ✓   90 ✓   120 ✓
3-UAV:  30 ✓   60 ✓   90 ✓   120 ✓
4-UAV:  30 _   60 _   90 _   120 _
5-UAV:  30 _   60 _   90 _   120 _

Completed: 8 of 16 configurations
```

---

## 3. Proven Pipeline (replicate for 4/5 UAVs)

| Step | Script / Command |
|---|---|
| Run DECK-GA planner 10x | `bash run_deckga_10x_{N}uav_{WPTS}pts.sh` |
| Run Gazebo flights 10x | `bash run_data_runs_uav{N}_{WPTS}pts.sh 1 10` |
| Generate summary table | `python3 make_summary_table.py {WPTS} {N}` → saves to `results_csv/` |

---

## 4. Spawn / Start Points

### 4.1 — 2-UAV Configs (all waypoint counts)
```
drone0:  x=7.00,  y=4.24,  z=32.31
drone1:  x=4.50,  y=4.24,  z=32.09
--start_points="7.00,4.24,32.31;4.50,4.24,32.09"
```

### 4.2 — 3-UAV Configs (start points differ per waypoint count — see Section 1)
Depot box used: x:4–7, y:4–8, z:32.00–32.99, all pairs ≥ 2 m apart.

---

## 5. CRITICAL: Start Point Assignment for 4-UAV and 5-UAV

**Each waypoint count produces different DCKmeans cluster geometry, so depot-to-drone-index assignment differs per config. NEVER reuse start points from one waypoint count for another without verifying.**

### Procedure for each new (N-UAV, W-waypoint) config:

1. Run planner once with any valid start points (all in box x:4–7, y:4–8, z:32–32.99, pairs ≥ 2 m apart).
2. Run the region-printing script below to get UAV index → cluster region mapping.
3. Assign each drone the depot corner pointing toward its cluster.
4. Verify with plot — no path crossings before running 10x.

### Region-printing script:
```python
python3 - << 'PY'
import pickle, numpy as np

p = "deckga_ros2/data/YOUR_TEST_PKL.pkl"
with open(p, "rb") as f:
    d = pickle.load(f)

paths = d["deckga_paths"]
for i, P in enumerate(paths):
    P = np.asarray(P, dtype=float)
    wp = P[1:]
    cx, cy = wp[:,0].mean(), wp[:,1].mean()
    region = ""
    region += "N" if cy > 8 else ("S" if cy < -8 else "")
    region += "E" if cx > 8 else ("W" if cx < -8 else "")
    print(f"UAV{i}: start=({P[0,0]:.2f},{P[0,1]:.2f})  cluster_center=({cx:.2f},{cy:.2f})  region={region or 'center'}")
PY
```

### Depot box constraints:
- All depots: x:4–7, y:4–8, z:32.00–32.99
- All pairs ≥ 2 m apart
- 4-UAV: need 4 depots in box — tight but possible
- 5-UAV: need 5 depots — requires careful geometry

---

## 6. What Changes for 4 / 5 UAVs

### 6.1 Start Points — Read from the correct YAML first
Each UAV count has its own world swarm config:
- 4 UAVs → `project_gazebo/config/world_swarm_4.yaml`
- 5 UAVs → `project_gazebo/config/world_swarm_5.yaml`

**Before building any script for N UAVs:** read `world_swarm_N.yaml`, extract all spawn xyz coordinates, and use those exact values for `--start_points` and `--num_uavs N` in DECK_GA. Do NOT reuse start points from another UAV count or waypoint count.

### 6.2 Pairwise Distance Logger
Must cover all drones. Examples:
- 4 UAVs: `--topics /drone0/.../pose,...,/drone3/.../pose --names drone0,drone1,drone2,drone3`
- 5 UAVs: add `/drone4/ground_truth/pose` and `drone4`

### 6.3 Flight Runner Script
All of these arrays must include every drone (drone0..droneN-1):
- `LIVE_POSE_TOPICS`
- `REQUIRED_ACTIONS` (TakeoffBehavior, GoToBehavior, LandBehavior, FollowPathBehavior for each drone)
- `TMUX_SESSIONS`
- `--namespaces` passed to the executor

### 6.4 z-range and Timing Flags
Verify per configuration from each command file Terminal 9. Do not assume they carry over. Speed is fixed at **1.9 m/s** across all configurations.

---

## 7. Next Session — 4 UAVs (all 4 waypoint counts)

1. Confirm `world_swarm_4.yaml` exists and has 4 drone entries with valid depot positions.
2. For each waypoint count (30/60/90/120):
   a. Run planner once with trial start points → run region-printing script → assign correct depots.
   b. Build `run_deckga_10x_4uav_{WPTS}pts.sh` with correct `--start_points` and `--num_uavs 4`.
   c. Run it 10x to generate PKLs ex1–ex10.
   d. Build `run_data_runs_uav4_{WPTS}pts.sh` — add drone3 to all arrays.
   e. Test with single run: `bash run_data_runs_uav4_{WPTS}pts.sh 1 1`
   f. Verify run is GOOD, then batch runs 2–10.
   g. Run `python3 make_summary_table.py {WPTS} 4`.
3. Repeat for 5 UAVs.

---

## 8. Verification Habit — Per-Run Health Check

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
