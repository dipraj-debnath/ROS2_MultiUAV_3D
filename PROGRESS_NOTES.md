# Experiment Progress Notes
Last updated: 2026-06-22

---

## 1. Experiment Status

### 2 UAVs — 30 Waypoints
| Stage | Status |
|---|---|
| DECK-GA planner 10 runs | COMPLETE — `deck_ga_2uavs_30_points_10_runs_results.txt` |
| Gazebo flight 10 runs | COMPLETE — all CSVs in `results_antarctica_csv/` |
| Summary table | COMPLETE — `results_antarctica_csv/summary_table_uav2_30pts.txt` |

Results: optimized total = 283.965 m, mission makespan mean = 128.4 s, min pairwise mean = 1.715 m.

### 2 UAVs — 60 Waypoints
| Stage | Status |
|---|---|
| DECK-GA planner 10 runs | COMPLETE — `deck_ga_2uavs_60_points_10_runs_results.txt`, PKLs ex1–ex10 in `deckga_ros2/data/` |
| Gazebo flight 10 runs | NOT STARTED — script not yet built |
| Summary table | NOT STARTED |

Results (planner only): optimized total = 370.698 m, GA time mean = 61.845 s.

### 2 UAVs — 90 Waypoints
| Stage | Status |
|---|---|
| DECK-GA planner 10 runs | NOT STARTED |
| Gazebo flight 10 runs | NOT STARTED |
| Summary table | NOT STARTED |

### 2 UAVs — 120 Waypoints
| Stage | Status |
|---|---|
| DECK-GA planner 10 runs | NOT STARTED |
| Gazebo flight 10 runs | NOT STARTED |
| Summary table | NOT STARTED |

---

## 2. Spawn / Start Points — All 2-UAV Configs

These must be consistent across `world_swarm_2.yaml`, `--start_points`, and all command files.

```
drone0:  x=7.00,  y=4.24,  z=32.31
drone1:  x=4.50,  y=4.24,  z=32.09

--start_points="7.00,4.24,32.31;4.50,4.24,32.09"
```

All four command files (30, 60, 90, 120 waypoints) have been updated to use these values.

---

## 3. Key Scripts

| Script | What it does |
|---|---|
| `run_deckga_10x.sh` | Runs DECK-GA planner 10x for 2 UAVs, 30 waypoints. Saves `deckga_output_antarctica_30_uav2_ex{1..10}.pkl`. |
| `run_deckga_10x_60pts.sh` | Same but for 60 waypoints. Saves `deckga_output_antarctica_60_uav2_ex{1..10}.pkl`. |
| `run_data_runs_uav2_30pts.sh <start> <end>` | Automated Gazebo flight runner for 30-waypoint 2-UAV experiment. Restarts Gazebo between runs, polls for live drone readiness (pose + FollowPathBehavior), runs pairwise logger and executor, cleans up, settles 15 s. Run one at a time from a fresh terminal: `bash run_data_runs_uav2_30pts.sh N N`. |
| `make_summary_table.py <waypoints> <uavs>` | Reads all 10 run CSVs from `results_antarctica_csv/` and writes an aligned summary table with planned/mission makespan, total distance, and min pairwise separation plus mean/std. Example: `python3 make_summary_table.py 30 2`. |
| `make_summary_table_uav2_30pts.py` | Original fixed version of the above for 30 pts / 2 UAVs — kept as reference. |

---

## 4. Next Step Tomorrow

**Build `run_data_runs_uav2_60pts.sh`** — the Gazebo flight automation script for the 60-waypoint 2-UAV case.

Base it on `run_data_runs_uav2_30pts.sh` but change the executor flags specific to 60 points:

```bash
--deckga_pkl "deckga_ros2/data/deckga_output_antarctica_60_uav2_ex${N}.pkl"
--zin_min 32.09 --zin_max 39.97
--zout_min 3.80 --zout_max 6.80
--z_base 32.2
--speed 1.5            # slower than 30-pt runs (was 1.9)
--wait_actions_s 30    # longer wait (was 10)
--takeoff_settle_s 3   # longer settle (was 1)
--log_dir results_antarctica_csv
--run_tag "ex${N}_60pts_uav2_deckga_run_${N}"
```

Pairwise logger run_tag: `ex${N}_60pts_uav2_pairwise_distance_run_${N}`

PKL pattern for make_summary_table.py will be: `*ex{N}_60pts_uav2_deckga_run_{N}*summary.csv`

After building the script, test with a single run: `bash run_data_runs_uav2_60pts.sh 1 1`

---

## 5. Verification Habit — Per-Run Health Check

After each Gazebo run completes, confirm the run is GOOD before moving to the next:

**GOOD run:** pairwise max > 40 m AND total executed distance ≈ expected planner total.

**FAILED run:** pairwise max < 5 m — drones never separated, likely a takeoff/arming failure due to stale Gazebo state.

Quick check command after each run:
```bash
python3 make_summary_table.py <waypoints> 2
```

Or manually inspect the latest pairwise summary CSV:
```bash
ls -t results_antarctica_csv/*pairwise_distance_summary* | head -1 | xargs cat
```

If a run fails, close the terminal, open a fresh one, and re-run that single run number.
