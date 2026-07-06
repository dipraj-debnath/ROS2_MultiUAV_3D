# Experiment Progress Notes
Last updated: 2026-07-08

---

## 1. Experiment Status

### 2 UAVs — 30 Waypoints ✓
Results: optimized total = 283.965 m, mission makespan mean = 128.4 s, min pairwise mean = 1.715 m.

### 2 UAVs — 60 Waypoints ✓
Results: optimized total = 370.698 m, mission makespan mean = 164.4 s, min pairwise mean = 2.476 m.

### 2 UAVs — 90 Waypoints ✓
Results: optimized total = 434.885 m, mission makespan mean = 211.6 s, min pairwise mean = 0.839 m.

### 2 UAVs — 120 Waypoints ✓
Results: optimized total = 450.477 m, mission makespan mean = 268.7 s, min pairwise mean = 1.392 m.

---

### 3 UAVs — 30 Waypoints ✓
Start points: `4.00,8.00,32.31 ; 7.00,6.00,32.09 ; 5.50,4.00,32.20`
Results: planned total = 302.686 m, mission makespan mean = 86.580 s, min pairwise mean = 1.692 m.

### 3 UAVs — 60 Waypoints ✓
Start points: `4.00,8.00,32.31 ; 7.00,6.00,32.09 ; 5.50,4.00,32.20`
Results: planned total mean = 338.703 m, mission makespan mean = 125.610 s, min pairwise mean = 1.559 m.

### 3 UAVs — 90 Waypoints ✓
Start points: `5.50,4.00,32.20 ; 7.00,6.00,32.09 ; 4.00,8.00,32.31`
Results: planned total mean = 381.681 m, mission makespan mean = 156.213 s, min pairwise mean = 1.510 m.

### 3 UAVs — 120 Waypoints ✓
Start points: `7.00,6.00,32.09 ; 5.50,4.00,32.20 ; 4.00,8.00,32.31`
Results: planned total mean = 438.334 m, mission makespan mean = 184.028 s, min pairwise mean = 1.952 m.

---

### 4 UAVs — 30 Waypoints ✓
Start points (DECK-GA planning): `4.00,8.00,32.31 ; 7.00,6.00,32.09 ; 4.00,4.00,32.20 ; 7.00,4.00,32.00`
Gazebo spawn (drone3 z=32.50 for visibility): drone3 xyz = 7.00, 4.00, 32.50
zin_min=32.00, zin_max=39.98
Results: GA opt=367.102 m (identical), mission makespan mean=77.011 s, dist=349.888 m, pairwise mean=1.729 m.

### 4 UAVs — 60 Waypoints ✓
Start points (DECK-GA planning): `7.00,4.00,32.00 ; 7.00,8.00,32.09 ; 4.00,8.00,32.31 ; 4.00,4.00,32.20`
Gazebo spawn (drone3 z=32.50 for visibility): drone3 xyz = 4.00, 4.00, 32.50
zin_min=32.00, zin_max=39.98
Results: GA opt=380.409 m (identical), mission makespan mean=122.442 s, dist=353.861 m, pairwise mean=2.938 m.

### 4 UAVs — 90 Waypoints ✓
Start points (DECK-GA planning): `4.00,4.00,32.50 ; 7.00,8.00,32.09 ; 4.00,8.00,32.31 ; 7.00,4.00,32.00`
Gazebo spawn (drone3 z=32.50 for visibility): drone3 xyz = 7.00, 4.00, 32.50
zin_min=32.00, zin_max=39.99
Results: GA opt mean=423.276 m, mission makespan mean=150.932 s, dist=386.388 m, pairwise mean=2.853 m.

### 4 UAVs — 120 Waypoints ✓
Start points: `7.00,8.00,32.09 ; 4.00,4.00,32.50 ; 4.00,8.00,32.31 ; 7.00,4.00,32.50`
zin_min=32.09, zin_max=39.99
Results: GA opt mean=476.862 m, mission makespan mean=185.592 s, dist=435.014 m, pairwise mean=1.494 m.

---

### 5 UAVs — 30 Waypoints ✓
Start points (DECK-GA planning + Gazebo spawn):
`4.50,6.00,32.80 ; 8.00,6.00,32.80 ; 4.60,2.50,32.80 ; 8.00,2.50,32.20 ; 2.50,3.50,33.20`
zin_min=32.20, zin_max=39.98
Results: GA opt=400.359 m (identical), mission makespan mean=76.641 s, dist=387.595 m, pairwise mean=1.497 m.

### 5 UAVs — 60 Waypoints ✓
Start points (DECK-GA planning + Gazebo spawn):
`8.00,4.50,32.60 ; 8.00,8.00,32.70 ; 4.50,8.00,32.80 ; 4.50,2.50,32.90 ; 6.00,5.00,32.99`
zin_min=32.60, zin_max=39.97
Results: GA opt=415.090m (identical), mission makespan mean=104.786s, dist=393.271m, pairwise mean=1.941m.

### 5 UAVs — 90 Waypoints ✓
Start points (DECK-GA planning + Gazebo spawn):
`4.50,2.50,32.90 ; 8.00,8.00,32.70 ; 4.50,8.00,32.80 ; 8.00,4.50,32.60 ; 6.00,5.00,32.99`
zin_min=32.60, zin_max=39.99
Results: GA opt mean=455.308m, mission makespan mean=102.708s, dist=419.607m, pairwise mean=1.907m.

### 5 UAVs — 120 Waypoints ✓
Start points (DECK-GA planning + Gazebo spawn):
`8.00,8.00,32.70 ; 4.50,2.50,32.90 ; 4.50,8.00,32.80 ; 8.00,4.50,32.60 ; 6.00,5.00,32.99`
zin_min=32.60, zin_max=39.99
Results: GA opt mean=510.384m, mission makespan mean=174.459s, dist=469.238m, pairwise mean=1.938m.

---

## 2. Overall Completion Tracker

```
2-UAV:  30 ✓   60 ✓   90 ✓   120 ✓
3-UAV:  30 ✓   60 ✓   90 ✓   120 ✓
4-UAV:  30 ✓   60 ✓   90 ✓   120 ✓
5-UAV:  30 ✓   60 ✓   90 ✓   120 ✓

Completed: 16 of 16 configurations — ALL DONE
```

---

## 3. Proven Pipeline (replicate for each config)

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

### 4.2 — 3-UAV Configs
Depot box used: x:4–7, y:4–8, z:32.00–32.99, all pairs ≥ 2 m apart.
See Section 1 for per-waypoint-count start points.

### 4.3 — 4-UAV Configs
Config in `world_swarm_4.yaml`. Drone3 Gazebo spawn z always set to 32.50 for visibility.
See Section 1 for per-waypoint-count start points.

### 4.4 — 5-UAV Configs
Config in `world_swarm_5.yaml`. Positions use wider spacing (x up to 8.00).
See Section 1 for per-waypoint-count start points.

---

## 5. CRITICAL: Start Point Assignment

**Each waypoint count produces different DCKmeans cluster geometry. NEVER reuse start points from one waypoint count for another without verifying.**

### Procedure for each new (N-UAV, W-waypoint) config:
1. Run planner once with trial start points (read from `world_swarm_N.yaml`).
2. Run the region-printing script to get UAV index → cluster region mapping.
3. Assign each drone the depot corner pointing toward its cluster.
4. Verify with plot — no path crossings before running 10x.

### Region-printing script:
```python
python3 - << 'PY'
import pickle, numpy as np

p = "deckga_ros2/data/YOUR_TEST_PKL.pkl"
with open(p, "rb") as f:
    d = pickle.load(f)

paths    = d["deckga_paths"]
centroids = np.array(d["centroids"])
offset   = np.array(d["offset_used"])

for i, (c, P) in enumerate(zip(centroids, paths)):
    P = np.asarray(P, dtype=float)
    cx = c[0] - offset[0]
    cy = c[1] - offset[1]
    depot_x = P[0, 0]
    depot_y = P[0, 1]
    depot_z = P[0, 2]
    region = ""
    if cy > 10:   region += "N"
    elif cy < -10: region += "S"
    if cx > 10:   region += "E"
    elif cx < -10: region += "W"
    if not region: region = "C"
    print(f"UAV {i}: region={region:<3}  cluster_center=({cx:+.1f},{cy:+.1f})  depot=({depot_x:.2f},{depot_y:.2f},{depot_z:.2f})  waypoints={len(P)-2}")
PY
```

---

## 6. Experiment Complete

All 16 configurations finished. Summary tables saved in `results_csv/`.

| Config | planned_makespan_s (mean) | mission_makespan_s (mean) | dist_m (mean) | pairwise_m (mean) |
|--------|--------------------------|--------------------------|---------------|-------------------|
| 2-UAV 30pt | — | 128.4 | 283.965 | 1.715 |
| 2-UAV 60pt | — | 164.4 | 370.698 | 2.476 |
| 2-UAV 90pt | — | 211.6 | 434.885 | 0.839 |
| 2-UAV 120pt | — | 268.7 | 450.477 | 1.392 |
| 3-UAV 30pt | — | 86.580 | 302.686 | 1.692 |
| 3-UAV 60pt | — | 125.610 | 338.703 | 1.559 |
| 3-UAV 90pt | — | 156.213 | 381.681 | 1.510 |
| 3-UAV 120pt | — | 184.028 | 438.334 | 1.952 |
| 4-UAV 30pt | — | 77.011 | 349.888 | 1.729 |
| 4-UAV 60pt | — | 122.442 | 353.861 | 2.938 |
| 4-UAV 90pt | — | 150.932 | 386.388 | 2.853 |
| 4-UAV 120pt | — | 185.592 | 435.014 | 1.494 |
| 5-UAV 30pt | — | 76.641 | 387.595 | 1.497 |
| 5-UAV 60pt | 63.537 | 104.786 | 393.271 | 1.941 |
| 5-UAV 90pt | 60.312 | 102.708 | 419.607 | 1.907 |
| 5-UAV 120pt | 80.107 | 174.459 | 469.238 | 1.938 |

---

## 7. Benchmarking Phase — NEXT (starting 2026-06-27)

All 16 DECK-GA configurations are complete. The next phase benchmarks DECK-GA results against two baseline algorithms across the same 16 configs (2/3/4/5 UAVs × 30/60/90/120 waypoints).

### Algorithms to benchmark

| Algorithm | Script | Description |
|-----------|--------|-------------|
| Traditional GA Divide & Conquer | `DEGA_Divide & Conquer/Traditional_GA_Divide & Conquer.py` | Divide-and-conquer GA without DCKmeans clustering |
| Classical KMeans DEGA | `Classical_Kmeans.py` + `run_kmeans_GA_10_times.py` | Standard KMeans clustering + GA path planning |

### Terminal command files (Traditional GA D&C) — already exist
```
experiment_commands/antarctica_deckga/traditional_ga_divide_conquer_{N}uavs_{WPTS}_waypoints_terminal_command.txt
```
Example for 2-UAV 30pt:
`traditional_ga_divide_conquer_2uavs_30_waypoints_terminal_command.txt`

### Resume guide

**Step 1 — Start with Traditional GA D&C, 2-UAV 30-waypoint:**
```bash
cd ~/Documents/GitHub/ROS2_MultiUAV_3D
git status   # confirm on branch: antarctica
cat experiment_commands/antarctica_deckga/traditional_ga_divide_conquer_2uavs_30_waypoints_terminal_command.txt
```

**Step 2 — Follow same pipeline as DECK-GA:**
- Run planner 10x (adapt script from `run_deckga_10x_` pattern)
- Run 10x Gazebo flights
- Generate summary table
- Commit + push

**Step 3 — Repeat for all 16 configs, then repeat for Classical KMeans DEGA.**

### Benchmark completion tracker
```
Traditional GA D&C:   2-UAV: 30 ✓  60 ✓  90 ✓  120 ✓
                      3-UAV: 30 ✓  60 ✓  90 ✓  120 ✓
                      4-UAV: 30 ✓  60 ✓  90 ✓  120 ✓
                      5-UAV: 30 ✓  60 ✓  90 ✓  120 ✓  ← ALL DONE

Classical KMeans DEGA: 2-UAV: 30 ✓  60 ✓  90 ✓  120 ✓
                       3-UAV: 30 ✓  60 ✓  90 ✓  120 ✓
                       4-UAV: 30 ✓  60 ✓  90 ✓  120 ✓
                       5-UAV: 30 ✓  60 _  90 _  120 _   ← NEXT (5-UAV 60pt)
```

### Traditional GA D&C Results (completed configs)

| Config | mean makespan (s) | mean total dist (m) | mean min pairwise (m) |
|--------|:-----------------:|:-------------------:|:---------------------:|
| 2-UAV 30pt  | 143.6  | 363.6   | 1.391 |
| 2-UAV 60pt  | 253.5  | 671.7   | 1.021 |
| 2-UAV 90pt  | 410.9  | 1172.3  | 1.147 |
| 2-UAV 120pt | 705.6  | 1668.5  | 1.540 |
| 3-UAV 30pt  | 101.1  | 380.6   | 0.695 |
| 3-UAV 60pt  | 161.0  | 575.5   | 1.195 |
| 3-UAV 90pt  | 210.9  | 723.5   | 1.173 |
| 3-UAV 120pt | 294.9  | 1047.0  | 1.085 |
| 4-UAV 30pt  | 95.399 | 416.482 | 0.593 |
| 4-UAV 60pt  | 143.237| 598.879 | 0.537 |
| 4-UAV 90pt  | 189.415| 704.942 | 1.719 |
| 4-UAV 120pt | 208.385| 910.494 | 0.627 |
| 5-UAV 30pt  | 87.539 | 515.159 | 0.973 |
| 5-UAV 60pt  | 99.616 | 590.741 | 0.498 |
| 5-UAV 90pt  | 120.393| 682.158 | 0.754 |
| 5-UAV 120pt | 152.297| 813.689 | 0.620 |

### Classical KMeans DEGA Results (completed configs)

| Config | mean planned_makespan (s) | mean mission_makespan (s) | mean total dist (m) | mean min pairwise (m) |
|--------|:-------------------------:|:-------------------------:|:-------------------:|:---------------------:|
| 2-UAV 30pt  | 83.718 | 115.390 | 277.580 | 0.740 |
| 2-UAV 60pt  | 98.481 | 165.424 | 351.167 | 2.482 |
| 2-UAV 90pt  | 110.569 | 211.904 | 383.048 | 2.298 |
| 2-UAV 120pt | 176.615 | 332.326 | 497.226 | 0.890 |
| 3-UAV 30pt  |  58.644 |  83.163 | 311.108 | 1.639 |
| 3-UAV 60pt  |  81.354 | 141.151 | 383.197 | 0.844 |
| 3-UAV 90pt  |  84.000 | 154.145 | 402.275 | 0.800 |
| 3-UAV 120pt |  86.866 | 181.528 | 409.645 | 1.903 |
| 4-UAV 30pt  |  55.156 |  80.223 | 355.840 | 0.377 |
| 4-UAV 60pt  |  68.317 | 119.690 | 392.980 | 1.281 |
| 4-UAV 90pt  |  78.303 | 153.452 | 427.912 | 0.887 |
| 4-UAV 120pt |  79.417 | 160.845 | 453.800 | 0.080 |
| 5-UAV 30pt  |  49.442 |  76.570 | 395.296 | 0.122 |

---

## 8. Key Files Reference

| Purpose | File |
|---|---|
| 5-UAV world config | `aerostack_examples/.../config/world_swarm_5.yaml` |
| 5-UAV 30pt DECK-GA script | `run_deckga_10x_5uav_30pts.sh` |
| 5-UAV 30pt flight script | `run_data_runs_uav5_30pts.sh` |
| Summary table generator | `make_summary_table.py {WPTS} {N}` |
| Results CSVs | `results_antarctica_csv/` |
| Summary tables | `results_csv/summary_table_uav{N}_{WPTS}pts.txt` |

---

## 8. Verification Habit — Per-Run Health Check

**GOOD run:** pairwise max climbs well past 40 m AND total executed distance ≈ planner's expected total.

**FAILED run:** pairwise max stays flat and low (< 5 m) — drones never separated.

```bash
python3 make_summary_table.py <waypoints> <uavs>
# or inspect latest pairwise CSV:
ls -t results_antarctica_csv/*pairwise_distance_summary* | head -1 | xargs cat
```
