# generate_input_for_DECKGA_and_QuickNav.py

import numpy as np
import pickle

def generate_input(num_waypoints=30, num_obstacles=100,
                   points_file="points_xyz_30.pkl",
                   obstacles_file="obstacles_xyz_100.pkl"):
    np.random.seed(42)  # Reproducibility

    # Step 1: Generate 3D waypoints
    waypoints = np.random.rand(num_waypoints, 3) * 100

    # Step 2: Generate obstacle centers BETWEEN waypoints
    obstacle_xyz = []
    attempts = 0
    max_attempts = 2000
    while len(obstacle_xyz) < num_obstacles and attempts < max_attempts:
        idx1, idx2 = np.random.choice(len(waypoints), size=2, replace=False)
        p1, p2 = waypoints[idx1], waypoints[idx2]
        alpha = np.random.uniform(0.3, 0.7)
        mid_point = (1 - alpha) * p1 + alpha * p2

        # Ensure obstacle is not too close to any waypoint or other obstacle
        if (
            np.all(np.linalg.norm(mid_point - waypoints, axis=1) > 5)
            and (len(obstacle_xyz) == 0 or np.all(np.linalg.norm(mid_point - np.array(obstacle_xyz), axis=1) > 5))
        ):
            obstacle_xyz.append(mid_point)
        attempts += 1

    obstacle_xyz = np.array(obstacle_xyz)

    # Step 3: Random obstacle sizes between 1 and 5
    obs_size = np.round(np.random.uniform(1.0, 5.0, size=len(obstacle_xyz)), 2)

    # Step 4: Save to pickle files
    with open(points_file, "wb") as f1:
        pickle.dump(waypoints, f1)

    with open(obstacles_file, "wb") as f2:
        pickle.dump({"obstacle_xyz": obstacle_xyz, "obs_size": obs_size}, f2)

    print("\n✅ Generated and saved:")
    print(f" - {len(waypoints)} waypoints → {points_file}")
    print(f" - {len(obstacle_xyz)} obstacles → {obstacles_file}")

if __name__ == "__main__":
    generate_input()