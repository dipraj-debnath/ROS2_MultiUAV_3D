# DECK_GA_and_QuickNav_AllInOne_Enhanced.py

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from DCKmeans import manual_kmeans_clustering
from GA_path_planning import ga_3d_pathplanning
from QuickNav_detection_3D import obstacle_detection
from QuickNav_avoidance_3D import obstacle_avoid

def calculate_path_length(path):
    return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))

def draw_cube(ax, center, size, color='cyan', alpha=0.25, edgecolor='k'):
    r = [-size, size]
    verts = np.array([[center[0]+x, center[1]+y, center[2]+z] for x in r for y in r for z in r])
    faces = [
        [verts[0], verts[1], verts[3], verts[2]],
        [verts[4], verts[5], verts[7], verts[6]],
        [verts[0], verts[1], verts[5], verts[4]],
        [verts[2], verts[3], verts[7], verts[6]],
        [verts[1], verts[3], verts[7], verts[5]],
        [verts[0], verts[2], verts[6], verts[4]]
    ]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors=edgecolor, alpha=alpha))

def deduplicate_preserve_order(arr, eps=1e-6):
    result = []
    for p in arr:
        if not any(np.allclose(p, r, atol=eps) for r in result):
            result.append(p)
    return np.array(result)

def quicknav_with_detection(route, obstacle_xyz, obs_size):
    # Like Call_QuickNav_3D.py but returns detected indices!
    max_iterations = 10000
    iteration = 0
    processed_obstacles = []
    route = route.copy()
    detected_indices = set()

    while iteration < max_iterations:
        iteration += 1
        Obstacle_route, new_ob, obs_route = obstacle_detection(route, obstacle_xyz, obs_size, visualize=False)

        # Map detected obstacles to their indices in obstacle_xyz
        detected = []
        for ob in new_ob:
            idxs = np.where(np.all(np.isclose(obstacle_xyz, ob, atol=1e-6), axis=1))[0]
            for idx in idxs:
                detected.append(idx)
                detected_indices.add(idx)

        # Filter already processed
        if processed_obstacles and len(new_ob):
            mask = np.array([
                not any(np.allclose(ob, po, atol=1e-6) for po in processed_obstacles) for ob in new_ob
            ], dtype=bool)
            new_ob = new_ob[mask]
            Obstacle_route = Obstacle_route[mask]
            obs_route = obs_route[mask]

        if len(new_ob) == 0:
            break

        # Remove overlapping
        qq = True
        while qq and len(new_ob) > 1:
            qq = False
            r = len(new_ob)
            j = 0
            while j < r - 1:
                gg = True
                for i in range(min(6, Obstacle_route.shape[1])):
                    if not np.isclose(Obstacle_route[j,i], Obstacle_route[j+1,i], atol=1e-6):
                        gg = False
                        break
                if gg:
                    Obstacle_route = np.delete(Obstacle_route, j+1, axis=0)
                    new_ob = np.delete(new_ob, j+1, axis=0)
                    obs_route = np.delete(obs_route, j+1, axis=0)
                    r -= 1
                    qq = True
                else:
                    j += 1

        matched_sizes = []
        for detected_ob in new_ob:
            idx = np.where((obstacle_xyz == detected_ob).all(axis=1))[0]
            if len(idx) > 0:
                matched_sizes.append(obs_size[idx[0]])
            else:
                matched_sizes.append(1.0)
        matched_sizes = np.array(matched_sizes)

        try:
            Obstacle_avoid_route = obstacle_avoid(Obstacle_route, new_ob, new_ob, matched_sizes)
        except Exception as e:
            break

        Final = route.copy()
        r_obs = len(obs_route)
        index = 0

        for i in range(r_obs):
            rr_final = len(Final)
            start_idx = obs_route[i,0] - 1 + index
            end_idx = obs_route[i,1] - 1 + index

            if start_idx < 0 or end_idx >= rr_final:
                continue

            avoid_path = Obstacle_avoid_route[:,:,i]

            if obs_route[i,0] == 1:
                FF = Final[end_idx+1:,:]
                Final = np.vstack([avoid_path, FF])
                index += 2
            elif (end_idx + 1) >= rr_final:
                Final = np.vstack([Final[:start_idx], avoid_path])
                index += 2
            else:
                part1 = Final[:start_idx]
                part2 = avoid_path
                part3 = Final[end_idx+1:]
                Final = np.vstack([part1, part2, part3])
                index += 2

        processed_obstacles.extend(new_ob.tolist())
        if np.array_equal(route, Final):
            break
        route = Final

    route = deduplicate_preserve_order(route)
    # Ensure closed loop
    if not np.allclose(route[0], route[-1]):
        route = np.vstack([route, route[0]])
    return route, sorted(list(detected_indices))

# ======= MAIN PIPELINE =======
with open("points_xyz_30.pkl", "rb") as f:
    points = pickle.load(f)
with open("obstacles_xyz_100.pkl", "rb") as f:
    obs_data = pickle.load(f)
    obstacle_xyz = obs_data["obstacle_xyz"]
    obs_size = obs_data["obs_size"]
print("\n===== 3D Waypoints (points_xyz_30.pkl) =====")
print(points)
print("\n===== Obstacle Centers (obstacles_xyz_100.pkl) =====")
print(obstacle_xyz)
print("\n===== Obstacle Sizes (obstacles_xyz_100.pkl) =====")
print(obs_size)
print(f"\nTotal waypoints: {len(points)}, Total obstacles: {len(obstacle_xyz)}")

NUM_UAVS = 3
START_POINTS = np.array([[10, 0, 0], [50, 20, 0], [90, 0, 0]])
colors = cm.rainbow(np.linspace(0, 1, NUM_UAVS))

# DCKmeans and DEGA
clusters, centroids = manual_kmeans_clustering(points, NUM_UAVS)
clusters_with_start = [np.vstack([START_POINTS[i], np.array(cluster)]) for i, cluster in enumerate(clusters)]

raw_lengths, dega_lengths, final_lengths = [], [], []
optimized_paths, quicknav_paths = [], []
all_detected_indices = []

for i, cluster in enumerate(clusters_with_start):
    print(f"\n--- UAV {i+1} ---")
    raw_len = calculate_path_length(cluster)
    raw_lengths.append(raw_len)
    print(f"  Raw path length (pre-DEGA): {raw_len:.2f}")

    path = ga_3d_pathplanning(cluster)
    if not np.allclose(path[0], path[-1]):
        path = np.vstack([path, path[0]])
    dega_len = calculate_path_length(path)
    dega_lengths.append(dega_len)
    print(f"  Path length after DEGA: {dega_len:.2f}")
    optimized_paths.append(path)

    safe_path, detected_indices = quicknav_with_detection(path, obstacle_xyz, obs_size)
    final_len = calculate_path_length(safe_path)
    final_lengths.append(final_len)
    quicknav_paths.append(safe_path)
    all_detected_indices.append(detected_indices)
    print(f"  Path length after QuickNav: {final_len:.2f}")
    print(f"  Obstacles detected: {len(detected_indices)}")
    print(f"  Detected obstacle numbers: {detected_indices}")

    # Plot for this UAV
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"UAV {i+1} Path with QuickNav Obstacle Avoidance")
    # Draw all obstacles (number on top if detected)
    for j in range(len(obstacle_xyz)):
        is_detected = j in detected_indices
        draw_cube(ax, obstacle_xyz[j], obs_size[j], color='cyan', alpha=0.25 if not is_detected else 0.7)
        if is_detected:
            ax.text(*obstacle_xyz[j], f"#{j}", color='red', fontsize=12, fontweight='bold', ha='center', va='center')
        else:
            ax.text(*obstacle_xyz[j], f"{j}", color='gray', fontsize=8, ha='center', va='center')
    # Plot path
    ax.plot(safe_path[:, 0], safe_path[:, 1], safe_path[:, 2], color=colors[i], linewidth=3, marker='o', label=f"UAV {i+1} Path")
    # Label start/end and intermediate waypoints
    for k, (x, y, z) in enumerate(safe_path):
        if k == 0 or k == len(safe_path)-1:
            ax.text(x, y, z+3, "Start/End", fontsize=13, color='red', weight='bold')
        else:
            ax.text(x, y, z+2, str(k), fontsize=11, color='black')
    ax.scatter(*safe_path[0], color='red', s=180, edgecolors='black', linewidths=2, label='Start/End')
    ax.scatter(*safe_path[-1], color='red', s=180, edgecolors='black', linewidths=2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()

# Combined plot for all UAVs
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Combined Final UAV Paths with All Obstacles")
for j in range(len(obstacle_xyz)):
    draw_cube(ax, obstacle_xyz[j], obs_size[j])
    ax.text(*obstacle_xyz[j], f"{j}", color='gray', fontsize=9, ha='center', va='center')
for i, path in enumerate(quicknav_paths):
    ax.plot(path[:, 0], path[:, 1], path[:, 2], color=colors[i], marker='o', linewidth=3, label=f"UAV {i+1}")
    ax.scatter(*path[0], color='red', s=180, edgecolors='black', linewidths=2)
    ax.text(*path[0], f"Start/End {i+1}", color='red', weight='bold')
    ax.scatter(*path[-1], color='red', s=180, edgecolors='black', linewidths=2)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.tight_layout()
plt.show()

# Print summary
print("\n--- Path Length Summary ---")
for i in range(NUM_UAVS):
    print(f"UAV {i+1}:")
    print(f"  Raw path length (pre-DEGA): {raw_lengths[i]:.2f}")
    print(f"  After DEGA optimization   : {dega_lengths[i]:.2f}")
    print(f"  After QuickNav avoidance : {final_lengths[i]:.2f}")
    print(f"  Obstacles detected: {len(all_detected_indices[i])}")
    print(f"  Detected obstacle numbers: {all_detected_indices[i]}")

print("\nTotal Path Lengths:")
print(f"  Total Raw (pre-DEGA): {sum(raw_lengths):.2f}")
print(f"  Total After DEGA    : {sum(dega_lengths):.2f}")
print(f"  Total After QuickNav: {sum(final_lengths):.2f}")
print("✅ All-in-one pipeline complete with obstacle detection reporting!")


# ===== Save Output for ROS2 Visualization =====
output_data = {
    "optimized_paths": optimized_paths,       # paths after DEGA
    "quicknav_paths": quicknav_paths,         # final paths after QuickNav
    "obstacle_xyz": obstacle_xyz,
    "obs_size": obs_size,
    "detected_indices": all_detected_indices,
    "raw_lengths": raw_lengths,
    "dega_lengths": dega_lengths,
    "final_lengths": final_lengths
}

output_filename = "deckga_quicknav_output.pkl"
with open(output_filename, "wb") as f:
    pickle.dump(output_data, f)

print(f"\n💾 Results saved to '{output_filename}' for ROS2 visualization.")
print("You can now load and visualize this data in your ROS2 node.")
