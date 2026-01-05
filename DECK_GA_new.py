# DECK_GA_new.py

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from DCKmeans import manual_kmeans_clustering, save_kmeans_results
from GA_path_planning import ga_3d_pathplanning

POINTS_FILE = "points_xyz_30.pkl"
OPTIMIZED_PATHS_FILE = "optimized_paths.pkl"
NUM_UAVS = 3
START_POINTS = np.array([[10, 0, 0], [50, 20, 0], [90, 0, 0]])

# Load waypoints
with open(POINTS_FILE, "rb") as f:
    points = pickle.load(f)

# DCKmeans clustering
clusters, centroids = manual_kmeans_clustering(points, NUM_UAVS)
save_kmeans_results(clusters, centroids)

# Attach start points to clusters
clusters_with_start = [np.vstack([START_POINTS[i], np.array(cluster)]) for i, cluster in enumerate(clusters)]

# GA path planning
optimized_paths = []
for i, cluster in enumerate(clusters_with_start):
    print(f"\n--- GA Input for UAV {i+1} ---")
    print("Waypoints:")
    print(cluster)
    path = ga_3d_pathplanning(cluster)
    # Ensure closed loop (start == end)
    if not np.allclose(path[0], path[-1]):
        path = np.vstack([path, path[0]])
    print(f"--- GA Output for UAV {i+1} ---")
    print("Optimized Route:")
    print(path)
    optimized_paths.append(path)

# Save all optimized paths to file
with open(OPTIMIZED_PATHS_FILE, "wb") as f:
    pickle.dump(optimized_paths, f)
print(f"\n✅ All optimized UAV paths saved to {OPTIMIZED_PATHS_FILE}")

# Optional: Plot all DEGA paths together
colors = cm.rainbow(np.linspace(0, 1, NUM_UAVS))
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_title("DEGA Optimized UAV Paths")
for i, path in enumerate(optimized_paths):
    ax.plot(path[:, 0], path[:, 1], path[:, 2], marker='o', label=f"UAV {i+1}", color=colors[i])
    ax.scatter(*START_POINTS[i], color='red', s=120, edgecolors='black', linewidths=2)
    ax.text(*START_POINTS[i], f"Start {i+1}", color='red', weight='bold')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.tight_layout()
plt.show()
