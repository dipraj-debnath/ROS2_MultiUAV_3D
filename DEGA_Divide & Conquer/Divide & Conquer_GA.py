import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from Divide_And_Conquer import divide_and_conquer_clustering, save_divide_and_conquer_results
from GA_path_planning import ga_3d_pathplanning
import matplotlib
matplotlib.rc('font', family='sans-serif')  # Set the font to sans-serif

# Configuration
#POINTS_FILE = "points_100.pkl"  # Change to "points_100.pkl" to use 100 points

# Function to load saved points
def load_points(filename="points.pkl"):
    """
    Load 3D waypoints from a file.

    :param filename: Name of the file to load points from.
    :return: Array of 3D points.
    """
    with open(filename, "rb") as f:
        points = pickle.load(f)
    print(f"Loaded points from {filename}")
    return points

# Function to calculate total path distance
def calculate_path_distance(path):
    return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))

# Main function to run Divide & Conquer and GA
def main():
    # Step 1: Load saved points
    points = load_points("points.pkl")
    num_uavs = 3

    # Print the input points for Divide & Conquer
    print("\n--- Divide & Conquer Input ---")
    print("Points:")
    print(points)
    print(f"\nNumber of UAVs: {num_uavs}")

    # Step 2: Run Divide & Conquer clustering
    start_time_divide_conquer = time.time()
    clusters, centroids = divide_and_conquer_clustering(points, num_uavs)
    elapsed_time_divide_conquer = time.time() - start_time_divide_conquer

    # Save Divide & Conquer results
    save_divide_and_conquer_results(clusters, centroids)

    # Print Divide & Conquer results
    print("\n--- Divide & Conquer Output ---")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1} Waypoints:")
        print(np.array(cluster))
    print("\nCentroids:")
    print(centroids)
    print(f"Divide & Conquer clustering completed in {elapsed_time_divide_conquer:.2f} seconds.")

    # Step 3: Run GA for each UAV
    start_points = [[10, 0, 0], [40, 40, 0], [80, 0, 0]]
    clusters_with_start = [np.vstack([start_points[i], np.array(cluster)]) for i, cluster in enumerate(clusters)]

    # Calculate total path distances before optimization
    total_distance_before = [calculate_path_distance(cluster) for cluster in clusters_with_start]

    print("\n--- Total Path Distances Before Optimization ---")
    for i, distance in enumerate(total_distance_before):
        print(f"UAV {i + 1}: {distance:.2f}")

    total_before = sum(total_distance_before)
    print(f"\nTotal Distance Before Optimization: {total_before:.2f}")

    optimized_paths = []
    start_time_ga = time.time()

    for i, cluster_points in enumerate(clusters_with_start):
        print(f"\n--- GA Input for UAV {i + 1} ---")
        print("Waypoints:")
        print(cluster_points)
        optimized_path = ga_3d_pathplanning(cluster_points)
        optimized_paths.append(optimized_path)
        print(f"\n--- GA Output for UAV {i + 1} ---")
        print("Optimized Route:")
        print(optimized_path)

    elapsed_time_ga = time.time() - start_time_ga

    # Calculate total path distances after optimization
    total_distance_after = [calculate_path_distance(path) for path in optimized_paths]

    print("\n--- Total Path Distances After Optimization ---")
    for i, distance in enumerate(total_distance_after):
        print(f"UAV {i + 1}: {distance:.2f}")

    total_after = sum(total_distance_after)
    print(f"\nTotal Distance After Optimization: {total_after:.2f}")

    # Calculate and print distance reduction
    print("\n--- Distance Reduction ---")
    for i in range(num_uavs):
        reduction = total_distance_before[i] - total_distance_after[i]
        print(f"UAV {i + 1}: {reduction:.2f} (Reduced by {reduction / total_distance_before[i] * 100:.2f}%)")

    # Calculate and print total path distances
    total_distance_before_sum = sum(total_distance_before)
    total_distance_after_sum = sum(total_distance_after)
    total_distance_reduction_sum = total_distance_before_sum - total_distance_after_sum

     # Print total distances
    print(f"\n--- Total Path Distances ---")
    print(f"Before Optimization: {total_distance_before_sum:.2f}")
    print(f"After Optimization: {total_distance_after_sum:.2f}")
    print(f"Total Distance Reduction: {total_distance_reduction_sum:.2f}")

    # Calculate total distance reduction percentage
    total_reduction_percentage = (total_distance_reduction_sum / total_distance_before_sum) * 100

    # Print total distance reduction with percentage
    print(f"\n--- Total Distance Reduction ---")
    print(f"Total Distance Reduction: {total_distance_reduction_sum:.2f} (Reduced by {total_reduction_percentage:.2f}%)")

    print(f"\nGA optimization completed in {elapsed_time_ga:.2f} seconds.")

    # # Step 4: Plot individual UAV paths
    # for i, path in enumerate(optimized_paths):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection="3d")
    #     ax.plot(path[:, 0], path[:, 1], path[:, 2], "*-", label=f"UAV {i + 1}")
    #     ax.scatter(*start_points[i], c="red", s=100, marker="o", label="Start Point")
    #     ax.set_title(f"Optimized Path for UAV {i + 1}")
    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
    #     ax.set_zlabel("Z")
    #     ax.legend()
    #     plt.show()

    # # Combined plot for all UAVs
    # fig_combined = plt.figure(figsize=(3.5,2.5), dpi= 200)
    # ax_combined = fig_combined.add_subplot(111, projection="3d")
    # colors = cm.rainbow(np.linspace(0, 1, len(optimized_paths)))

    # for i, path in enumerate(optimized_paths):
    #     ax_combined.plot(path[:, 0], path[:, 1], path[:, 2], "*-", label=f"UAV {i + 1}", color=colors[i])
    #     ax_combined.scatter(*start_points[i], c="red", s=100, marker="o", label=f"UAV {i + 1} Start Point")

    # ax_combined.set_title("Optimised UAV Paths", fontsize=9)
    # ax_combined.set_xlabel("X", fontsize=9)
    # ax_combined.set_ylabel("Y", fontsize=9)
    # ax_combined.set_zlabel("Z", fontsize=9)
    # #ax_combined.legend()
    # plt.show()

    # Summary of timing
    print(f"\n--- Summary ---")
    print(f"Divide & Conquer Time: {elapsed_time_divide_conquer:.2f} seconds")
    print(f"GA Time: {elapsed_time_ga:.2f} seconds")
    print(f"Total Time: {elapsed_time_divide_conquer + elapsed_time_ga:.2f} seconds")

if __name__ == "__main__":
    main()
