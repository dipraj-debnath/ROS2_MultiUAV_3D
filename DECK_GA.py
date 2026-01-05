from DCKmeans import manual_kmeans_clustering, save_kmeans_results
from GA_path_planning import ga_3d_pathplanning
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import pickle
import matplotlib
matplotlib.rc('font', family='sans-serif')  # Set the font to sans-serif

# Configuration
POINTS_FILE = "points.pkl"  # Change to "points_100.pkl" to use 100 points

# Load the saved points
def load_points(filename="points_30.pkl"):
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


# Main function to run KMeans and GA
def main():
    # Step 1: Load 3D waypoints
    points = load_points()
    num_points = len(points)
    num_uavs = 3
    plotting = True

    # Print the input points for KMeans
    print("\n--- KMeans Input ---")
    print("Points:")
    print(points)
    print(f"\nNumber of UAVs: {num_uavs}")

    # Step 2: Run KMeans clustering
    start_time_kmeans = time.time()
    clusters, centroids = manual_kmeans_clustering(points, num_uavs)
    elapsed_time_kmeans = time.time() - start_time_kmeans

    # Save KMeans results
    save_kmeans_results(clusters, centroids)

    # Print KMeans results
    print("\n--- KMeans Output ---")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1} Waypoints:")
        print(np.array(cluster))
    print("\nCentroids:")
    print(centroids)
    print(f"KMeans clustering completed in {elapsed_time_kmeans:.2f} seconds.")

    # Plot KMeans clustering results
    if plotting:
        fig_kmeans = plt.figure()
        ax_kmeans = fig_kmeans.add_subplot(111, projection="3d")
        ax_kmeans.set_title("KMeans Clustering")
        colors = cm.rainbow(np.linspace(0, 1, num_uavs))
        for i, cluster in enumerate(clusters):
            cluster_points = np.array(cluster)
            ax_kmeans.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], color=colors[i], label=f"Cluster {i+1}")
        ax_kmeans.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c="black", s=100, marker="x", label="Centroids")
        ax_kmeans.set_xlabel("X")
        ax_kmeans.set_ylabel("Y")
        ax_kmeans.set_zlabel("Z")
        ax_kmeans.legend()
        plt.show()

    # Step 3: Run GA for each UAV
    start_points = [[10, 0, 0], [50, 20, 0], [90, 0, 0]]
    clusters_with_start = [np.vstack([start_points[i], np.array(cluster)]) for i, cluster in enumerate(clusters)]

    # Calculate total path distances before optimization
    total_distance_before = [calculate_path_distance(cluster) for cluster in clusters_with_start]

    print("\n--- Total Path Distances Before Optimization ---")
    for i, distance in enumerate(total_distance_before):
        print(f"UAV {i+1}: {distance:.2f}")

    total_before = sum(total_distance_before)
    print(f"\nTotal Distance Before Optimization: {total_before:.2f}")

    optimized_paths = []
    start_time_ga = time.time()

    for i, cluster_points in enumerate(clusters_with_start):
        print(f"\n--- GA Input for UAV {i+1} ---")
        print("Waypoints:")
        print(cluster_points)
        optimized_path = ga_3d_pathplanning(cluster_points)
        optimized_paths.append(optimized_path)
        print(f"\n--- GA Output for UAV {i+1} ---")
        print("Optimized Route:")
        print(optimized_path)

    elapsed_time_ga = time.time() - start_time_ga
    print(f"\nGA optimization completed in {elapsed_time_ga:.2f} seconds.")

    # Calculate total path distances after optimization
    total_distance_after = [calculate_path_distance(path) for path in optimized_paths]

    print("\n--- Total Path Distances After Optimization ---")
    for i, distance in enumerate(total_distance_after):
        print(f"UAV {i+1}: {distance:.2f}")

    total_after = sum(total_distance_after)
    print(f"\nTotal Distance After Optimization: {total_after:.2f}")

    # Calculate and print distance reduction
    print("\n--- Distance Reduction ---")
    total_reduction = 0
    for i in range(num_uavs):
        reduction = total_distance_before[i] - total_distance_after[i]
        total_reduction += reduction
        print(f"UAV {i+1}: {reduction:.2f} (Reduced by {reduction / total_distance_before[i] * 100:.2f}%)")
    print(f"\nTotal Distance Reduction: {total_reduction:.2f}")
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

    



    # Plot individual UAV paths
    if plotting:
        for i, path in enumerate(optimized_paths):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(path[:, 0], path[:, 1], path[:, 2], "*-", label=f"UAV {i+1}")
            ax.scatter(start_points[i][0], start_points[i][1], start_points[i][2], c="red", s=100, marker="o", label="Start Point")
            ax.set_title(f"Optimized Path for UAV {i+1}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.legend()
            plt.show()

        # Combined plot for all UAVs
        fig_combined = plt.figure(figsize=(3.5,2.5), dpi= 200)
        ax_combined = fig_combined.add_subplot(111, projection="3d")
        colors = cm.rainbow(np.linspace(0, 1, len(optimized_paths)))

        for i, path in enumerate(optimized_paths):
            ax_combined.plot(path[:, 0], path[:, 1], path[:, 2], '*-', label=f"UAV {i+1}", color=colors[i])
            ax_combined.scatter(start_points[i][0], start_points[i][1], start_points[i][2], c="red", s=100, marker="o", label=f"Start Point UAV {i+1}")
        
        ax_combined.set_title("Optimised UAV Paths", fontsize=9)
        ax_combined.set_xlabel("X", fontsize=9)
        ax_combined.set_ylabel("Y", fontsize=9)
        ax_combined.set_zlabel("Z", fontsize=9)
        #ax_combined.legend()
        plt.show()

    # Summary of timing
    print(f"\n--- Summary ---")
    print(f"KMeans Time: {elapsed_time_kmeans:.2f} seconds")
    print(f"GA Time: {elapsed_time_ga:.2f} seconds")
    print(f"Total Time: {elapsed_time_kmeans + elapsed_time_ga:.2f} seconds")


if __name__ == "__main__":
    main()
