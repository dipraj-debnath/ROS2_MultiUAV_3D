import numpy as np
import pickle

# Function to calculate the total distance for a UAV given a sequence of points
def calculate_total_distance(points):
    return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))

# Function to perform Load Balancing clustering
def load_balancing_clustering(points, num_uavs):
    """
    Perform Load Balancing clustering on 3D points for multiple UAVs.

    :param points: Array of 3D points.
    :param num_uavs: Number of UAVs.
    :return: Dictionary of UAV assignments and centroids.
    """
    uav_position = np.array([0, 0, 0])  # Common start point for all UAVs
    uav_assignments = {i: [] for i in range(num_uavs)}
    remaining_points = points.copy()

    while len(remaining_points) > 0:
        for uav in range(num_uavs):
            if len(remaining_points) == 0:
                break
            # Calculate the current total distance for the UAV
            current_uav_points = np.array(uav_assignments[uav])
            if current_uav_points.size > 0:
                current_distance = calculate_total_distance(np.vstack([uav_position, current_uav_points]))
            else:
                current_distance = 0

            # Find the nearest point to the current UAV position
            distances = np.linalg.norm(remaining_points - uav_position, axis=1)
            nearest_point_index = np.argmin(distances)
            nearest_point = remaining_points[nearest_point_index]

            # Assign the nearest point to the UAV
            uav_assignments[uav].append(nearest_point)

            # Remove the assigned point from the remaining points
            remaining_points = np.delete(remaining_points, nearest_point_index, axis=0)

    centroids = np.array([np.mean(uav_assignments[uav], axis=0) if len(uav_assignments[uav]) > 0 else uav_position for uav in range(num_uavs)])
    return uav_assignments, centroids

# Function to save clustering results
def save_load_balancing_results(clusters, centroids, filename="load_balancing_output.pkl"):
    """
    Save clustering results to a file.

    :param clusters: Dictionary of UAV assignments.
    :param centroids: Array of centroids.
    :param filename: Name of the file to save the results.
    """
    with open(filename, "wb") as f:
        pickle.dump((clusters, centroids), f)
    print(f"Load Balancing results saved to {filename}")

# Example usage (uncomment to test)
# points = np.random.rand(30, 3) * 100
# num_uavs = 3
# clusters, centroids = load_balancing_clustering(points, num_uavs)
# save_load_balancing_results(clusters, centroids)
