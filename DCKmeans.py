import numpy as np
import pickle

def calculate_total_distance(cluster, centroid):
    """
    Calculate the total distance of points in a cluster from the centroid.
    """
    return np.sum(np.linalg.norm(np.array(cluster) - centroid, axis=1))

def initialize_centroids_kmeans_plus_plus(points, num_clusters):
    """
    Initialize centroids using KMeans++ method.
    """
    centroids = [points[np.random.choice(len(points))]]
    for _ in range(1, num_clusters):
        distances = np.min([np.linalg.norm(points - c, axis=1) for c in centroids], axis=0)
        probabilities = distances / np.sum(distances)
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()
        for i, p in enumerate(cumulative_probabilities):
            if r < p:
                centroids.append(points[i])
                break
    return np.array(centroids)

def manual_kmeans_clustering(points, num_clusters, max_iterations=100000, tolerance=1e-4):
    """
    Optimized KMeans Clustering Algorithm for 3D points.

    :param points: Array of 3D points to cluster.
    :param num_clusters: Number of desired clusters.
    :param max_iterations: Maximum number of iterations for convergence.
    :param tolerance: Threshold for centroid movement to determine convergence.
    :return: Clusters (list of points per cluster), Centroids (array of cluster centroids).
    """
    np.random.seed(42)  # For reproducibility

    # Step 1: Initialize centroids using KMeans++ method
    centroids = initialize_centroids_kmeans_plus_plus(points, num_clusters)
    prev_centroids = np.zeros_like(centroids)

    for iteration in range(max_iterations):
        # Step 2: Assign points to the closest centroid
        distances = np.linalg.norm(points[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Step 3: Form clusters
        clusters = [[] for _ in range(num_clusters)]
        for i, label in enumerate(labels):
            clusters[label].append(points[i])

        # Step 4: Update centroids based on balanced clusters
        centroids = []
        for cluster in clusters:
            if len(cluster) > 0:
                centroids.append(np.mean(cluster, axis=0))
            else:
                centroids.append(np.zeros(3))  # If empty, place at origin
        centroids = np.array(centroids)

        # Step 5: Check for convergence
        centroid_shift = np.linalg.norm(centroids - prev_centroids)
        if centroid_shift < tolerance:
            print(f"Converged in {iteration+1} iterations.")
            break
        prev_centroids = centroids.copy()

    # Convert clusters to arrays
    clusters = [np.array(cluster) for cluster in clusters]
    return clusters, centroids

def save_kmeans_results(clusters, centroids, filename="kmeans_output.pkl"):
    """
    Save KMeans clustering results to a file.

    :param clusters: List of clusters.
    :param centroids: Array of cluster centroids.
    :param filename: Output file name.
    """
    with open(filename, "wb") as f:
        pickle.dump({'clusters': clusters, 'centroids': centroids}, f)
    print(f"KMeans results saved to {filename}")
