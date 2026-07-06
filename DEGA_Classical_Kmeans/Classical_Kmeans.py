import numpy as np
import pickle

# Helper function to calculate Euclidean distance (Pythagoras)
def calculate_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid)**2))

# Manual KMeans clustering function
def manual_kmeans_clustering(points, num_uavs, max_iter=300):
    np.random.seed(42)
    centroids = points[np.random.choice(points.shape[0], num_uavs, replace=False)]
    
    for iteration in range(max_iter):
        clusters = [[] for _ in range(num_uavs)]
        for point in points:
            distances = [calculate_distance(point, centroid) for centroid in centroids]
            closest_centroid = np.argmin(distances)
            clusters[closest_centroid].append(point)

        new_centroids = np.array([np.mean(cluster, axis=0) if len(cluster) > 0 else centroids[i] for i, cluster in enumerate(clusters)])
        if np.allclose(new_centroids, centroids):
            print(f"Converged after {iteration+1} iterations")
            break
        centroids = new_centroids

    return clusters, centroids

# Save KMeans results to a file
def save_kmeans_results(clusters, centroids, filename="kmeans_output.pkl"):
    with open(filename, "wb") as f:
        pickle.dump({"clusters": clusters, "centroids": centroids}, f)
    print(f"KMeans results saved to {filename}")
