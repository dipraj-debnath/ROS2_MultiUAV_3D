import numpy as np
import pickle

def generate_and_save_points(num_points=100, filename="points_100.pkl"):
    """
    Generate random 3D waypoints and save them to a file.

    :param num_points: Number of 3D points to generate.
    :param filename: Name of the file to save the points.
    """
    np.random.seed(42)  # Ensures reproducibility
    points = np.random.rand(num_points, 3) * 100  # Scale points to [0, 100]
    with open(filename, "wb") as f:
        pickle.dump(points, f)
    print(f"Generated {num_points} points and saved to {filename}")

if __name__ == "__main__":
    generate_and_save_points()


