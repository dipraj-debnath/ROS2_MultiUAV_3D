import pickle
import numpy as np

# Input PKL file
pkl_path = "obstacles_xyz_100.pkl"

# Output TXT file
txt_path = "obstacles_xyz_100.txt"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

obstacle_xyz = np.array(data["obstacle_xyz"])
obs_size = np.array(data["obs_size"])

with open(txt_path, "w") as f:
    for center, radius in zip(obstacle_xyz, obs_size):
        line = f"{center[0]:.3f} {center[1]:.3f} {center[2]:.3f} {radius:.3f}\n"
        f.write(line)

print(f"✅ Saved {len(obstacle_xyz)} obstacles to {txt_path}")
