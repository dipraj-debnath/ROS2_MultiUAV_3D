# Modified test_quicknav_avoidance_3D.py
import numpy as np

def obstacle_avoid(Obstacle_route, new_ob, obstacle_xyz, obs_size):
    num_obs = len(new_ob)
    Obstacle_avoid_route = np.zeros((4, 3, num_obs))

    for i in range(num_obs):
        x0, y0, z0 = Obstacle_route[i, 0:3]
        x1, y1, z1 = Obstacle_route[i, 3:6]
        cx, cy, cz = new_ob[i]
        s = obs_size[i]

        if np.isscalar(s):
            sx = sy = sz = s
        elif isinstance(s, (np.ndarray, list, tuple)) and len(s) == 3:
            sx, sy, sz = s
        else:
            sx = sy = sz = s

        # 8 cube corners
        corners = np.array([
            [cx - sx, cy - sy, cz - sz],  # 0
            [cx + sx, cy - sy, cz - sz],  # 1
            [cx + sx, cy + sy, cz - sz],  # 2
            [cx - sx, cy + sy, cz - sz],  # 3
            [cx - sx, cy - sy, cz + sz],  # 4
            [cx + sx, cy - sy, cz + sz],  # 5
            [cx + sx, cy + sy, cz + sz],  # 6
            [cx - sx, cy + sy, cz + sz],  # 7
        ])

        # Faces defined by corner indices (each face: 4 corners)
        faces = {
            "bottom": [0, 1, 2, 3],
            "top":    [4, 5, 6, 7],
            "front":  [0, 1, 5, 4],
            "back":   [2, 3, 7, 6],
            "left":   [0, 3, 7, 4],
            "right":  [1, 2, 6, 5]
        }

        all_paths = []
        for face_indices in faces.values():
            c = [corners[idx] for idx in face_indices]
            # Use the same 8 path combinations as 2D QuickNav:
            all_paths.extend([
                [[x0,y0,z0], c[0], c[1], [x1,y1,z1]],
                [[x0,y0,z0], c[1], c[0], [x1,y1,z1]],
                [[x0,y0,z0], c[2], c[3], [x1,y1,z1]],
                [[x0,y0,z0], c[3], c[2], [x1,y1,z1]],
                [[x0,y0,z0], c[0], c[3], [x1,y1,z1]],
                [[x0,y0,z0], c[3], c[0], [x1,y1,z1]],
                [[x0,y0,z0], c[1], c[2], [x1,y1,z1]],
                [[x0,y0,z0], c[2], c[1], [x1,y1,z1]],
            ])

        # Find the shortest path
        best_path = None
        best_dist = np.inf
        for path in all_paths:
            dist = sum(np.linalg.norm(np.array(path[j+1]) - np.array(path[j])) for j in range(3))
            if dist < best_dist:
                best_dist = dist
                best_path = path

        Obstacle_avoid_route[:, :, i] = best_path

    return Obstacle_avoid_route
