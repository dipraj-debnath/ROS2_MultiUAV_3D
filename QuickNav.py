# QuickNav_utils.py

import numpy as np
from QuickNav_detection_3D import obstacle_detection
from QuickNav_avoidance_3D import obstacle_avoid

def apply_quicknav(route, obstacle_xyz, obs_size, max_iterations=10):
    """
    Applies QuickNav obstacle detection and avoidance to a 3D path.

    :param route: np.ndarray, shape (N, 3) - original route
    :param obstacle_xyz: np.ndarray, shape (M, 3) - obstacle centers
    :param obs_size: np.ndarray, shape (M,) - obstacle sizes
    :param max_iterations: int - max iterations for multi-obstacle rerouting
    :return: np.ndarray - final obstacle-avoiding route
    """
    def deduplicate_preserve_order(arr, eps=1e-6):
        result = []
        for p in arr:
            if not any(np.allclose(p, r, atol=eps) for r in result):
                result.append(p)
        return np.array(result)

    iteration = 0
    processed_obstacles = []

    while iteration < max_iterations:
        iteration += 1
        Obstacle_route, new_ob, obs_route = obstacle_detection(route, obstacle_xyz, obs_size, visualize=False)

        if processed_obstacles:
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
            matched_sizes.append(obs_size[idx[0]] if len(idx) > 0 else 1.0)

        try:
            Obstacle_avoid_route = obstacle_avoid(Obstacle_route, new_ob, new_ob, np.array(matched_sizes))
        except Exception:
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
    return route
