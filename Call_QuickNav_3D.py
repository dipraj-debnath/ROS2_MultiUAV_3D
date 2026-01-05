# test_call_quicknav_3D.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from QuickNav_detection_3D import obstacle_detection
from QuickNav_avoidance_3D import obstacle_avoid

def calculate_distance(route):
    distances = np.zeros(len(route) - 1)
    for i in range(len(route) - 1):
        distances[i] = np.linalg.norm(route[i+1] - route[i])
    return distances

def draw_cube(ax, center, size):
    r = [-size, size]
    verts = np.array([[center[0]+x, center[1]+y, center[2]+z] for x in r for y in r for z in r])
    faces = [
        [verts[0], verts[1], verts[3], verts[2]],
        [verts[4], verts[5], verts[7], verts[6]],
        [verts[0], verts[1], verts[5], verts[4]],
        [verts[2], verts[3], verts[7], verts[6]],
        [verts[1], verts[3], verts[7], verts[5]],
        [verts[0], verts[2], verts[6], verts[4]]
    ]
    ax.add_collection3d(Poly3DCollection(faces, alpha=0.25, facecolors='cyan', edgecolors='k', linewidths=1))

def plot_final_route(route, obstacle_xyz, obs_size):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Draw all obstacle cubes
    for i in range(len(obstacle_xyz)):
        draw_cube(ax, obstacle_xyz[i], obs_size[i])

    # Draw path as thick green line
    ax.plot(route[:,0], route[:,1], route[:,2], color='green', linewidth=3, marker='o',
            markerfacecolor='blue', markeredgewidth=0, label='Route')

    # Draw waypoints as blue spheres
    ax.scatter(route[:,0], route[:,1], route[:,2], color='blue', s=80, label='Waypoints')
    # Start in red, end in yellow
    ax.scatter(route[0,0], route[0,1], route[0,2], color='red', s=120, label='Start', edgecolors='k', linewidths=2, zorder=5)
    ax.scatter(route[-1,0], route[-1,1], route[-1,2], color='gold', s=120, label='End', edgecolors='k', linewidths=2, zorder=5)

    # Numbered labels for each waypoint
    for i, (x, y, z) in enumerate(route):
        ax.text(x, y, z+0.3, str(i), color='black', fontsize=12, ha='center', weight='bold')

    ax.set_xlabel('X', fontsize=13)
    ax.set_ylabel('Y', fontsize=13)
    ax.set_zlabel('Z', fontsize=13)
    ax.set_title('3D QuickNav Route and Obstacle Avoidance', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True)

    try:
        ax.set_box_aspect([np.ptp(route[:,0]), np.ptp(route[:,1]), np.ptp(route[:,2])])
    except Exception:
        pass

    plt.tight_layout()
    plt.show()

def deduplicate_preserve_order(arr, eps=1e-6):
    result = []
    for p in arr:
        if not any(np.allclose(p, r, atol=eps) for r in result):
            result.append(p)
    return np.array(result)

def main():
    xyz = np.array([[3, 5, 2], [6, 25, 3], [24, 24, 4], [21, 4, 2], [3, 5, 2]])
    route = xyz.copy()
    obstacle_xyz = np.array([[3, 10, 1], [5, 20, 2], [15, 24, 4], [22, 10, 5], [10, 4, 3]])
    obs_size = np.array([1.5, 2, 2.7, 3, 1])

    initial_distance = calculate_distance(route)
    sum_ini_distance = np.sum(initial_distance)

    max_iterations = 10
    iteration = 0
    processed_obstacles = []

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        print("Current route:", route)

        Obstacle_route, new_ob, obs_route = obstacle_detection(route, obstacle_xyz, obs_size, visualize=False)
        print(f"Detected {len(new_ob)} obstacles")

        # Filter already processed
        if processed_obstacles:
            mask = np.array([
                not any(np.allclose(ob, po, atol=1e-6) for po in processed_obstacles) for ob in new_ob
            ], dtype=bool)
            new_ob = new_ob[mask]
            Obstacle_route = Obstacle_route[mask]
            obs_route = obs_route[mask]

        if len(new_ob) == 0:
            print("No new obstacles detected")
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
                    print("Removed overlapping obstacle")
                else:
                    j += 1

        print(f"Processing {len(new_ob)} obstacles...")

        matched_sizes = []
        for detected_ob in new_ob:
            idx = np.where((obstacle_xyz == detected_ob).all(axis=1))[0]
            if len(idx) > 0:
                matched_sizes.append(obs_size[idx[0]])
            else:
                matched_sizes.append(1.0)
        matched_sizes = np.array(matched_sizes)

        try:
            Obstacle_avoid_route = obstacle_avoid(Obstacle_route, new_ob, new_ob, matched_sizes)
        except Exception as e:
            print(f"Error in avoidance: {e}")
            break

        Final = route.copy()
        r_obs = len(obs_route)
        index = 0

        for i in range(r_obs):
            rr_final = len(Final)
            start_idx = obs_route[i,0] - 1 + index
            end_idx = obs_route[i,1] - 1 + index

            if start_idx < 0 or end_idx >= rr_final:
                print("Invalid indices - skipping obstacle")
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
            print("Route unchanged - exiting")
            break

        route = Final

    route = deduplicate_preserve_order(route)
    route = np.vstack([route, route[0]])  # Close loop
    final_distance = calculate_distance(route)
    sum_final_distance = np.sum(final_distance)

    print("\nFinal Route Coordinates:")
    for point in route:
        print(tuple(point))

    print(f"\nInitial total distance: {sum_ini_distance:.2f}")
    print(f"Final total distance: {sum_final_distance:.2f}")

    plot_final_route(route, obstacle_xyz, obs_size)

if __name__ == "__main__":
    main()
