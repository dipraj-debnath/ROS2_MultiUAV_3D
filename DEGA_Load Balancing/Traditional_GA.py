import numpy as np

def basic_ga_path_planning(cluster_points, population_size=50, num_iterations=100000):
    """
    Basic Genetic Algorithm to optimize the path for a given cluster of waypoints.
    
    :param cluster_points: Waypoints to visit.
    :param population_size: Number of individuals in the population.
    :param num_iterations: Number of iterations to evolve the population.
    :return: Optimized path.
    """
    num_points = len(cluster_points)
    if num_points == 0:
        return np.array([])  # Return an empty array for empty clusters
    
    # Randomly initialize the population
    population = [np.random.permutation(num_points) for _ in range(population_size)]
    best_path = None
    best_distance = float('inf')
    
    for _ in range(num_iterations):
        # Calculate fitness
        fitness = np.array([1 / np.sum(np.linalg.norm(cluster_points[ind[:-1]] - cluster_points[ind[1:]], axis=1)) for ind in population])
        
        # Select parents based on fitness
        fitness_sum = fitness.sum()
        if fitness_sum == 0:
            print("Warning: Fitness sum is zero. Terminating early.")
            break
        parents_indices = np.random.choice(np.arange(population_size), size=population_size, p=fitness / fitness_sum)
        parents = [population[i] for i in parents_indices]
        
        # Crossover and mutation
        next_generation = []
        for i in range(0, population_size, 2):
            if i + 1 < population_size:
                p1, p2 = parents[i], parents[i + 1]
                split = np.random.randint(1, num_points)
                child1 = np.concatenate((p1[:split], [x for x in p2 if x not in p1[:split]]))
                child2 = np.concatenate((p2[:split], [x for x in p1 if x not in p2[:split]]))
                next_generation.extend([child1, child2])
            else:
                next_generation.append(parents[i])
        
        # Apply mutation
        for ind in next_generation:
            if np.random.rand() < 0.1:  # Mutation rate
                swap_idx = np.random.choice(num_points, 2, replace=False)
                ind[swap_idx[0]], ind[swap_idx[1]] = ind[swap_idx[1]], ind[swap_idx[0]]
        
        population = next_generation
        
        # Track the best path
        for ind in population:
            dist = np.sum(np.linalg.norm(cluster_points[ind[:-1]] - cluster_points[ind[1:]], axis=1))
            if dist < best_distance:
                best_distance = dist
                best_path = cluster_points[ind]
    
    return np.vstack((best_path, best_path[0])) if best_path is not None else np.array([])
