import numpy as np

# GA Path Planning function (your original code)
def ga_3d_pathplanning(xyz):
    # Initialize the variables
    N = xyz.shape[0]  # Number of points
    popSize = 8       # Population size
    numIter = int(1e5)  # Number of iterations

    # Generate distance matrix in 3D
    a = np.meshgrid(np.arange(N), np.arange(N))
    dmat = np.round(np.sqrt(np.sum((xyz[a[0], :] - xyz[a[1], :])**2, axis=2)))

    # Sanity checks for population size
    popSize = 4 * int(np.ceil(popSize / 4))
    numIter = max(1, round(numIter))

    # Initialize population
    pop = np.zeros((popSize, N), dtype=int)
    pop[0, :] = np.arange(N)  # Start with sequential route
    for k in range(1, popSize):
        pop[k, :] = np.random.permutation(N)  # Random routes for the population

    # GA variables
    globalMin = np.inf
    distHistory = np.zeros(numIter)

    for iter in range(numIter):
        totalDist = np.zeros(popSize)

        # Calculate total distance for each member of the population
        for p in range(popSize):
            d = dmat[pop[p, -1], pop[p, 0]]  # Closed path
            for k in range(1, N):
                d += dmat[pop[p, k - 1], pop[p, k]]
            totalDist[p] = d

        minDist = np.min(totalDist)
        index = np.argmin(totalDist)
        distHistory[iter] = minDist

        if minDist < globalMin:
            globalMin = minDist
            optRoute = pop[index, :]

        # Genetic Algorithm Operators: mutation and crossover
        randomOrder = np.random.permutation(popSize)
        newPop = np.zeros((popSize, N), dtype=int)
        for p in range(0, popSize, 4):
            rtes = pop[randomOrder[p:p+4], :]
            dists = totalDist[randomOrder[p:p+4]]
            idx = np.argmin(dists)
            bestOf4Route = rtes[idx, :]
            routeInsertionPoints = np.sort(np.random.randint(0, N, size=2))
            I, J = routeInsertionPoints

            # Mutation operations (flip, swap, slide)
            tmpPop = np.zeros((4, N), dtype=int)
            tmpPop[0, :] = bestOf4Route

            # Flip mutation
            if I < J:
                tmpPop[1, I:J+1] = bestOf4Route[I:J+1][::-1]
            elif I > J:
                tmpPop[1, J:I+1] = bestOf4Route[J:I+1][::-1]
            else:
                tmpPop[1, :] = bestOf4Route

            # Swap mutation
            tmpPop[2, :] = bestOf4Route.copy()
            tmpPop[2, [I, J]] = bestOf4Route[[J, I]]

            # Slide mutation
            tmpPop[3, :] = bestOf4Route.copy()
            tmpPop[3, I:J+1] = np.roll(bestOf4Route[I:J+1], shift=-1)

            # Ensure population integrity: No duplicates and valid route
            for i in range(4):
                if len(np.unique(tmpPop[i, :])) != N:
                    tmpPop[i, :] = np.random.permutation(N)

            newPop[p:p+4, :] = tmpPop
        pop = newPop  # Update the population

    # Final optimized route
    final_route = xyz[optRoute, :]
    start_point = np.where(np.all(final_route == xyz[0], axis=1))[0][0]
    rearranged_route = np.concatenate((final_route[start_point:], final_route[:start_point+1]))

    # Print GA Input and Output
    print("\n--- GA Input ---")
    print("Waypoints:")
    print(xyz)
    print("\n--- GA Output ---")
    print("Optimized Route:")
    print(rearranged_route)

    return rearranged_route
