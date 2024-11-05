import random
import numpy as np


class BinPackingProblem:
    def __init__(self, num_items, num_bins, weights):
        self.num_items = num_items
        self.num_bins = num_bins
        self.weights = weights


# Random seeds
def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


# Initialize pheromones for each edge in the construction graph
def initialize_pheromones(num_items, num_bins):
    pheromones = {}
    for item_index in range(num_items):
        for bin_index in range(num_bins):
            if item_index == 0:
                # First node
                pheromones[("S", f"{item_index}, {bin_index}")] = np.random.rand()
            else:
                for prev_bin_index in range(num_bins):
                    pheromones[
                        (
                            f"{item_index - 1}, {prev_bin_index}",
                            f"{item_index}, {bin_index}")] = np.random.rand()
            if item_index == num_items - 1:
                pheromones[(f"{item_index}, {bin_index}", "E")] = np.random.rand()
    return pheromones


# Generate paths for each ant
def generate_ant_paths(problem, pheromones, num_ants):
    paths = np.zeros((num_ants, problem.num_items), dtype=int)
    for ant in range(num_ants):
        current_node = "S"

        for item in range(problem.num_items):
            pheromone_levels = np.array(
                [pheromones[(current_node, f"{item}, {bin_index}")] for bin_index in
                 range(problem.num_bins)])

            # Only accept paths with positive pheromone levels
            valid_indices = pheromone_levels > 0

            if not np.any(valid_indices):
                # To avoid -> ValueError: 'a' cannot be empty unless no samples are taken
                valid_indices = np.ones(problem.num_bins, dtype=bool)

            valid_pheromone_levels = pheromone_levels[valid_indices]
            total_pheromone = np.sum(valid_pheromone_levels)

            if total_pheromone == 0:
                probabilities = np.ones(len(valid_indices)) / len(valid_indices)
            else:
                probabilities = valid_pheromone_levels / total_pheromone

            chosen_bin = np.random.choice(np.arange(problem.num_bins)[valid_indices], p=probabilities)
            paths[ant, item] = chosen_bin
            current_node = f"{item}, {chosen_bin}"

    return paths


# Calculate the fitness based on the CA specification
def calculate_fitness(problem, paths):
    bin_weights = np.zeros((paths.shape[0], problem.num_bins))
    for ant in range(paths.shape[0]):
        for item in range(problem.num_items):
            bin_index = paths[ant, item]
            bin_weights[ant, bin_index] += problem.weights[item]
    fitnesses = np.max(bin_weights, axis=1) - np.min(bin_weights, axis=1)
    return fitnesses


# Update pheromones based on the ant paths and fitnesses
def update_pheromones(pheromones, paths, fitnesses):
    for ant in range(paths.shape[0]):
        current_node = "S"
        for item in range(paths.shape[1]):
            bin_index = paths[ant, item]
            next_node = f"{item}, {bin_index}"
            pheromones[(current_node, next_node)] += 100 / fitnesses[ant]
            current_node = next_node
        pheromones[(current_node, "E")] += 100 / fitnesses[ant]


# Evaporate pheromones by a fixed rate
def evaporate_pheromones(pheromones, evaporation_rate):
    for edge in pheromones:
        pheromones[edge] *= evaporation_rate


def ant_colony_optimization(problem, num_ants, evaporation_rate, max_evaluations):
    # Generate a random seed
    seed_value = random.randint(0, 10000)
    print(f"Using seed value: {seed_value}")
    set_random_seed(seed_value)

    pheromones = initialize_pheromones(problem.num_items, problem.num_bins)
    best_fitness = float('inf')
    best_path = None
    evaluations = 0

    while evaluations < max_evaluations:
        paths = generate_ant_paths(problem, pheromones, num_ants)
        fitnesses = calculate_fitness(problem, paths)

        min_fitness_index = np.argmin(fitnesses)
        if fitnesses[min_fitness_index] < best_fitness:
            best_fitness = fitnesses[min_fitness_index]
            best_path = paths[min_fitness_index]
            print('FITNESS: ', best_fitness)
        update_pheromones(pheromones, paths, fitnesses)
        evaporate_pheromones(pheromones, evaporation_rate)
        evaluations += num_ants
    best_fitness_last_evaluation = np.min(fitnesses)

    return best_fitness, best_path, best_fitness_last_evaluation


def print_bin_contents(problem, best_path):
    bin_contents = [[] for _ in range(problem.num_bins)]
    for item_index, bin_index in enumerate(best_path):
        bin_contents[bin_index].append(item_index)
    for bin_index, items in enumerate(bin_contents):
        print(f"Bin {bin_index} contains items {items}")


def print_pheromone_levels(pheromones, current_node):
    print("\n***NODES***")
    for edge, level in pheromones.items():
        if edge[0] == current_node:
            print(f"Edge {edge}: {level:.2f}")


# Our two Bin Packing Problems
BPP1 = BinPackingProblem(500, 10, list(range(1, 501)))
BPP2 = BinPackingProblem(500, 50, [(i ** 2) / 2 for i in range(1, 501)])

# Parameters for the experiments
experiments = [
    (100, 0.90),
    (100, 0.60),
    (10, 0.90),
    (10, 0.60)
]

problems = [BPP1, BPP2]
for problem in problems:
    print(f"\nRunning experiments on problem with {problem.num_items} items and {problem.num_bins} bins")

    best_path = None
    best_fitness = float('inf')
    best_fitnesses = []

    for num_ants, evaporation_rate in experiments:
        print(f"\nExperiment with p = {num_ants} and e = {evaporation_rate}")

        for trial in range(5):
            print(f"\nTrial {trial + 1} of 5")
            current_fitness, current_path, best_fitness_last_evaluation = ant_colony_optimization(
                problem, num_ants=num_ants, evaporation_rate=evaporation_rate, max_evaluations=10000)

            print(f"BPP1 current fitness: {current_fitness}")

            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_path = current_path

            best_fitnesses.append(best_fitness_last_evaluation)

        print(f"Best fitness for p = {num_ants}, e = {evaporation_rate}: {best_fitness}")

    print(f"\nBest fitness: {best_fitness} and best path: {best_path}")
    for i, fitness in enumerate(best_fitnesses):
        print(f"Best fitness of the last evaluation for trial {i + 1}: {fitness}")

    # Calculate and print additional statistics
    average = np.mean(best_fitnesses)
    median = np.median(best_fitnesses)
    std_dev = np.std(best_fitnesses)
    print(f"\nStatistics for problem with {problem.num_items} items and {problem.num_bins} bins:")
    print(f"Average fitness: {average}")
    print(f"Median fitness: {median}")
    print(f"Standard deviation of fitness: {std_dev}")
    print(f"Best fitness: {best_fitness}")