import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random


class BinPackingProblem:
    def __init__(self, num_items, num_bins, weights):
        self.num_items = num_items
        self.num_bins = num_bins
        self.weights = weights


# Function to seed the random number generator
def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def initialize_pheromones(num_items, num_bins):
    pheromones = {}
    for item_index in range(num_items):
        for bin_index in range(num_bins):
            if item_index == 0:
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


def generate_ant_paths(problem, pheromones, num_ants, evaporation_rate):
    paths = np.zeros((num_ants, problem.num_items), dtype=int)
    for ant in range(num_ants):
        current_node = "S"
        bin_weights = np.zeros(problem.num_bins)  # Track weights in each bin for heuristic

        for item_index in range(problem.num_items):
            pheromone_levels = np.array(
                [pheromones[(current_node, f"{item_index}, {bin_index}")] for bin_index in
                 range(problem.num_bins)])

            valid_indices = pheromone_levels > 0

            if not np.any(valid_indices):
                # To avoid ValueError: 'a' cannot be empty unless no samples are taken
                valid_indices = np.ones(problem.num_bins, dtype=bool)

            valid_pheromone_levels = pheromone_levels[valid_indices]
            total_pheromone = np.sum(valid_pheromone_levels)

            if total_pheromone == 0:
                probabilities = np.ones(len(valid_indices)) / len(valid_indices)
            else:
                probabilities = valid_pheromone_levels / total_pheromone

            chosen_bin = np.random.choice(np.arange(problem.num_bins)[valid_indices], p=probabilities)
            # print_pheromone_levels(pheromones, current_node)
            paths[ant, item_index] = chosen_bin
            current_node = f"{item_index}, {chosen_bin}"
            # print(f"Item {item_index} dropped in bin {chosen_bin} with pheromone level {pheromone_levels[chosen_bin]}")

        # Update pheromones after each ant completes its path
        # fitness = calculate_fitness(problem, paths[ant:ant + 1])
        # update_pheromones(pheromones, paths[ant:ant + 1], fitness)
        # evaporate_pheromones(pheromones, evaporation_rate)

    return paths


def calculate_fitness(problem, paths):
    bin_weights = np.zeros((paths.shape[0], problem.num_bins))
    for ant in range(paths.shape[0]):
        for item_index in range(problem.num_items):
            bin_index = paths[ant, item_index]
            bin_weights[ant, bin_index] += problem.weights[item_index]
    fitnesses = np.max(bin_weights, axis=1) - np.min(bin_weights, axis=1)
    return fitnesses


def update_pheromones(pheromones, paths, fitnesses):
    for ant_index in range(paths.shape[0]):
        current_node = "S"
        for item_index in range(paths.shape[1]):
            bin_index = paths[ant_index, item_index]
            next_node = f"{item_index}, {bin_index}"
            pheromones[(current_node, next_node)] += 100 / fitnesses[ant_index]
            # print(f"Edge {current_node} -> {next_node} updated to {pheromones[(current_node, next_node)]:.2f}")
            current_node = next_node
        pheromones[(current_node, "E")] += 100 / fitnesses[ant_index]


def evaporate_pheromones(pheromones, evaporation_rate):
    for edge in pheromones:
        pheromones[edge] *= evaporation_rate


def ant_colony_optimization(problem, num_ants, evaporation_rate, max_evaluations):
    seed_value = random.randint(0, 10000)  # Generate a random seed
    print(f"Using seed value: {seed_value}")
    set_random_seed(seed_value)

    pheromones = initialize_pheromones(problem.num_items, problem.num_bins)
    best_fitness = float('inf')
    best_path = None
    evaluations = 0

    # Additional variables to record
    fitness_history = []  # Record fitness at each iteration
    path_lengths = []  # Optionally record path lengths

    while evaluations < max_evaluations:
        paths = generate_ant_paths(problem, pheromones, num_ants, evaporation_rate)
        fitnesses = calculate_fitness(problem, paths)
        # print('Fitness for evaluation: ', evaluations, ' is: ', fitnesses)

        # Record current fitnesses and pheromone levels
        fitness_history.append(np.min(fitnesses))
        path_lengths.append(np.mean([np.sum(path) for path in paths]))  # Average path length

        min_fitness_index = np.argmin(fitnesses)
        if fitnesses[min_fitness_index] < best_fitness:
            best_fitness = fitnesses[min_fitness_index]
            best_path = paths[min_fitness_index]
            print('FITNESS: ', best_fitness)
        update_pheromones(pheromones, paths, fitnesses)
        evaporate_pheromones(pheromones, evaporation_rate)
        evaluations += num_ants
        # print_pheromone_levels(pheromones)  # Debugging pheromone levels
    best_fitness_last_evaluation = np.min(fitnesses)

    return best_fitness, best_path, fitness_history, path_lengths, best_fitness_last_evaluation


def plot_construction_graph(problem):
    G = nx.DiGraph()
    G.add_node("S")
    G.add_node("E")

    middle_pos = (problem.num_bins + 1) / 2
    pos = {"S": (0, middle_pos), "E": (problem.num_items + 1, middle_pos)}

    for item_index in range(problem.num_items):
        for bin_index in range(problem.num_bins):
            bin_node = f"{item_index + 1}, {bin_index + 1}"
            G.add_node(bin_node)
            pos[bin_node] = (item_index + 1, bin_index + 1)

            if item_index == 0:
                G.add_edge("S", bin_node)
            else:
                for prev_bin_index in range(problem.num_bins):
                    prev_bin_node = f"{item_index}, {prev_bin_index + 1}"
                    G.add_edge(prev_bin_node, bin_node)

            if item_index == problem.num_items - 1:
                G.add_edge(bin_node, "E")

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', font_size=8, font_weight='bold', node_size=2000)
    plt.title("Construction Graph for Bin-Packing Problem")
    plt.show()


# Write a function which print what items are in each bin
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


"""
def plot_construction_graph(problem, pheromones, best_path):
    G = nx.DiGraph()
    G.add_node("S")  # Start node
    G.add_node("E")  # End node

    middle_pos = (problem.num_bins + 1) / 2
    pos = {"S": (0, middle_pos), "E": (problem.num_items + 1, middle_pos)}  # Set S and E at the middle height

    # Add nodes and edges for each item and bin
    for item_index in range(problem.num_items):
        for bin_index in range(problem.num_bins):
            bin_node = f"{item_index}, {bin_index}"
            G.add_node(bin_node)
            pos[bin_node] = (item_index, bin_index)

            # Connect start node to the first item nodes
            if item_index == 0:
                G.add_edge("S", bin_node, pheromone=pheromones[("S", bin_node)])
            else:
                for prev_bin_index in range(problem.num_bins):
                    prev_bin_node = f"{item_index - 1}, {prev_bin_index}"
                    G.add_edge(prev_bin_node, bin_node)

            # Connect to the end node only after the last item
            if item_index == problem.num_items - 1:
                G.add_edge(bin_node, "E")

    # Highlight the best path
    current_node = "S"
    for item_index, bin_index in enumerate(best_path):
        next_node = f"{item_index + 1}, {bin_index + 1}"
        G.edges[current_node, next_node]['color'] = 'red'
        G.edges[current_node, next_node]['width'] = 3
        current_node = next_node
    G.edges[current_node, "E"]['color'] = 'red'
    G.edges[current_node, "E"]['width'] = 3

    plt.figure(figsize=(12, 8))
    edges = G.edges(data=True)
    colors = [edge[2].get('color', 'black') for edge in edges]
    widths = [edge[2].get('width', 1) for edge in edges]
    nx.draw(G, pos, with_labels=True, node_color='lightblue', font_size=8, font_weight='bold', node_size=2000,
            edge_color=colors, width=widths, arrows=True)
    plt.title("Construction Graph for Bin-Packing Problem")
    plt.show()
"""

BPP1 = BinPackingProblem(500, 10, list(range(1, 501)))
BPP2 = BinPackingProblem(500, 50, [(i ** 2) / 2 for i in range(1, 501)])

# Parameters for the experiments
experiments = [
    (100, 0.90),
    (100, 0.60),
    (10, 0.90),
    (10, 0.60)
]

# p = 10 and e = 0.90 -> 400ish fitness
print("Running experiments on BPP1")
problems = [BPP2]

for problem in problems:
    print(f"\nRunning experiments on problem with {problem.num_items} items and {problem.num_bins} bins")

    best_path = None
    best_fitness = float('inf')
    best_fitnesses = []
    all_fitness_histories = []

    for num_ants, evaporation_rate in experiments:
        print(f"\nExperiment with p = {num_ants} and e = {evaporation_rate}")

        for trial in range(5):  # Run 5 trials
            print(f"\nTrial {trial + 1} of 5")
            current_fitness, current_path, fitness_history, path_lengths, best_fitness_last_evaluation = ant_colony_optimization(
                problem, num_ants=num_ants, evaporation_rate=evaporation_rate, max_evaluations=10000)

            print(f"BPP1 current fitness: {current_fitness}")

            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_path = current_path

            best_fitnesses.append(best_fitness_last_evaluation)
            all_fitness_histories.append(fitness_history)

        print(f"Best fitness for p = {num_ants}, e = {evaporation_rate}: {best_fitness}")

    #print("Pheromone levels at iteration 1:", pheromone_levels[0])
    #print("Average path length history:", path_lengths)
    print(f"\nBest fitness: {best_fitness} and best path: {best_path}")
    # Print the fitnesses of the last iteration for each configuration
    for i, fitness in enumerate(best_fitnesses):
        print(f"Best fitness of the last evaluation for trial {i + 1}: {fitness}")

    # Calculate and print additional statistics
    average_fitness = np.mean(best_fitnesses)
    median_fitness = np.median(best_fitnesses)
    std_dev_fitness = np.std(best_fitnesses)

    print(f"\nStatistics for problem with {problem.num_items} items and {problem.num_bins} bins:")
    print(f"Average fitness: {average_fitness}")
    print(f"Median fitness: {median_fitness}")
    print(f"Standard deviation of fitness: {std_dev_fitness}")
    print(f"Best fitness: {best_fitness}")

    # Optionally, print the fitness history for each trial
    for i, fitness_history in enumerate(all_fitness_histories):
        print(f"\nFitness history for trial {i + 1}: {fitness_history}")

    # Calculate and print the total weight in each bin
    # print_bin_contents(problem, best_path)

    # Plot the construction graph with the best path highlighted
    # plot_construction_graph(problem, initialize_pheromones(problem.num_items, problem.num_bins), best_path)
    # plot_construction_graph(problem)
