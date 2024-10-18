import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class BinPackingProblem:
    def __init__(self, num_items, num_bins, weights):
        self.num_items = num_items
        self.num_bins = num_bins
        self.weights = weights


def initialize_pheromones(num_items, num_bins):
    pheromones = {}
    for item_index in range(num_items):
        for bin_index in range(num_bins):
            if item_index == 0:
                pheromones[("S", f"{item_index + 1}, {bin_index + 1}")] = np.round(np.random.rand(), 2)
            else:
                for prev_bin_index in range(num_bins):
                    pheromones[
                        (f"{item_index}, {prev_bin_index + 1}", f"{item_index + 1}, {bin_index + 1}")] = np.round(
                        np.random.rand(), 2)
            if item_index == num_items - 1:
                pheromones[(f"{item_index + 1}, {bin_index + 1}", "E")] = np.round(np.random.rand(), 2)
    return pheromones


def generate_ant_paths(problem, pheromones, num_ants, evaporation_rate):
    paths = np.zeros((num_ants, problem.num_items), dtype=int)
    for ant in range(num_ants):
        current_node = "S"
        for item_index in range(problem.num_items):
            pheromone_levels = np.round(np.array(
                [pheromones[(current_node, f"{item_index + 1}, {bin_index + 1}")] for bin_index in
                 range(problem.num_bins)]), 2)
            total_pheromone = np.sum(pheromone_levels)

            probabilities = pheromone_levels / total_pheromone

            if np.isnan(probabilities).any():
                probabilities = np.ones(problem.num_bins) / problem.num_bins

            chosen_bin = np.random.choice(problem.num_bins, p=probabilities)
            print_pheromone_levels(pheromones, current_node)
            paths[ant, item_index] = chosen_bin
            current_node = f"{item_index + 1}, {chosen_bin + 1}"
            print(
                f"Item {item_index + 1} dropped in bin {chosen_bin + 1} with pheromone level {pheromone_levels[chosen_bin]}")

        # Update pheromones after each ant completes its path
        fitness = calculate_fitness(problem, paths[ant:ant + 1])
        update_pheromones(pheromones, paths[ant:ant + 1], fitness)
        evaporate_pheromones(pheromones, evaporation_rate)

    return paths


def calculate_fitness(problem, paths):
    bin_weights = np.zeros((paths.shape[0], problem.num_bins))
    for item_index in range(problem.num_items):
        for ant_index in range(paths.shape[0]):
            bin_index = paths[ant_index, item_index]
            bin_weights[ant_index, bin_index] += problem.weights[item_index]
    fitnesses = np.max(bin_weights, axis=1) - np.min(bin_weights, axis=1)
    return fitnesses


def update_pheromones(pheromones, paths, fitnesses):
    print("Updating pheromones")
    for ant_index in range(paths.shape[0]):
        current_node = "S"
        for item_index in range(paths.shape[1]):
            bin_index = paths[ant_index, item_index]
            next_node = f"{item_index + 1}, {bin_index + 1}"
            pheromones[(current_node, next_node)] = np.round(
                pheromones[(current_node, next_node)] + 1 / fitnesses[ant_index], 2)
            current_node = next_node


def evaporate_pheromones(pheromones, evaporation_rate):
    for edge in pheromones:
        pheromones[edge] *= evaporation_rate


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


def ant_colony_optimization(problem, num_ants, evaporation_rate, max_evaluations):
    pheromones = initialize_pheromones(problem.num_items, problem.num_bins)
    best_fitness = float('inf')
    best_path = None
    evaluations = 0

    while evaluations < max_evaluations:
        paths = generate_ant_paths(problem, pheromones, num_ants, evaporation_rate)
        print(paths)
        fitnesses = calculate_fitness(problem, paths)

        print('----------------')
        for path, fitness in zip(paths, fitnesses):
            print(f"Path: {path}, Fitness: {fitness}")
        print('----------------')

        min_fitness_index = np.argmin(fitnesses)
        if fitnesses[min_fitness_index] < best_fitness:
            best_fitness = fitnesses[min_fitness_index]
            best_path = paths[min_fitness_index]
            print('BEST PATH: ', best_path)
        update_pheromones(pheromones, paths, fitnesses)
        evaporate_pheromones(pheromones, evaporation_rate)
        evaluations += num_ants

    return best_fitness, best_path


def calculate_bin_weights(problem, best_path):
    bin_weights = np.zeros(problem.num_bins)
    for item_index, bin_index in enumerate(best_path):
        bin_weights[bin_index] += problem.weights[item_index]
    return bin_weights


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
            bin_node = f"{item_index + 1}, {bin_index + 1}"
            G.add_node(bin_node)
            pos[bin_node] = (item_index + 1, bin_index + 1)

            # Connect start node to the first item nodes
            if item_index == 0:
                G.add_edge("S", bin_node, pheromone=pheromones[("S", bin_node)])
            else:
                for prev_bin_index in range(problem.num_bins):
                    prev_bin_node = f"{item_index}, {prev_bin_index + 1}"
                    G.add_edge(prev_bin_node, bin_node, pheromone=pheromones[(prev_bin_node, bin_node)])

            # Connect to the end node only after the last item
            if item_index == problem.num_items - 1:
                G.add_edge(bin_node, "E", pheromone=pheromones[(bin_node, "E")])

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

# Define the BinPackingProblem instance with 6 items and 3 bins
BPP1 = BinPackingProblem(6, 3, [3, 6, 2, 5, 1, 4])
# Chosn Bin does num_items * 100
# BPP1 = BinPackingProblem(500, 10, list(range(1, 501)))
BPP2 = BinPackingProblem(500, 50, [(i ** 2) / 2 for i in range(1, 501)])
problem = BPP1

# Run a single experiment
best_fitness, best_path = ant_colony_optimization(problem, num_ants=10, evaporation_rate=0.9, max_evaluations=1)
print(f"Best fitness: {best_fitness}")

# Calculate and print the total weight in each bin
# bin_weights = calculate_bin_weights(problem, best_path)
# for bin_index, weight in enumerate(bin_weights):
#    print(f"Bin {bin_index + 1} total weight: {weight}")

# Plot the construction graph with the best path highlighted
# plot_construction_graph(problem, initialize_pheromones(problem.num_items, problem.num_bins), best_path)
plot_construction_graph(problem)
