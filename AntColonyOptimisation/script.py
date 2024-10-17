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


def generate_ant_paths(problem, pheromones, num_ants):
    paths = np.zeros((num_ants, problem.num_items), dtype=int)
    for ant_index in range(num_ants):
        current_node = "S"
        for item_index in range(problem.num_items):
            pheromone_levels = np.array(
                [pheromones[(current_node, f"{item_index + 1}, {bin_index + 1}")] for bin_index in
                 range(problem.num_bins)])
            total_pheromone = np.sum(pheromone_levels)
            probabilities = pheromone_levels / total_pheromone
            chosen_bin = np.random.choice(problem.num_bins, p=probabilities)
            paths[ant_index, item_index] = chosen_bin
            current_node = f"{item_index + 1}, {chosen_bin + 1}"
    return paths


def calculate_fitness(problem, paths):
    bin_weights = np.zeros((paths.shape[0], problem.num_bins))
    for item_index in range(problem.num_items):
        for ant_index in range(paths.shape[0]):
            bin_index = paths[ant_index, item_index]
            bin_weights[ant_index, bin_index] += problem.weights[item_index]
    fitnesses = np.max(bin_weights, axis=1) - np.min(bin_weights, axis=1)
    print(fitnesses)
    return fitnesses


def update_pheromones(pheromones, paths, fitnesses):
    for ant_index in range(paths.shape[0]):
        current_node = "S"
        for item_index in range(paths.shape[1]):
            bin_index = paths[ant_index, item_index]
            next_node = f"{item_index + 1}, {bin_index + 1}"
            pheromones[(current_node, next_node)] += 1 / fitnesses[ant_index]
            current_node = next_node


def evaporate_pheromones(pheromones, evaporation_rate):
    for edge in pheromones:
        pheromones[edge] *= evaporation_rate


def plot_construction_graph(problem, pheromones, best_path):
    G = nx.DiGraph()
    G.add_node("S")  # Start node
    G.add_node("E")  # End node

    pos = {"S": (0, 2), "E": (problem.num_items + 1, 2)}  # Set S and E at the same height as bin 2

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


def ant_colony_optimization(problem, num_ants, evaporation_rate, max_evaluations):
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
        update_pheromones(pheromones, paths, fitnesses)
        evaporate_pheromones(pheromones, evaporation_rate)
        evaluations += num_ants

    return best_fitness, best_path


# Define the BinPackingProblem instance with 6 items and 3 bins
BPP1 = BinPackingProblem(6, 4, [17, 12, 19, 6, 4, 28])

# Run a single experiment
best_fitness, best_path = ant_colony_optimization(BPP1, num_ants=10, evaporation_rate=0.9, max_evaluations=100)
print(f"Best fitness: {best_fitness}")

# Print the bin assignments for each item in bold
for item_index, bin_index in enumerate(best_path):
    print(f"**Item {item_index + 1} is in Bin {bin_index + 1}**")

# Plot the construction graph with the best path highlighted
plot_construction_graph(BPP1, initialize_pheromones(BPP1.num_items, BPP1.num_bins), best_path)
