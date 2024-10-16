import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class BinPackingProblem:
    def __init__(self, num_items, num_bins, weights):
        self.num_items = num_items
        self.num_bins = num_bins
        self.weights = weights

def initialize_pheromones(num_items, num_bins):
    return np.round(np.random.rand(num_items, num_bins), 2)

def generate_ant_paths(problem, pheromones, num_ants):
    paths = np.zeros((num_ants, problem.num_items), dtype=int)
    for item_index in range(problem.num_items):
        pheromone_levels = pheromones[item_index]
        total_pheromone = np.sum(pheromone_levels)
        probabilities = pheromone_levels / total_pheromone
        chosen_bins = np.random.choice(problem.num_bins, size=num_ants, p=probabilities)
        paths[:, item_index] = chosen_bins
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
    for item_index in range(paths.shape[1]):
        for ant_index in range(paths.shape[0]):
            bin_index = paths[ant_index, item_index]
            pheromones[item_index, bin_index] += 1 / fitnesses[ant_index]

def evaporate_pheromones(pheromones, evaporation_rate):
    pheromones *= evaporation_rate

def plot_construction_graph(problem, pheromones):
    G = nx.DiGraph()
    G.add_node("S")  # Start node
    G.add_node("E")  # End node

    pos = {"S": (1, 0), "E": (-1, 0)}  # Fixed positions for start and end nodes

    # Add nodes and edges for each item and bin
    for item_index in range(problem.num_items):
        item_node = f"Item {item_index + 1}"
        G.add_node(item_node)
        G.add_edge("S", item_node)
        pos[item_node] = (0, item_index - problem.num_items / 2)

        for bin_index in range(problem.num_bins):
            bin_node = f"Item {item_index + 1} -> Bin {bin_index + 1} ({pheromones[item_index, bin_index]:.2f})"
            G.add_node(bin_node)
            G.add_edge(item_node, bin_node, weight=pheromones[item_index, bin_index])
            G.add_edge(bin_node, "E")
            pos[bin_node] = (-0.5, item_index - problem.num_items / 2 + bin_index * 0.2)

    plt.figure(figsize=(12, 8))
    edges = G.edges(data=True)
    weights = [edge[2].get('weight', 1) for edge in edges]
    nx.draw(G, pos, with_labels=True, node_color='lightblue', font_size=8, font_weight='bold', node_size=2000, width=weights, arrows=True)
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
problem = BinPackingProblem(6, 3, [17, 12, 19, 6, 4, 28])

# Run a single experiment
best_fitness, best_path = ant_colony_optimization(problem, num_ants=10, evaporation_rate=0.9, max_evaluations=100)
print(f"Best fitness: {best_fitness}")

# Print the bin assignments for each item
for item_index, bin_index in enumerate(best_path):
    print(f"Item {item_index + 1} is in Bin {bin_index + 1}")

# Plot the construction graph
plot_construction_graph(problem, initialize_pheromones(problem.num_items, problem.num_bins))