# neat_xor2.py
import neat
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# XOR inputs and expected outputs
xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_outputs = np.array([[0], [1], [1], [0]])

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 4.0
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2

def run():
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                'config-feedforward')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(PlotReporter(True))
    winner = p.run(eval_genomes, 50)
    return winner, stats, config

def test_winner(winner, config):
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    results = []
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        results.append((xi, xo, output))
    return results

def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False, node_colors=None, fmt='png'):
    """Receives a genome and draws a neural network with arbitrary topology."""
    # Create the network graph
    network = nx.DiGraph()

    # Add nodes with their attributes
    for node in genome.nodes:
        network.add_node(node, **genome.nodes[node].__dict__)

    # Add edges with their attributes
    for connection in genome.connections:
        if genome.connections[connection].enabled or show_disabled:
            network.add_edge(connection[0], connection[1], weight=genome.connections[connection].weight)

    # Prune unused nodes
    if prune_unused:
        used_nodes = set()
        for conn in genome.connections.values():
            if conn.enabled:
                used_nodes.add(conn.key[0])
                used_nodes.add(conn.key[1])
        for node in list(network.nodes):
            if node not in used_nodes:
                network.remove_node(node)

    # Set node colors
    node_colors = node_colors if node_colors else {}
    node_colors = [node_colors.get(node, 'lightblue') for node in network.nodes]

    # Set node names
    if node_names:
        labels = {i: node_names.get(i, str(i)) for i in network.nodes}
    else:
        labels = {i: str(i) for i in network.nodes}

    # Draw the network
    pos = nx.spring_layout(network)  # positions for all nodes
    nx.draw(network, pos, labels=labels, with_labels=True, node_size=500, node_color=node_colors, font_size=10, font_color='black', edge_color='grey')
    nx.draw_networkx_edge_labels(network, pos, edge_labels={(u, v): f"{d['weight']:.2f}" for u, v, d in network.edges(data=True)})

    # Show the plot
    if view:
        plt.show()

    # Save the plot
    if filename:
        plt.savefig(filename, format=fmt)

    plt.clf()

class PlotReporter(neat.reporting.BaseReporter):
    def __init__(self, show_plot):
        self.show_plot = show_plot
        self.generation = []
        self.best_fitness = []
        self.avg_fitness = []

        if self.show_plot:
            plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots(figsize=(10, 5))
            self.line1, = self.ax.plot([], [], label='Best Fitness')
            self.line2, = self.ax.plot([], [], label='Average Fitness')
            self.ax.set_xlabel('Generation')
            self.ax.set_ylabel('Fitness')
            self.ax.set_title('Training Progress')
            self.ax.legend()
            self.ax.grid()

    def post_evaluate(self, config, population, species, best_genome):
        self.generation.append(len(self.generation))
        best_fitness = max([genome.fitness for genome in population.values()])
        avg_fitness = np.mean([genome.fitness for genome in population.values()])
        self.best_fitness.append(best_fitness)
        self.avg_fitness.append(avg_fitness)
        
        if self.show_plot:
            self.plot_progress()

    def plot_progress(self):
        self.line1.set_data(self.generation, self.best_fitness)
        self.line2.set_data(self.generation, self.avg_fitness)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        

# Example usage
if __name__ == '__main__':
    winner, stats, config = run()
    draw_net(config, winner, view=True, filename="winner_net.png")
    test_results = test_winner(winner, config)
    for result in test_results:
        input = result[0]
        expected_output = result[1]
        predicted_output = result[2]
        print(f"Input: {input}, Expected Output: {expected_output}, Predicted Output: {predicted_output}")
