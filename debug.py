import neat
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle

f = open("demofile2.txt", "a")
f.write("Now the file has more content!")
f.close()

def save_winner(winner, config, filename):
    with open(filename, 'wb') as f:
        pickle.dump((config, winner), f)
    print(f"Winner network saved to {filename}")


def load_winner(filename):
    with open(filename, 'rb') as f:
        config, winner = pickle.load(f)
    print(f"Winner network loaded from {filename}")
    return config, winner

config, winner = load_winner('winner_net.pkl')

def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False, node_colors=None, fmt='png'):
    network = nx.DiGraph()
    for node in genome.nodes:
        network.add_node(node, **genome.nodes[node].__dict__)
    for connection in genome.connections:
        if genome.connections[connection].enabled or show_disabled:
            network.add_edge(connection[0], connection[1], weight=genome.connections[connection].weight)
    if prune_unused:
        used_nodes = set()
        for conn in genome.connections.values():
            if conn.enabled:
                used_nodes.add(conn.key[0])
                used_nodes.add(conn.key[1])
        for node in list(network.nodes):
            if node not in used_nodes:
                network.remove_node(node)
    node_colors = node_colors if node_colors else {}
    node_colors = [node_colors.get(node, 'lightblue') for node in network.nodes]
    if node_names:
        labels = {i: node_names.get(i, str(i)) for i in network.nodes}
    else:
        labels = {i: str(i) for i in network.nodes}
    pos = nx.spring_layout(network)
    nx.draw(network, pos, labels=labels, with_labels=True, node_size=500, node_color=node_colors, font_size=10, font_color='black', edge_color='grey')
    nx.draw_networkx_edge_labels(network, pos, edge_labels={(u, v): f"{d['weight']:.2f}" for u, v, d in network.edges(data=True)})
    if view:
        plt.show()
    if filename:
        plt.savefig(filename, format=fmt)
    plt.clf()


print('finished.  winner:', winner, 'config:', config)
