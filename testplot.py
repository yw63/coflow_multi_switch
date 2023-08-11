import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import json
import ast
import networkx as nx
from networkx.algorithms import bipartite

G = nx.DiGraph()

partition1 = [1, 2, 3]
partition2 = ['A', 'B', 'C']

G.add_nodes_from(partition1, bipartite=0)
G.add_nodes_from(partition2, bipartite=1)

edges_partition1_to_partition2 = [(1, 'A'), (1, 'B'), (2, 'B')]
edges_partition2_to_partition1 = [('A', 1), ('B', 1), ('B', 2), ('C', 3)]

G.add_edges_from(edges_partition1_to_partition2, color='red')
G.add_edges_from(edges_partition2_to_partition1, color='blue')

pos = nx.bipartite_layout(G, partition1)

node_colors = ['skyblue' if node in partition1 else 'lightgreen' for node in G.nodes()]

edge_colors = [G.edges[edge]['color'] for edge in G.edges()]

nx.draw_networkx(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=12, edge_color=edge_colors, arrows=True)

plt.savefig("plt/test.png", format="PNG")
