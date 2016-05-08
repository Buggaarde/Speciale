import numpy as np
import networkx as nx
import sys
sys.path.insert(0, r'/path/to/solvers/')
from solvers import sync_solver, local_solver

# -- Creating the network -- #

# NetworkX has a feature that can create a network from a specified
# adjacency matrix. Note that NetworkX only accepts only accept
# adjacency matricies with 1's or 0's in the entries.
adjacency = np.loadtxt('/path/to/adjacency_matrix.txt')
Europe = nx.from_numpy_matrix(adjacency)

# We can associate to each node and edge almost arbitrary attributes.
# This is done via dictionaries, and is a very powerful feature of
# NetworkX.
# Adding country name and position can be done with
names = ['AT', 'FI', 'NL', 'BA', 'FR', 'NO', 'BE', 'GB', 'PL', 'BG',
         'GR', 'PT', 'CH', 'HR', 'RO', 'CZ', 'HU', 'RS', 'DE', 'IE',
         'SE', 'DK', 'IT', 'SI', 'ES', 'LU', 'SK', 'EE', 'LV', 'LT']

positions = [[0.55,0.45],[.95,1.1],[0.40,0.85],[0.65,0.15],[0.15,0.60],
             [0.5,1.1],[0.275,0.775],[0.10,1.05],[0.75,0.8],[0.9,0.0],
             [0.7,0.0],[0.0,0.15],[0.4,0.45],[0.75,0.3],[1.0,0.15],
             [0.75,0.60],[1.0,0.45],[0.85,0.15],[0.45,0.7],[0.0,0.95],
             [0.75,1.0],[0.5,0.875],[0.4,0.2],[0.55,0.3],[0.15,0.35],
             [0.325,0.575],[0.90,0.55],[1.03,0.985],[0.99,0.85],[0.925,0.72]]

# Create dictionaries that we can feed directly to NetworkX
name_dict = dict(izip((range(len(names)), names))) # Change to python3 syntax if necessary
pos_dict = dict(izip((range(len(positions)), positions)))
# Setting the names and positions
nx.set_node_attributes(Europe, 'name', name_dict)
nx.set_node_attributes(Europe, 'pos', pos_dict)
# -------------------------- #

# -- Loading the ISET data into the network -- #

# Since we may want to vary the alpha and gamma parameters,
# we define them now.
alpha = 0.8
gamma = 1.0

# We want to add a Load and a Mismatch timeseries to each node,
# since that is what the solver requires in order to run.
for node in Europe.nodes(): # consider using .nodes_iter() with large networks
    data = np.load('/path/to/ISET/ISET_country_{}.npz'.format(names[node]))
    load = data['L']
    # There are some peculiarities with the ISET data. For instance,
    # the solar and wind generation are both normed with respect
    # to the mean of the loads.
    mean_load = np.mean(load)
    gw = data['Gw']*mean_load # Wind generation
    gs = data['Gs']*mean_load # Solar generation

    # Adding the data to the network
    Europe.node[node]['Load'] = load
    # The mismatch is the difference between the generation and the load
    # and with gamma being the renewable penetration, and alpha being
    # the split between wind and solar generation
    Europe.node[node]['Mismatch'] = gamma*(alpha*gw + (1 - alpha)*gs) - load
# --------------------------------------------- #

# -- Solving the network -- #

# With the load and injection pattern timeseries already
# added to the network, it is straight forward to solve them.
Europe_sync = sync_solver(Europe)
Europe_local = local_solver(Europe)
# sync and local describes a solve using the synchronized and localized
# flowscheme, repsectively.

# Europe_sync(local) contains timeseries for load, mismatch,
# injection pattern and balance for each node and a timeseries
# for flow for each edge in the network and is otherwise a complete copy
# of Europe.

# The flows are defined such that flow on a link between node m and node n,
# m < n, is positive if there is flow from m->n and negative if there
# is flow from n->m. In other words, there is positive flow if a node
# with a lower number transfers power to a node with higher number, and
# vice versa.
# -------------------------- #

# -- Extracting information -- #

# It is very straight forward to extract relevant information
# from the solved networks when one is familiar with NetworkX.
# If one wishes to extract any information from the nodes, one could
# write
for node in Europe_sync.nodes_iter():
    load = Europe_sync.node[node]['Load']
    mismatch = Europe_sync.node[node]['Mismatch']
    injection = Europe_sync.node[node]['Injection Pattern']
    balance = Europe_sync.node[node]['Balance']
# where load, etc., is the complete timeseries for the given
# node.
# Backup and curtailment is extracted easily from balance:
backup = -balance
backup[balance > 0] = 0
curtailment = balance
curtailment[balance < 0] = 0
# If one wishes to extract the flows from the edges of the network,
# one could write
for m,n in Europe_local.edges_iter():
    flow = Europe_local[m][n]['Flow']
# As with the nodes, flow is a full timeseries for the given edge.
# ---------------------------- #
