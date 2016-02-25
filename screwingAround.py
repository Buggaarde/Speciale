#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from math import sqrt
from operator import itemgetter
import random

import Network_solve as ns
import sfg as sfg
from functions import *
import timeit


nodes = 30
steps = 20000
G = nx.powerlaw_cluster_graph(nodes, 3, 0.2)
np.savetxt('./settings/adjacency.txt', nx.adjacency_matrix(G).todense())
np.savetxt('./settings/mismatch.txt', np.random.randn(steps, nodes))

start = timeit.default_timer()
ns.networkSolver(nodes)
end = timeit.default_timer()
print 'time to solve %d timesteps: %2.2f' % (steps, (end - start))
print 'average %f sec per step' % ((end - start)/steps)
