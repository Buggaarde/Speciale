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

# F = squareGrid(5**2)
# print F.node
# nx.draw(F, nx.get_node_attributes(F, 'pos'))
# plt.show()

# R = randomGeometric(100)
# nx.draw(R, nx.get_node_attributes(R, 'pos'))
# plt.show()

B = nx.powerlaw_cluster_graph(100, 2, 1)
print "average clustering: %f" % nx.average_clustering(B)
print 'average shortest path length: %f' % nx.average_shortest_path_length(B)
plt.subplot(211)
nx.draw(B)

l = []
for node in B.nodes():
    l.append(len(B.neighbors(node)))
plt.subplot(212)
plt.hist(l, max(l))
plt.show()

S = nx.navigable_small_world_graph(5)
S = S.to_undirected()
avgNodeDegree = 0
for node in S.nodes():
    avgNodeDegree += len(S.neighbors(node))
avgNodeDegree = avgNodeDegree/len(S.nodes())

print 'average node degree: %f' % avgNodeDegree
print "average clustering: %f" % nx.average_clustering(S)
print 'average shortest path length: %f' % nx.average_shortest_path_length(S)
nx.draw(S)
plt.show()
        


# PosDict = generatePosDict(4, ofType = "random geometric")
# edgeTouples = generateEdgeTouples(PosDict, accordingToRule = "square grid")
# G = nx.Graph()
# G.add_nodes_from(PosDict.keys())
# #G.add_edges_from(edgeTouples)

# nodeDegree = {}
# dist = {}
# for node in G.nodes():
#     dist[node] = findDistanceToNeighbours(node, PosDict)
#     nodeDegree[node] = nodeDegreePoisson(avg = 3)

# nx.set_node_attributes(G, 'dists', dist)
# nx.set_node_attributes(G, 'node degree', nodeDegree)
# connectNodes(G)



# # print nodeDegree.values()
# # print [len(G.neighbors(node)) - nodeDegree.values()[node] for node in G.nodes()]

# avg = 0.0
# for node in G.nodes():
#     avg = avg + G.degree(node)

# print avg/len(G.nodes())

# A = nx.adjacency_matrix(G)


# I = nx.incidence_matrix(G)

# np.savetxt('./settings/adjacency.txt', A.todense(), fmt = '%d')
# np.savetxt('./settings/incidence.txt', I.todense())
# # np.savetxt('./settings/mismatch.txt', np.random.randn(2, len(G.nodes())))
# mismatch = np.zeros((2, len(G.nodes())))
# mismatch[1, :] = np.random.randn(len(G.nodes()))
# for node in G.nodes():
#     G.node[node]['mismatch'] = mismatch[1, node]
# # print mismatch
# np.savetxt('./settings/mismatch.txt', mismatch)


# #ns.networkSolver(len(G.nodes()))



# results = np.load('./results/TestFlow.npy')
# #print results
# # print results.shape[0]
# # print len(G.edges())
# print G.edges()
# print
# balance = nx.get_node_attributes(G, 'mismatch').values()
# print
# print 'balance'
# print balance
# print
# print 'flows'
# print results[:, 1]

# assignFlowsToEdges(G, results[:, 1])
# flows = nx.get_edge_attributes(G, 'flow')

# # print flows

# newBalance = balance
# for node in G.nodes():
#     for edge in G.edges(node):
#         if edge[0] < edge[1]:
#             newBalance[node] = newBalance[node] - flows[edge]
#         else:
#             edge = edge[::-1]
#             newBalance[node] = newBalance[node] + flows[edge]

# print
# print 'new balance'
# print newBalance

            
        



# print 'balance before = '
# print balance

# for node in G.nodes():
#     for edge in G.edges(node):
#         if edge[0] < edge[1]:
#             balance[node] =  balance[node] + flows[edge]

# print 'balance after = '
# print balance




# for n, m in G.edges_iter():
#     print '(%d, %d)' % (n, m)


# nx.draw(G, PosDict, with_labels = True)
# plt.show()

# C = nx.powerlaw_cluster_graph(1000, 2, 0.5)
# avg = 0.0
# for node in C.nodes():
#     avg = avg + C.degree(node)

# print "Nodes: %d" % len(C.nodes())
# print "Average node degree: %f" % (avg/len(C.nodes()))

# print "Clustering coefficient: %f" % nx.average_clustering(C)

# plt.hist([C.degree(node) for node in C.nodes()], len(C.nodes()))
# plt.show()
# # nx.draw(C)
# # plt.show()







# nx.draw(W, pos)
# plt.show()
