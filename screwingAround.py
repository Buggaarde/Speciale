#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from math import sqrt
from operator import itemgetter
import random

import Network_solve as ns


def generatePosDict(nrOfNodes, ofType = "square grid"):
    posDict = {}
    nodeIndex = 0
    if ofType == "square grid":
        # Assumes that nrOfNodes is a square number
        for y in xrange(int(sqrt(nrOfNodes))):
            for x in xrange(int(sqrt(nrOfNodes))):
                posDict[nodeIndex] = (x, y)
                nodeIndex = nodeIndex + 1


    elif ofType == "random geometric":
        for i in xrange(nrOfNodes):
            posDict[nodeIndex] = (np.random.random_sample(), np.random.random_sample())
            nodeIndex = nodeIndex + 1


            
    return posDict


def generateEdgeTouples(posDict, accordingToRule = "square grid"):

    edgeTouples = []
        
    if accordingToRule == "square grid":
            # Assuming that nrOfNodes equals some square number

            nrOfNodes = len(posDict)
            mMax = sqrt(nrOfNodes) # A total of mMax entries in the x-direction
            nMax = sqrt(nrOfNodes) # A total of nMax entries in the downwards y-direction

            for k, v in posDict.iteritems():
                m = k % mMax
                n = (k - m) / nMax

                if not m - 1 < 0:
                    edgeTouples.append((k, k - 1))
                if not m + 1 > mMax - 1:
                    edgeTouples.append((k, k + 1))
                if not n - 1 < 0:
                    edgeTouples.append((k, k - mMax))
                if not n + 1 > nMax - 1:
                    edgeTouples.append((k, k + mMax))

    if accordingToRule == "random geometric":
        # First we want to find the number of neighbours
        pass


    return edgeTouples


def findDistanceToNeighbours(nodeIndex, posDict):
    """
    Returns
    -------
    distTouples : [(nodeIndex, euclidean distance), ...]
    
    distTouples is reverse sorted after distance.
    """
    # Assuming that the nodeIndex is in the posDict.
    distTouples = []
    
    x, y = posDict[nodeIndex]
    pos = list(posDict.values())
    
    while pos:
        x1, y1 = pos.pop()
        r = (x - x1)**2 + (y - y1)**2
        distTouples.append((len(pos), sqrt(r)))

        dist = sorted(distTouples, key=itemgetter(1), reverse = True)

    return dist

def nodeDegreePoisson(avg = 3):
    """
    Determines the degree of a node according to a simple Poisson distribution.

    Parameters
    ----------
    avg : int
          The average node degree

    Math
    ----
    P(k events in interval) = lambda^k * exp(-lambda)/k!
    """
    return np.random.poisson(avg)


def nodeDegreeInversePowerLaw(exponent):
    return random.paretovariate(exponent - 1)

def nodeDegreeExponential():
    pass


def connectNodes(Graph):
    """
    Returns a list of edges for a graph containing nodes with predetermined node degrees.
    This function tries to match nodes with smallest euclidean distance that also has free links.
    """
    nodes = Graph.nodes()
    for i in range(len(nodes)):
        # print " Node %d" % i
        n = nodes[i]
        dists = Graph.node[n]['dists'][: len(Graph.node[n]['dists']) - 1] #exclude the last entry which is the node itself.
        nDegree = Graph.node[n]['node degree']

        if len(Graph.neighbors(n)) >= nDegree:
            # print "exit 1"
            continue
        else:
            while dists:
                d = dists.pop() # choosing the nearest neighbour we haven't checked yet
                m = d[0] # m is the node we are checking against
                dist = d[1] # dist is the distance to the node we are checking against
                mDegree = Graph.node[m]['node degree']
                if len(Graph.neighbors(n)) >= nDegree:
                    # print "exit 2"
                    break
                elif len(Graph.neighbors(m)) >= mDegree:
                    # print "exit 3"
                    continue
                elif m in Graph.neighbors(n):
                    # print "exit 4"
                    continue
                else:
                    # print "exit 5"
                    Graph.add_edge(n, m)


def assignFlowsToEdges(Graph, flows):
    i = 0
    for n, m in Graph.edges_iter():
        Graph[n][m]['flow'] = flows[i]
        i = i + 1
        
    
                    

PosDict = generatePosDict(4, ofType = "random geometric")
edgeTouples = generateEdgeTouples(PosDict, accordingToRule = "square grid")
G = nx.Graph()
G.add_nodes_from(PosDict.keys())
#G.add_edges_from(edgeTouples)

nodeDegree = {}
dist = {}
for node in G.nodes():
    dist[node] = findDistanceToNeighbours(node, PosDict)
    nodeDegree[node] = nodeDegreePoisson(avg = 3)

nx.set_node_attributes(G, 'dists', dist)
nx.set_node_attributes(G, 'node degree', nodeDegree)
connectNodes(G)



# print nodeDegree.values()
# print [len(G.neighbors(node)) - nodeDegree.values()[node] for node in G.nodes()]

avg = 0.0
for node in G.nodes():
    avg = avg + G.degree(node)

print avg/len(G.nodes())

A = nx.adjacency_matrix(G)


I = nx.incidence_matrix(G)

np.savetxt('./settings/adjacency.txt', A.todense(), fmt = '%d')
np.savetxt('./settings/incidence.txt', I.todense())
# np.savetxt('./settings/mismatch.txt', np.random.randn(2, len(G.nodes())))
mismatch = np.zeros((2, len(G.nodes())))
mismatch[1, :] = np.random.randn(len(G.nodes()))
for node in G.nodes():
    G.node[node]['mismatch'] = mismatch[1, node]
# print mismatch
np.savetxt('./settings/mismatch.txt', mismatch)


ns.networkSolver(len(G.nodes()))



results = np.load('./results/TestFlow.npy')
#print results
# print results.shape[0]
# print len(G.edges())
print G.edges()
print
balance = nx.get_node_attributes(G, 'mismatch').values()
print
print 'balance'
print balance
print
print 'flows'
print results[:, 1]

assignFlowsToEdges(G, results[:, 1])
flows = nx.get_edge_attributes(G, 'flow')

# print flows

newBalance = balance
for node in G.nodes():
    for edge in G.edges(node):
        if edge[0] < edge[1]:
            newBalance[node] = newBalance[node] - flows[edge]
        else:
            edge = edge[::-1]
            newBalance[node] = newBalance[node] + flows[edge]

print
print 'new balance'
print newBalance

            
        



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
