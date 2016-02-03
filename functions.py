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

def connectNodes(Graph):
    """
    This functions attempts to connect nodes in a network with the smallest distance
    between them.

    The nodes must each have as an attribute a list of touples of the form (nodeIndex, distance),
    where nodeIndex is the index of the other nodes in the network.
    
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
    """
    flows is a dictionary of edge touples as keys and flows as values
    """
    i = 0
    for n, m in Graph.edges_iter():
        Graph[n][m]['flow'] = flows[i]
        i = i + 1
