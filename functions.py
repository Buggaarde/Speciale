import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from math import sqrt
from operator import itemgetter
import random
import multiprocessing as mp
from functools import partial

import Network_solve as ns
import os
import os.path
from itertools import izip
from syncSolver import syncSolver


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

def powerLaw(k, exp):
    return 1.0/(k**exp)


def inversePowerLawList(exponent, maximum = 20):
    k = 1e8
    l= []
    for i in xrange(1, maximum):
        val = int(k*powerLaw(i, exponent))
        for n in xrange(val):
            l.append(i)

    return l

def inversePowerLawDistribution(powerLawList):
    return powerLawList[np.random.randint(1, len(powerLawList))]

                    
def assignFlowsToEdges(Graph, flows):
    """
    flows is a dictionary of edge touples as keys and flows as values
    """
    i = 0
    for n, m in Graph.edges_iter():
        Graph[n][m]['flow'] = flows[i]
        i = i + 1


def squareGrid(numberOfNodes):
    posDict = generatePosDict(numberOfNodes, ofType = 'square grid')
    edgeTouples = generateEdgeTouples(posDict, accordingToRule = "square grid")
    G = nx.Graph()
    G.add_nodes_from(posDict.keys())
    G.add_edges_from(edgeTouples)
    nx.set_node_attributes(G, 'pos', posDict)

    return G

def randomGeometric(numberOfNodes, nodeDegreeDistribution = 'poisson'):
    posDict = generatePosDict(numberOfNodes, ofType = 'random geometric')
    G = nx.Graph()
    G.add_nodes_from(posDict.keys())
    nx.set_node_attributes(G, 'pos', posDict)
    print 'Populated graph with nodes'
    
    nodeDegree = {}
    dist = {}
    for node in G.nodes():
        dist[node] = findDistanceToNeighbours(node, posDict)

    print 'Calculated distances'
    
    if nodeDegreeDistribution == 'power law':
        pl25 = inversePowerLawList(2.5)
    for node in G.nodes():
        if nodeDegreeDistribution == 'poisson':
            nodeDegree[node] = nodeDegreePoisson(avg = 3)
        elif nodeDegreeDistribution == 'power law':
            nodeDegree[node] = inversePowerLawDistribution(pl25)
   
    print 'Assigned node degrees according to a %s distribution' % nodeDegreeDistribution

    nx.set_node_attributes(G, 'dists', dist)
    nx.set_node_attributes(G, 'node degree', nodeDegree)
    connectNodes(G)

    print 'Attempted to connect nodes to nearest neighbors'

    return G

def solve_for_new_data(Graph, alpha):
    # Attempts to save all the results to './Results/europe_sorted_by_alpha.npz'

    if not os.path.isfile('./Results/europe_sorted_by_alpha.npz'):
        print 'Folder Results/ not found'
        print 'Creating folder Results/'
        os.makedirs(r'./Results')
    
        alpha_list = [alpha]
        balance_dict_list = [[]]
        ip_dict_list = [[]]
        flow_dict_list = [[]]
        print 'Calculating powerflows for alpha = %2.2f' % alpha
        syncSolver(Graph)
        for node in Graph.nodes_iter():
            balance_dict_list[-1].append({node: Graph.node[node]['Balance']})
            ip_dict_list[-1].append({node: Graph.node[node]['Injection Pattern']})
        
        for (m, n) in Graph.edges_iter():
            flow_dict_list[-1].append({(m, n): Graph[m][n]['flow']})

        np.savez_compressed('Results/europe_sorted_by_alpha.npz',\
                            alpha_list = alpha_list,\
                            balance_dict_list = balance_dict_list,\
                            ip_dict_list = ip_dict_list,\
                            flow_dict_list = flow_dict_list)
    
        print 'Saved data to Results/europe_sorted_by_alpha.npz'
        
    else:
        print 'Loading data from Results/europe_sorted_by_alpha.npz'
    
        Graph_data = np.load('Results/europe_sorted_by_alpha.npz')
        if not alpha in Graph_data['alpha_list']:
            print alpha
            np.append(Graph_data['alpha_list'], alpha)
            print Graph_data['alpha_list']
            np.append(Graph_data['balance_dict_list'], [])
            np.append(Graph_data['ip_dict_list'], [])
            np.append(Graph_data['flow_dict_list'], [])
            
            print 'Calculating powerflows for alpha = %2.2f' % alpha
            syncSolver(Graph)
            for node in Graph.nodes_iter():
                np.append(Graph_data['balance_dict_list'][-1], {node: Graph.node[node]['Balance']})
                np.append(Graph_data['ip_dict_list'][-1], {node: Graph.node[node]['Injection Pattern']})
                #balance_dict_list[-1].append({node: Graph.node[node]['Balance']})
                #ip_dict_list[-1].append({node: Graph.node[node]['Injection Pattern']})
        
            for (m, n) in Graph.edges_iter():
                np.append(Graph_data['flow_dict_list'][-1], {(m, n): Graph[m][n]['flow']})
                #flow_dict_list[-1].append({(m, n): Graph[m][n]['flow']})
        
            np.savez_compressed('Results/europe_sorted_by_alpha.npz',\
                                alpha_list = Graph_data['alpha_list'],\
                                balance_dict_list = Graph_data['balance_dict_list'],\
                                ip_dict_list = Graph_data['ip_dict_list'],\
                                flow_dict_list = Graph_data['flow_dict_list'])
    
            print 'Saved data to Results/europe_sorted_by_alpha.npz'
        else:
            print 'Data values for alpha = %2.2f found... Skipping...' % alpha


col_dict = {'HEADER' : '\033[95m',
            'OKBLUE' : '\033[94m',
            'OKGREEN' : '\033[92m',
            'WARNING' : '\033[93m',
            'FAIL' : '\033[91m',
            'ENDC' : '\033[0m',
            'BOLD' : '\033[1m',
            'UNDERLINE' : '\033[4m'}
            

def multi_processor_solve(Graph, solver, alpha):
    # -- Setting up for the solve --
    print 'Solving network for alpha = %1.3f' % alpha
    for node in Graph.nodes_iter():
        Graph.node[node]['Mismatch'] = float(alpha)*Graph.node[node]['Gw'] \
                                       + (1 - float(alpha))*Graph.node[node]['Gs'] \
                                       - Graph.node[node]['Load']

    # -- Solving --
    if solver == 'synchronized':
        syncSolver(Graph)
    else:
        print 'Valid solvers are:\n- \'synchronized\''

    # -- Adding new data to the proper containers --
    balance_dict = {}
    ip_dict = {}
    flow_dict = {}
    for node in Graph.nodes_iter():
        balance_dict.update({node: Graph.node[node]['Balance']})
        ip_dict.update({node: Graph.node[node]['Injection Pattern']})

    for (m, n) in Graph.edges_iter():
        flow_dict.update({(m, n): Graph[m][n]['flow']})

    return [{alpha: balance_dict}, {alpha: ip_dict}, {alpha: flow_dict}]


def solve_for_new_dataNEW(Graph, alpha_list, path='Results/', filename='file',
                          multiproc=True, solver='synchronized', savefile=True):
    '''
    Parameters
    ----------
    Graph:
    A NetworkX Graph()-object that for each node contains the attributes
    - 'Gw' for wind generation,
    - 'Gs' for for solar generation,
    - 'Load',
    - 'Balance',
    - 'Injection Pattern'.
    Note that all these attributes has to be 1D numpy arrays containing the values for each
    time step except for 'Balance' and 'Injection Pattern' which has to be numpy arrays of zeros
    with length equal to 'Gw', 'Gs' and 'Load'.
    All lists have to be of same length.

    alpha_list:
    Contains a simple list of all the alpha values that we want to run the solver over.

    path:
    The path to the output directory. The function checks to see if the path exists, and if it
    doesn't, it will create it.

    filename:
    The funcion will save the results to path/filename.npz

    solver:
    A flag to indicate which type of flowscheme the user want the solver to use. Accepts
    - 'syncronized'.

    savefile:
    Change savefile to False if you for some reason do not wish to save the results for later use.
    '''
    if not os.path.isdir(path):
        print 'Path %s not found.' % path
        print 'Creating %s.' % path
        os.makedirs(r'%s' % path)

        print 'New alpha values:'
        print alpha_list

        if multiproc:
            # -- Setting up the multicore process
            cores = mp.cpu_count()
            print 'Using multiple cores.'
            print 'Detecting %d cores.' % cores
            pool = mp.Pool(cores) # initializing multiple cores
            mps0 = partial(multi_processor_solve, Graph) # giving the solver the necessary
            mps1 = partial(mps0, solver)                 # arguments
            print col_dict['WARNING'] + 'Warning: Don\'t interrupt the program while it runs. This may take a long time.' + col_dict['ENDC']
            alpha_list = [float(alpha) for alpha in alpha_list]
            results = pool.map(mps1, alpha_list) # solving and outputting to results
            pool.close()
            pool.join()
        else:
            print 'For the time coming, don\'t run this program with multiproc = False'
            
            # for step in xrange(len(alpha_list)):
            #     # -- Setting up for the solve --
            #     print 'Solving network for alpha = %1.3f' % alpha_list[step]
            #     for node in Graph.nodes_iter():
            #         Graph.node[node]['Mismatch'] = alpha_list[step]*Graph.node[node]['Gw'] \
            #                                        + (1 - alpha_list[step])*Graph.node[node]['Gs'] \
            #                                        - Graph.node[node]['Load']

            #     # -- Solving --
            #     if solver == 'synchronized':
            #         syncSolver(Graph)
            #     else:
            #         print 'Valid solvers are:\n- \'synchronized\''

            #     # -- Adding new data to the proper containers --
            #     for node in Graph.nodes_iter():
            #         balance_dict_list[step].append({node: Graph.node[node]['Balance']})
            #         ip_dict_list[step].append({node: Graph.node[node]['Injection Pattern']})

            #     for (m, n) in Graph.edges_iter():
            #         flow_dict_list[step].append({(m, n): Graph[m][n]['flow']})

        # -- Saving the data to path/filename.npz --
        if savefile:
            # -- 
            balance_alpha_dict = {}
            ip_alpha_dict = {}
            flow_alpha_dict = {}
            for data in results:
                balance_alpha_dict.update(data[0])    
                ip_alpha_dict.update(data[1])    
                flow_alpha_dict.update(data[2])

                
            print 'Saving to %s%s.npz' % (path, filename)
            np.savez_compressed('%s%s.npz' % (path, filename),\
                                alpha_list = sorted(alpha_list),\
                                balance_alpha_dict = balance_alpha_dict,\
                                ip_alpha_dict = ip_alpha_dict,\
                                flow_alpha_dict = flow_alpha_dict)
            print 'Saved...'

    else:
        print 'Found %s%s.npz... Loading data...' % (path, filename)
        dataset = np.load('%s%s.npz' % (path, filename))
        alpha_stored = dataset['alpha_list'].tolist()

        # -- Checking to see whether the provided alpha's are already in stored data --
        new_alpha_list = []
        for alpha in alpha_list:
            if alpha not in alpha_stored:
                new_alpha_list.append(alpha)
        if new_alpha_list:
            print 'Found new alpha values:'
            print new_alpha_list
            alpha_list = new_alpha_list
        else:
            print 'No new alpha values found... Skipping...'

        if new_alpha_list:
            if multiproc:
                # -- Setting up the multicore process
                cores = mp.cpu_count()
                print 'Using multiple cores.'
                print 'Detecting %d cores.' % cores
                pool = mp.Pool(cores) # initializing multiple cores
                mps0 = partial(multi_processor_solve, Graph) # giving the solver the necessary
                mps1 = partial(mps0, solver)                 # arguments
                print col_dict['WARNING'] + 'Warning: Don\'t interrupt the program while it runs. This may take a long time.' + col_dict['ENDC']
                results = pool.map(mps1, alpha_list) # solving and outputting to results
                pool.close()
                pool.join()
            else:
                print 'For the time coming, don\'t run this program with multiproc = False'

            # -- Saving the data to path/filename.npz --
            if savefile:
                balance_alpha_dict = dataset['balance_alpha_dict'].item()
                #print balance_alpha_dict[0]
                ip_alpha_dict = dataset['ip_alpha_dict'].item()
                flow_alpha_dict = dataset['flow_alpha_dict'].item()
                
                for data in results:
                    balance_alpha_dict.update(data[0])
                    ip_alpha_dict.update(data[1])    
                    flow_alpha_dict.update(data[2])

                for alpha in alpha_list:
                    alpha_stored.append(alpha)
                alpha_list = alpha_stored
                print alpha_list
                    
                print 'Saving to %s%s.npz' % (path, filename)
                np.savez_compressed('%s%s.npz' % (path, filename),\
                                    alpha_list = alpha_list,\
                                    balance_alpha_dict = balance_alpha_dict,\
                                    ip_alpha_dict = ip_alpha_dict,\
                                    flow_alpha_dict = flow_alpha_dict)
                print 'Saved...'

        
                
