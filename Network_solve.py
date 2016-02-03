#! /usr/bin/env python
__author__ = 'Raunbak'
import networkx as nx
import numpy as np
import aurespf.solvers as au
import regions.classes as region


def networkSolver(numberOfNodes, flowMode = 'synchronized', pathToAdjacency = './settings/adjacency.txt', pathToMismatch = './settings/mismatch.txt'):
    G = nx.Graph()
    # Antal tidsskridt og antal nodes, evt lav til gen from
    numberOfNodes = numberOfNodes

    # sigma * np.random.randn(...) + mu
    mismatch = np.loadtxt(pathToMismatch)
    timesteps = len(mismatch[:, 0])
    # mismatch = np.random.randn(timesteps, numberOfNodes)

    
    # Saa her dannes der dummy lande filer, til at smide ind i solveren, da den laver selv data ind.
    # Burde veare til at lave om paa saa den tager et array ind
    # Da mismatch = (G_w + G_w) - L smider vi alt i G_w og bruger gamma = 1. Load seattes til 1
    files = []
    for i in range(numberOfNodes):
        arrs = {'Gw': mismatch[:, i],
                'Gs': np.zeros((timesteps)),
                'L': np.ones(timesteps)/1000 ,
                'datalabel': str(i)}
        np.savez('data/test_data_'+str(i)+'.npz', **arrs)
        files.append(str(i)+'.npz')

    print 'files created'

    N = region.Nodes(admat = pathToAdjacency, path='./data/', prefix = "test_data_", files=files, load_filename=None, full_load=None, alphas=1., gammas=1.)

    print 'Starting to Solve'

    # Changing naming conventions of the code to better represent scientific literature
    if flowMode == 'synchronized':
        flowmode = 'square'
    elif flowMode == 'localized':
        flowmode = 'linear'
    else:
        print "flowMode can be either \'synchronized\' or \'localized\'"
    
    M, F = au.solve(N, mode='copper ' + flowmode)

    np.save('results/TestFlow.npy', F)

    
if __name__ == '__main__':
    pass
