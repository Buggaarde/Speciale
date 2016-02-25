#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pylab
import networkx as nx
from syncSolver import syncSolver
from itertools import izip
import os
import os.path

from functions import solve_for_new_data, solve_for_new_dataNEW
import timeit

# -- Generate the europe network --
euroAdjMat = np.loadtxt('/home/simon/Dropbox/Root/SandBox/eadmat.txt')
np.fill_diagonal(euroAdjMat, 0)
edge_indicies = euroAdjMat > 0
euroAdjMat[edge_indicies] = 1
europe = nx.from_numpy_matrix(euroAdjMat)
# -- --

# -- Give the nodes proper data labels and positions --
countries = ['AT', 'FI', 'NL', 'BA', 'FR', 'NO', 'BE', 'GB', 'PL', 'BG',
             'GR', 'PT', 'CH', 'HR', 'RO', 'CZ', 'HU', 'RS', 'DE', 'IE',
             'SE', 'DK', 'IT', 'SI', 'ES', 'LU', 'SK', 'EE', 'LV', 'LT']

positions = [[0.55,0.45],[.95,1.1],[0.40,0.85],[0.65,0.15],[0.15,0.60],
             [0.5,1.1],[0.275,0.775],[0.10,1.05],[0.75,0.8],[0.9,0.0],
             [0.7,0.0],[0.0,0.15],[0.4,0.45],[0.75,0.3],[1.0,0.15],
             [0.75,0.60],[1.0,0.45],[0.85,0.15],[0.45,0.7],[0.0,0.95],
             [0.75,1.0],[0.5,0.875],[0.4,0.2],[0.55,0.3],[0.15,0.35],
             [0.325,0.575],[0.90,0.55],[1.03,0.985],[0.99,0.85],[0.925,0.72]]

for node in europe.nodes_iter():
    europe.node[node]['datalabel'] = countries[node]

country_labels = dict(izip(range(len(countries)), countries))
country_pos = dict(izip(range(len(countries)), positions))
#nx.draw(europe, pos = country_pos, labels = country_labels)
# -- --

# -- Load the ISET data into network --
for node in europe.nodes_iter():
    data = np.load('/home/simon/Dropbox/Root/Data/ISET/ISET_country_%s.npz' % countries[node])
    # Wind and solar energy are normed, so we need to multiply by the mean of the loads
    mean_load = np.mean(data['L'])
    europe.node[node]['Gw'] = mean_load*data['Gw']
    europe.node[node]['Gs'] = mean_load*data['Gs']
    europe.node[node]['Load'] = data['L']
#     europe.node[node]['Mismatch'] = alpha*data['Gw'] + (1 - alpha)*data['Gs'] - data['L']
    europe.node[node]['Balance'] = np.zeros(len(data['Gw']))
    europe.node[node]['Injection Pattern'] = np.zeros(len(data['Gw']))
# -- --

# -- Calculate the flows in the network and save to disk --
alpha_list = [0.05*x for x in xrange(21)] # alpha is the split between wind and solar generation, alpha = 1 means 100% wind and vice versa.
path = 'Results/'
filename = 'europe_by_alpha'

solve_for_new_dataNEW(europe, alpha_list, path = path, filename=filename)
# -- --

# -- Load data and extract balances --
start = timeit.default_timer()
dataset = np.load('%s%s.npz' % (path, filename))
balance_alpha_dict = dataset['balance_alpha_dict'].item()
ip_alpha_dict = dataset['ip_alpha_dict'].item()
flow_alpha_dict = dataset['flow_alpha_dict'].item()
alpha_list = sorted(dataset['alpha_list'])
end = timeit.default_timer()
print end - start
#alpha_list = alpha_list[0:-1:2]
# -- --

# -- Define helper function --
def calc_backup(country, alpha, cutzeros=False):
    balances = balance_alpha_dict[alpha][country]
    if cutzeros:
        backup = -balances.clip(max = 0)
        return backup[np.nonzero(backup)]
    else:
        return -balances.clip(max = 0)
# -- --

# -- Calculate 99% quantile --
def quantile(quantile, dataset, cutzeros=True):
    """
    Takes a list of numbers, converts it to a list without zeros
    and returns the value of the 99% quantile.
    """
    if cutzeros:
        # Removing zeros
        dataset = dataset[np.nonzero(dataset)]
    # Convert to numpy histogram
    hist, bin_edges = np.histogram(dataset, bins = 10000, normed = True)
    dif = np.diff(bin_edges)[0]
    q = 0
    for index, val in enumerate(reversed(hist)):
        q += val*dif
        if q > 1 - float(quantile)/100:
            #print 'Found %.3f quantile' % (1 - q)
            return bin_edges[-index]
        else:
            'Something went wrong...'
# -- --

# -- Calculate 99% quantile of backup for all alpha values and plot results --
def backup_quantile(country, alpha_list):
    backups = []
    for alpha in alpha_list:
        backup = calc_backup(country, alpha, cutzeros = True)
        backups.append(quantile(99, backup))
    return backups

DKBackups = backup_quantile(24, alpha_list)
#plt.plot(alpha_list, DKBackups)
# plt.show()     
# -- --

# -- Calculate maximum storage needed --
def storage_needs(backup, quantile, alpha):
    storage = np.zeros(len(backup))
    for index, val in enumerate(backup):
        if val >= quantile:
            storage[index] = storage[index] - (val - quantile)
        else:
            storage[index] = storage[index] + (quantile - val)
            if storage[index] > 0:
                storage[index] = 0
    return -min(storage), storage

backup = calc_backup(21, 0.8, cutzeros = True)
q = quantile(99, backup)
maxStorage, storage = storage_needs(backup, q, 0.8)
plt.plot(storage/np.mean(europe.node[21]['Load']))
plt.show()

def storage_needs_alpha_list(country, alpha_list):
    """
    Parameters
    ----------
    alpha_list:
    A list of the alpha values used to calculate the quantiles.
    """
    storage_needs_alpha_list = np.zeros(len(alpha_list))
    for index, alpha in enumerate(alpha_list):
        backups = calc_backup(country, alpha, cutzeros = True)
        q = quantile(99, backups)
        maxStorage, storageNeeds = storage_needs(backups, q, alpha)
        storage_needs_alpha_list[index] = maxStorage
    return storage_needs_alpha_list

def storage_needs_network(Graph, alpha_list):
    loads = np.zeros(len(Graph.node[0]['Load']))
    maxStorage = np.zeros(len(alpha_list))
    for country in Graph.nodes_iter():
        print country
        maxStorage += storage_needs_alpha_list(country, alpha_list)
        loads += Graph.node[country]['Load']
    return maxStorage/np.mean(loads)

    
DKStorageNeeds = storage_needs_alpha_list(21, alpha_list)
EUStorageNeeds = storage_needs_network(europe, alpha_list)
plt.plot(alpha_list, EUStorageNeeds)
plt.title(r'$\gamma = 1$', fontsize = 20)
plt.xlabel(r'$\alpha$', fontsize = 20)
plt.ylabel(r'$\frac{\sum_n \mathcal{K}_n}{\langle \sum_n L_n \rangle}$', fontsize = 20)
plt.grid(True)
pylab.savefig('../LaTeX/Graphics/emergency_storage_gamma1.pdf', bbox_inches = 'tight')
plt.show()




# # -- Calculate flow quantiles --
# def flow_quantile(link, alpha_list):
#     flows = []
#     for alpha in alpha_list:
#         flows.append(quantile(99, flow_alpha_dict[alpha][link]))
#     return flows

# flows = np.zeros(len(alpha_list))
# for link in europe.edges_iter():
#     print link
#     flow = flow_quantile(link, alpha_list)
#     flows = flows + flow

# print len(flows)
# plt.plot(alpha_list, flows)
# plt.show()








