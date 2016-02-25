#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import colormaps as cmaps


countries = ['AT', 'FI', 'NL', 'BA', 'FR', 'NO', 'BE', 'GB', 'PL', 'BG',
             'GR', 'PT', 'CH', 'HR', 'RO', 'CZ', 'HU', 'RS', 'DE', 'IE',
             'SE', 'DK', 'IT', 'SI', 'ES', 'LU', 'SK', 'EE', 'LV', 'LT']

loads = [np.load('/home/simon/Dropbox/Root/Data/ISET/ISET_country_%s.npz' % countries[node])['L']\
        for node in xrange(len(countries))]


alpha_values = np.linspace(0, 1, 21)
gamma_values = np.linspace(0, 2, 21)


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
            

def storage_needs(backup, quantile):
    storage = np.zeros(len(backup))
    for index, val in enumerate(backup):
        if val >= quantile:
            storage[index] = storage[index] - (val - quantile)
        else:
            storage[index] = storage[index] + (quantile - val)
            if storage[index] > 0:
                storage[index] = 0
    return -min(storage), storage

caps = np.zeros((len(alpha_values), len(gamma_values)))

ag_list = list(product(alpha_values, gamma_values))

def find_caps(country):
    """
    Finds the emergency capacities for a country for every alpha gamma pair arranges them
    in an array for use with np.pcolormesh()
    """
    for index, (a, g) in enumerate(ag_list):
        print 'balancing/%.2f_%.2f.npz' % (a, g)
        ia, ig = divmod(index, len(alpha_values))
        backup = np.load('balancing/%.2f_%.2f.npz' % (a, g))['arr_0'][country]
        q = quantile(99, backup, cutzeros=True)
        caps[ia, ig], storage = storage_needs(backup, q)
        np.savez_compressed('')
    return caps
    

for c in xrange(len(countries)):
    print c
    caps += find_caps(c)

#print caps
a, g = np.mgrid[slice(0, 1.1, 0.05),
                slice(0, 2.2, 0.1)]


loadSum = np.zeros(len(loads[21]))
for l in xrange(len(loads)):
    print l
    loadSum += loads[l]


plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.set_cmap(cmaps.viridis)
plt.pcolormesh(a, g, caps/np.mean(loadSum))
plt.title(r'$\frac{\sum_n \mathcal{K}_n}{\left\langle\sum_n L_n\right\rangle}$', fontsize = 20)
plt.xlabel(r'$\alpha$', fontsize = 20)
plt.ylabel(r'$\gamma$', fontsize = 20)
plt.axis([a.min(), a.max(), g.min(), g.max()])
plt.yticks(np.arange(g.min(), g.max(), np.diff(g)[0, 0]))
plt.colorbar()
plt.show()
