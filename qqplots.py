import numpy as np
from scipy.optimize import curve_fit
from __future__ import division

def GEV(x, mu, sigma, xi):
    return 1/sigma*(1 + xi*(x - mu)/sigma)**((-1/xi)-1)*np.exp(-(1 + xi*(x - mu)/sigma)**(-1/xi))

xdata = 
