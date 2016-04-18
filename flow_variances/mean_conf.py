import numpy as np
import scipy as sp
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    mean, std_error = np.mean(a), scipy.stats.sem(a)
    conf = std_error*sp.stats.t._ppf((1 + confidence)/2., n - 1)
    return m, conf
