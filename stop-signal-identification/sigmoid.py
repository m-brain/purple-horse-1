import numpy as np

def sigmoid(z):
    g = 1.0/(1.0 + np.power(np.e, -z))
    return g
