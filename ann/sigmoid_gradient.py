import numpy as np
from sigmoid import *

def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))
