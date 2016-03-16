import numpy as np
from sigmoid import * 

def costFunction(theta, X, y, lam):
    X = np.c_[np.ones(X.shape[0]), X]
    m = y.shape[0]
    J = 0

    HX = sigmoid(X.dot(theta.T))
    
    firstPartOfCost = -( (y.T).dot(np.log(HX)))
    secondPartOfCost = ((1.0 - y).T).dot(np.log(1.0-HX))
    regularizationTerm = (lam/(2.0 * m)) * np.sum( np.power(theta[1:],2)) 
    
    J = ((1.0/m) * (firstPartOfCost - secondPartOfCost)) + regularizationTerm
    return J[0]


def gradients(theta, X, y, lam):
    X = np.c_[np.ones(X.shape[0]), X]
    m = y.shape[0]
    J = 0
    grads = np.zeros(theta.shape[0])

    HX = sigmoid(X.dot(theta.T))

    grads[0] = (1.0/m) * ((HX - y.T).dot(X[:,0]))
    grads[1:] = ((1.0/m) * ((HX - y.T).dot(X[:,1:]))) + ((lam/m) * theta[1:])
    return grads
