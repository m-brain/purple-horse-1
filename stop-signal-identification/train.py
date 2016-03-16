from cost_function import *
from scipy.optimize import fmin_bfgs

initial_theta = np.array([0,0,0,0])
X = np.array([[11,12,11],[11,12,11],[11,12,11],[11,12,11],[11,12,11],[11,12,11],[11,12,11],[11,12,11],[11,12,11],[11,12,11],[11,12,11],[12,12,12],[12,12,12],[12,12,12],[12,12,12],[12,12,12],[12,12,12],[12,12,12],[12,12,12],[12,12,12],[12,12,12],[12,12,12]])
y = np.array([ [1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0] ])

def costFunctionWrapper(theta):
    return costFunction(theta, X, y, 1.0)

def gradientsWrapper(theta):
    return gradients(theta, X, y, 1.0)

opt = fmin_bfgs(costFunctionWrapper, initial_theta, fprime=gradientsWrapper)
print(opt)
