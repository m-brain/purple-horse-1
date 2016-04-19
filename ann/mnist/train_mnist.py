import sys

sys.path.append('../')

from cost_function import *
from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_cg
from scipy.optimize import fmin_l_bfgs_b
from batch_gradient_descent import fmin_gradient_descent
import cv2
import glob
import cPickle as pickle
import time
import datetime
import json
from mnist import MNIST

print "Loading mnsit data..."
mndata = MNIST('../../../mnist-data/')
(imgs, labels) =  mndata.load_training()

IM_DIMEN = (28, 28)
y = None
X = None

X = np.array(imgs)
lam = 1.0
input_layer_size = X.shape[1]
hidden_layer_size = 50
num_labels = 10

id_mat = np.eye(num_labels)
y = id_mat[labels, :]

try:
    lam = float(sys.argv[1])
    hidden_layer_size = int(sys.argv[2])
except(NameError, IndexError):
    print "Provide correct number of arugments"
    exit()

np.random.seed(0)

initial_theta = np.random.randn(((input_layer_size + 1) * hidden_layer_size) + ((hidden_layer_size + 1) * num_labels))

def costFunctionWrapper(theta):
    return costFunction(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lam)

def gradientsWrapper(theta):
    return gradients(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lam)

print "X = ", X.shape
print "y = ", y.shape
print "theta = ", initial_theta.shape
print "lambda = ", lam
print "hidden_layer_size = ", hidden_layer_size


print "Optimizing..."
(optim_theta, fval, d) = fmin_l_bfgs_b(costFunctionWrapper, initial_theta, fprime=gradientsWrapper)
print "Value of func at the minimum = ", fval
print d

#optim_theta = fmin_bfgs(costFunctionWrapper, initial_theta, fprime=gradientsWrapper)
#optim_theta = fmin_cg(costFunctionWrapper, initial_theta, fprime=gradientsWrapper)
#fmin_gradient_descent(costFunctionWrapper, gradientsWrapper, initial_theta)

model = {'hs': hidden_layer_size, 'optim_theta': optim_theta, 'lam': lam}

tstamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
tstamplam = tstamp + "_l" + str(lam) + "_h" + str(hidden_layer_size)
with open("optimized_thetas/model_mnist_" + tstamplam + ".pkl", 'wb') as out:
    pickle.dump(model, out, pickle.HIGHEST_PROTOCOL)
