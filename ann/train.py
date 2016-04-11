from cost_function import *
from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_cg
import cv2
import glob
import cPickle as pickle
import time
import datetime
import sys

IM_DIMEN = (75, 75)
y = np.array([[0]])
IM_ARRAY = np.array([np.zeros(IM_DIMEN[0] * IM_DIMEN[1])])

print "Loading images to array..."
for filename in glob.glob("../stop-images/train/positive/*"):
    imarray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    imarray = cv2.resize(imarray, IM_DIMEN)
    IM_ARRAY = np.r_[IM_ARRAY, [imarray.flatten()]]
    y = np.r_[y, [[1]]]

for filename in glob.glob("../stop-images/train/negative/*"):
    imarray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    imarray = cv2.resize(imarray, IM_DIMEN)
    IM_ARRAY = np.r_[IM_ARRAY, [imarray.flatten()]]
    y = np.r_[y, [[0]]]

X = IM_ARRAY[1:, :]
y = y[1:, :]

lam = 1.0

input_layer_size = X.shape[1]
hidden_layer_size = 50
num_labels = 1

np.random.seed(0)

initial_theta = np.random.randn(((input_layer_size + 1) * hidden_layer_size) + ((hidden_layer_size + 1) * num_labels))

def costFunctionWrapper(theta):
    return costFunction(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lam)

def gradientsWrapper(theta):
    return gradients(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lam)

print "Optimizing..."
#optim_theta = fmin_bfgs(costFunctionWrapper, initial_theta, fprime=gradientsWrapper)
optim_theta = fmin_cg(costFunctionWrapper, initial_theta, fprime=gradientsWrapper)

tstamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
tstamplam = tstamp + "_l" + str(lam)
with open("optimized_thetas/optim_theta_" + tstamplam + ".pkl", 'wb') as out:
    pickle.dump(optim_theta, out, pickle.HIGHEST_PROTOCOL)
