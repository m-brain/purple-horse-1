from cost_function import *
from scipy.optimize import fmin_bfgs
import cv2
import glob
import cPickle as pickle
import time
import datetime

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
initial_theta = np.c_[0, [np.zeros_like(X[0:1,:]).flatten()] ].flatten()
lam =1.5

print "X = ", X.shape
print "y = ", y.shape
print "theta = ", initial_theta.shape
print "lambda = ", lam

def costFunctionWrapper(theta):
    return costFunction(theta, X, y, lam)

def gradientsWrapper(theta):
    return gradients(theta, X, y, lam)

print "Optimizing..."
optim_theta = fmin_bfgs(costFunctionWrapper, initial_theta, fprime=gradientsWrapper)

tstamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

with open("optimized_thetas/optim_theta_" + tstamp + ".pkl", 'wb') as out:
    pickle.dump(optim_theta, out, pickle.HIGHEST_PROTOCOL)
