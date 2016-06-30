from cost_function import *
from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_cg
from scipy.optimize import fmin_l_bfgs_b
import glob
import cPickle as pickle
import time
import datetime
import sys
import scipy.misc

IM_DIMEN = (75, 75)
y = np.array([[0]])
IM_ARRAY = np.array([np.zeros(IM_DIMEN[0] * IM_DIMEN[1])])

print "Loading images to array..."
for filename in glob.glob("../stop-images/train/positive/*"):
    image_array = scipy.misc.imread(filename, flatten=True)
    imarray = scipy.misc.imresize(image_array, IM_DIMEN)
    IM_ARRAY = np.r_[IM_ARRAY, [imarray.flatten()]]
    y = np.r_[y, [[1]]]

for filename in glob.glob("../stop-images/train/negative/*"):
    image_array = scipy.misc.imread(filename, flatten=True)
    imarray = scipy.misc.imresize(image_array, IM_DIMEN)
    IM_ARRAY = np.r_[IM_ARRAY, [imarray.flatten()]]
    y = np.r_[y, [[0]]]

X = IM_ARRAY[1:, :]
y = y[1:, :]

lam = 1.0

input_layer_size = X.shape[1]
hidden_layer_size = 50
num_labels = 1

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
(optim_theta,fval,d) = fmin_l_bfgs_b(costFunctionWrapper, initial_theta, fprime=gradientsWrapper)
print "Value of func at the minimum = ", fval
print d

#optim_theta = fmin_bfgs(costFunctionWrapper, initial_theta, fprime=gradientsWrapper)
#optim_theta = fmin_cg(costFunctionWrapper, initial_theta, fprime=gradientsWrapper)

model = {'hs': hidden_layer_size, 'optim_theta': optim_theta, 'lam': lam}

tstamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
tstamplam = tstamp + "_l" + str(lam) + "_h" + str(hidden_layer_size)
with open("optimized_thetas/model_" + tstamplam + ".pkl", 'wb') as out:
    pickle.dump(model, out, pickle.HIGHEST_PROTOCOL)
