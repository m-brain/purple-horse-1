from cost_function import *
import glob
import cPickle as pickle
import time
import datetime
import sys
import os
import scipy.misc

print sys.argv[1]

IM_DIMEN = (75, 75)
image_array = scipy.misc.imread(sys.argv[1], flatten=True)
imarray = scipy.misc.imresize(image_array, IM_DIMEN)

model_file = ""

if not model_file:
    model_file = max(glob.iglob('./optimized_thetas/*.pkl'), key=os.path.getctime)

print model_file

model = None
with open(model_file, 'rb') as inp:
    model = pickle.load(inp)

thetas = model['optim_theta']
hidden_layer_size = model['hs']

print "Theta size = ", thetas.shape
print "Hidden layer size = ", hidden_layer_size
print "Lambda = ", model['lam']

x = imarray.flatten()
input_layer_size = x.shape[0]
num_labels = 1

x = np.c_[[1], [imarray.flatten()]].flatten()

theta1_params = thetas[0: (hidden_layer_size * (input_layer_size + 1))]
theta2_params = thetas[(hidden_layer_size * (input_layer_size + 1)):] 

theta_1 = theta1_params.reshape(hidden_layer_size, input_layer_size + 1)
theta_2 = theta2_params.reshape(num_labels, (hidden_layer_size + 1))

z2 = x.dot(theta_1.T)
a2 = sigmoid(z2)


z3 = np.c_[[1], [a2]].dot(theta_2.T)
a3 = hx = sigmoid(z3)

print "y^ = ", float(hx)
