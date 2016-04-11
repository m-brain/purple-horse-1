from cost_function import *
import cv2
import glob
import cPickle as pickle
import time
import datetime
import sys

print sys.argv[1]

IM_DIMEN = (75, 75)
imarray = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
imarray = cv2.resize(imarray, IM_DIMEN)


thetas = None
with open("optimized_thetas/optim_theta_2016-04-12_00-56-30_l1.0.pkl", 'rb') as inp:
    thetas = pickle.load(inp)

x = imarray.flatten()
input_layer_size = x.shape[0]
hidden_layer_size = 50
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

print(hx)
