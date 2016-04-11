from cost_function import *
import cv2
import glob
import cPickle as pickle
import time
import datetime
import sys

thetas = None
with open("optimized_thetas/optim_theta_2016-04-12_00-07-02_l1e-07.pkl", 'rb') as inp:
    thetas = pickle.load(inp)

#x = np.array([[1,2,3,4,5,6]])    
#x = np.array([[3,2,3,4,5,6]])    
#x = np.array([[0,0]])

input_layer_size = x.shape[1]
hidden_layer_size = 1
num_labels = 1
x = np.c_[[1], [x.flatten()]].flatten()

theta1_params = thetas[0: (hidden_layer_size * (input_layer_size + 1))] 
theta2_params = thetas[(hidden_layer_size * (input_layer_size + 1)):] 

theta_1 = theta1_params.reshape(hidden_layer_size, input_layer_size + 1)
theta_2 = theta2_params.reshape(num_labels, (hidden_layer_size + 1))

z2 = x.dot(theta_1.T)
a2 = sigmoid(z2)


z3 = np.c_[[1], [a2]].dot(theta_2.T)
a3 = hx = sigmoid(z3)

print(hx)
