import sys
sys.path.append('../')
from cost_function import *
import cv2
import glob
import cPickle as pickle
import time
import datetime
import os
import json
from mnist import MNIST

print "Loading mnsit data..."
mndata = MNIST('../../../mnist-data/')
(imgs, labels) =  mndata.load_training()

IM_DIMEN = (28, 28)
IM_INDEX = 500
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


x = np.array(imgs[IM_INDEX])
sigImageArray = x.reshape(IM_DIMEN)
input_layer_size = x.shape[0]
num_labels = 10
print sigImageArray.shape

x = np.c_[[1], [x]].flatten()
print x.shape

theta1_params = thetas[0: (hidden_layer_size * (input_layer_size + 1))]
theta2_params = thetas[(hidden_layer_size * (input_layer_size + 1)):] 

theta_1 = theta1_params.reshape(hidden_layer_size, input_layer_size + 1)
theta_2 = theta2_params.reshape(num_labels, (hidden_layer_size + 1))

z2 = x.dot(theta_1.T)
a2 = sigmoid(z2)


z3 = np.c_[[1], [a2]].dot(theta_2.T)
a3 = hx = sigmoid(z3)

print labels[IM_INDEX]
yvalue = hx.flatten().tolist()
print "Predicts:"
print hx
print "y^ = ", max(yvalue)
print yvalue.index(max(yvalue))
cv2.imshow('Digit', cv2.resize( np.uint8(sigImageArray) , (280,280) ))
cv2.waitKey(0)
