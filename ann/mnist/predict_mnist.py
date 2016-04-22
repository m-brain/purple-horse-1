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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from mnist import MNIST

print "Loading mnsit data..."
mndata = MNIST('../../../mnist-data/')
(imgs, labels) =  mndata.load_testing()

IM_DIMEN = (28, 28)
IM_INDEX = 100
model_file = "./optimized_thetas/model_mnist_2016-04-20_07-17-02_l0.0_h50.pkl"

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

x = np.c_[[1], [x]].flatten()

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

first_act = theta_1[:,1:][10]

fig = plt.figure()
plt.axis('off')

print theta_2.shape

ele_w2 = np.c_[[1], [a2]] * theta_2 

'''
gs = gridspec.GridSpec(2, 5)
for idx, row in enumerate((ele_w2)[:,1:]):
    fig.add_subplot(gs[(idx/5), idx%5])
    print row.shape
    img = plt.imshow(np.float32(row.reshape(10,5)), cmap='gray')
    img.axes.get_xaxis().set_visible(False)
    img.axes.get_yaxis().set_visible(False)
'''

gs = gridspec.GridSpec(5, 10)
for idx, row in enumerate((x * theta_1)[:,1:]):
    fig.add_subplot(gs[(idx/10), idx%10])
    img = plt.imshow(np.float32(row.reshape(28,28)), cmap='gray')
    img.axes.get_xaxis().set_visible(False)
    img.axes.get_yaxis().set_visible(False)
fig2 = plt.figure()
plt.imshow(np.uint8(sigImageArray), cmap='gray')

plt.show()



    
#cv2.imshow(str(idx), cv2.resize(np.uint8(row.reshape(28,28)), (280, 280)) )
#cv2.imshow('Digit', cv2.resize( np.uint8(sigImageArray) , (280,280) ))
#cv2.waitKey(0)
#cv2.waitKey(0)
