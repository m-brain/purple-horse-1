import sys
sys.path.append('../')
from cost_function import *
import numpy as np

def predict(model, x):
    thetas = model['optim_theta']
    hidden_layer_size = model['hs']

 #   print "Theta size = ", thetas.shape
 #   print "Hidden layer size = ", hidden_layer_size
#    print "Lambda = ", model['lam']

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

#    print "Predicts:"
    yvalue = hx.flatten().tolist()
#    print yvalue.index(max(yvalue))
    
    return yvalue.index(max(yvalue))
