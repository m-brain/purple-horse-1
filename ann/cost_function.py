import numpy as np
from sigmoid import *
from sigmoid_gradient import *

def costFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lam):
    theta1_params = nn_params[0: (hidden_layer_size * (input_layer_size + 1))] 
    theta2_params = nn_params[(hidden_layer_size * (input_layer_size + 1)):] 

    theta_1 = theta1_params.reshape(hidden_layer_size, input_layer_size + 1)
    theta_2 = theta2_params.reshape(num_labels, (hidden_layer_size + 1))

    m = X.shape[0]
    
    Z2 = np.c_[np.ones(m), X].dot(theta_1.T)
    A2 = sigmoid(Z2)

    Z3 = np.c_[np.ones(A2.shape[0]), A2].dot(theta_2.T)
    A3 = HX = sigmoid(Z3)
    
    firstPartOfCost = -( (y) * np.log(HX) )
    secondPartOfCost = ((1.0 - y) * np.log(1.0-HX))

    allThetas = np.append(theta_1.flatten()[1:], theta_2.flatten()[1:])
    regularizationTerm = (lam/(2.0 * m)) * np.sum( np.power(allThetas, 2)) 
    
    J = ((1.0/m) * np.sum(np.sum(firstPartOfCost - secondPartOfCost)) ) + regularizationTerm

    return J

    
def gradients(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lam):
    theta1_params = nn_params[0: (hidden_layer_size * (input_layer_size + 1))] 
    theta2_params = nn_params[(hidden_layer_size * (input_layer_size + 1)):] 

    theta_1 = theta1_params.reshape(hidden_layer_size, input_layer_size + 1)
    theta_2 = theta2_params.reshape(num_labels, (hidden_layer_size + 1))

    m = X.shape[0]

    Z2 = np.c_[np.ones(m), X].dot(theta_1.T)
    A2 = sigmoid(Z2)

    Z3 = np.c_[np.ones(A2.shape[0]), A2].dot(theta_2.T)
    A3 = HX = sigmoid(Z3)

    d3 = A3 - y;
    d2 = d3.dot(theta_2[:, 1:]) * sigmoidGradient(Z2)

    Delta1 = d2.T.dot(np.c_[np.ones(m), X])
    Delta2 = d3.T.dot(np.c_[np.ones(A2.shape[0]), A2])

    theta_1[:, 0] = 0
    theta_2[:, 0] = 0

    Theta1Grad = ((1.0/m) * Delta1) + ((lam/m) * theta_1)
    Theta2Grad = ((1.0/m) * Delta2) + ((lam/m) * theta_2)
    grads = np.append(Theta1Grad.flatten(), Theta2Grad.flatten())
    return grads
