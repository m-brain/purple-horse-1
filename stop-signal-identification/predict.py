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

x = np.c_[[1], [imarray.flatten()]].flatten()

theta = None
with open("optimized_thetas/optim_theta_2016-03-17_13-34-16.pkl", 'rb') as inp:
    theta = pickle.load(inp)


print sigmoid(x.dot(theta))
