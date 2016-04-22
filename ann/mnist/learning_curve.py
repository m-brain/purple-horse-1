from predict import *
from mnist import MNIST
import cPickle as pickle

print "Loading mnsit data..."
mndata = MNIST('../../../mnist-data/')
(imgs, labels) =  mndata.load_training()

IM_INDEX = 1000
model_file = "./optimized_thetas/model_mnist_2016-04-22_08-31-43_l0.5_h300.pkl"
predictions = []

print model_file

model = None
with open(model_file, 'rb') as inp:
    model = pickle.load(inp)

for idx, img in enumerate(imgs):
#    print idx
#    print labels[idx]
    predictions.append(predict(model, np.array(img)))

diff_predictions = np.uint8(np.array(predictions) == labels)
print sum(diff_predictions)
wrong_prediction_idxs = [idx for idx, x in enumerate(diff_predictions) if x == 0]

print (sum(diff_predictions)/float(len(imgs))) * 100.0
