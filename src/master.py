import model
import infer
import loaddata
import numpy as np
import theano
floatX = theano.config.floatX
import pymc3 as pm
from sklearn.metrics import accuracy_score

###################### Configurations ########################
inference_alg = 'advi'  # Can be advi, nuts
model = 'bnn' # can be bnn or cnn
data = 'MNIST' # can be MNIST or CIFAR10
h_layer_size = 50
mean = 0
std = 1
nPosterior_samples = 500
##############################################################

def run_config(model, inference_alg, data):
	X_train, Y_train, X_val, Y_val, X_test, Y_test = loaddata.load_MNIST_dataset()
	
	X_train = np.asarray([entry.flatten() for entry in X_train])
	#X_val = np.asarray([entry.flatten() for entry in X_val])
	#X_test = np.asarray([entry.flatten() for entry in X_test])

	nn_input = theano.shared(X_train.astype(floatX))
	nn_output = theano.shared(Y_train.astype(floatX))

	# Get neural network model
	nn = model.create_NN(h_layer_size, mean, std, nn_input, nn_output, X_train, Y_train)

	# Train the model
	pred_train, trace = infer.train_model('advi', nn, nPosterior_samples, X_train, X_test, Y_train, Y_test)
	accuracies = accuracy_score(Y_train, pred_train) * 100

	#return accuracies, detect_rates
	return accuracies

if __name__ == "__main__":
	# Run the main program
	accuracies = run_config(model, inference_alg, data)