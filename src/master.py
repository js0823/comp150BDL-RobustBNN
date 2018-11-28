import model
import infer
import loaddata
import numpy as np
import theano
floatX = theano.config.floatX
import theano.tensor as T
import pymc3 as pm
from sklearn.metrics import accuracy_score

###################### Configurations ########################
inference_alg = 'nuts'  # Can be advi, nuts
modeltype = 'bnn' # can be bnn or cnn
data = 'MNIST' # can be MNIST or CIFAR10
h_layer_size = 50
mean = 0
var = 1
nPosterior_samples = 200
##############################################################

def run_config(modeltype, inference_alg, data):
	posterior_sample_filename = inference_alg + '-' + modeltype + '-' + data + '.pkl'
	if data is 'MNIST':
		X_train, Y_train, X_test, Y_test = loaddata.load_MNIST_dataset()
	elif data is 'CIFAR10':
		X_train, Y_train, X_test, Y_test = loaddata.load_CIFAR10_dataset()
		Y_train = np.concatenate(Y_train)
		Y_test = np.concatenate(Y_test)
		Y_test = np.uint8(Y_test)
	
	if modeltype is 'bnn':
		X_train = np.asarray([entry.flatten() for entry in X_train])
		X_test = np.asarray([entry.flatten() for entry in X_test])

	nn_input = theano.shared(X_train.astype(floatX))
	nn_output = theano.shared(Y_train.astype(floatX))

	# Get neural network model
	nn = model.create_NN(h_layer_size, mean, var, nn_input, nn_output, X_train, Y_train)

	# Train the model
	if inference_alg is 'advi':
		pred_test, trace = infer.train_model('advi', nn, nPosterior_samples, nn_input, nn_output, X_train, Y_train, X_test, Y_test)
	elif inference_alg is 'nuts':
		pred_test, trace = infer.train_model('nuts', nn, nPosterior_samples, nn_input, nn_output, X_train, Y_train, X_test, Y_test)
	
	infer.save_posterior(nn, trace, posterior_sample_filename)
	
	accuracies = accuracy_score(Y_test, pred_test)

	#return accuracies, detect_rates
	return accuracies

if __name__ == "__main__":
	# Run the main program
	accuracies = run_config(modeltype, inference_alg, data)
	print("Test accuracy = {}%".format(accuracies * 100))