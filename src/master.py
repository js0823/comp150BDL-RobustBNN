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
modeltype = 'bnn' # can be bnn or cnn
data = 'MNIST' # can be MNIST or CIFAR10
h_layer_size = 50
mean = 0
std = 1
nPosterior_samples = 500
posterior_sample_filename = 'advi-sample.pkl'
##############################################################

def run_config(modeltype, inference_alg, data):
	if data is 'MNIST':
		X_train, Y_train, X_val, Y_val, X_test, Y_test = loaddata.load_MNIST_dataset()
	
	X_train = np.asarray([entry.flatten() for entry in X_train])
	#X_val = np.asarray([entry.flatten() for entry in X_val])
	#X_test = np.asarray([entry.flatten() for entry in X_test])

	nn_input = theano.shared(X_train.astype(floatX))
	nn_output = theano.shared(Y_train.astype(floatX))

	# Get neural network model
	nn = model.create_NN(h_layer_size, mean, std, nn_input, nn_output, X_train, Y_train)

	# Train the model
	if inference_alg is 'advi':
		pred_train, trace = infer.train_model('advi', nn, nPosterior_samples, nn_input, nn_output, X_train, Y_train)
	elif inference_alg is 'nuts':
		pred_train, trace = infer.train_model('nuts', nn, nPosterior_samples, nn_input, nn_output, X_train, Y_train)
	
	infer.save_posterior(nn, trace, posterior_sample_filename)
	
	accuracies = accuracy_score(Y_train, pred_train) * 100

	#return accuracies, detect_rates
	return accuracies

if __name__ == "__main__":
	# Run the main program
	accuracies = run_config(modeltype, inference_alg, data)