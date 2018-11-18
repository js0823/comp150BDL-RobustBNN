import model
import infer
import loaddata
import numpy as np
import theano
floatX = theano.config.floatX
import pymc3 as pm

def run_config(model, inference_alg, data):
	X_train, Y_train, X_val, Y_val, X_test, Y_test = loaddata.load_MNIST_dataset()
	
	X_train = np.asarray([entry.flatten() for entry in X_train])
	X_val = np.asarray([entry.flatten() for entry in X_val])
	X_test = np.asarray([entry.flatten() for entry in X_test])
	
	minibatch_x = pm.Minibatch(X_train.astype(floatX), batch_size=500)
	minibatch_y = pm.Minibatch(Y_train.astype(floatX), batch_size=500)

	nn = model.create_NN(50, 0, 1, X_train, Y_train)
	#return accuracies, detect_rates

if __name__ == "__main__":
	# Run the main program
	pass