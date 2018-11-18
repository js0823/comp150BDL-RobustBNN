import theano
floatX = theano.config.floatX
import pymc3 as pm
from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
import theano.tensor as T
import loaddata

def create_NN(n_hidden, mean, var, X_train, Y_train, conv=False):
	# Initialize random weights between each layer
	init_1 = np.random.randn(X_train.shape[1], n_hidden).astype(floatX)
	init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)
	init_out = np.random.randn(n_hidden,10).astype(floatX)

	ann_input = theano.shared(X_train.astype(floatX))
	ann_output = theano.shared(Y_train.astype(floatX))
	
	with pm.Model() as neural_network:
		# Weights from input to hidden layer
		weights_in_1 = pm.Normal('w_in_1', mean, sd=np.sqrt(var),
								shape=(X_train.shape[1], n_hidden),
								testval=init_1)
		
		# Weights from 1st to 2nd layer
		weights_1_2 = pm.Normal('w_1_2', mean, sd=np.sqrt(var), 
                                shape=(n_hidden, n_hidden), 
                                testval=init_2)
        
        # Weights from hidden layer to output
		weights_2_out = pm.Normal('w_2_out', mean, sd=np.sqrt(var), 
                                shape=(n_hidden,10), 
                                testval=init_out)

		# Build neural-network using tanh activation function
		act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1))
		act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2))
		act_out = T.nnet.softmax(pm.math.dot(act_2, weights_2_out))
		
		out = pm.Categorical('out', act_out, observed=ann_output, total_size=Y_train.shape[0]) # IMPORTANT for minibatches
        
	return neural_network
		