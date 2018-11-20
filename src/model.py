import theano
floatX = theano.config.floatX
import pymc3 as pm
from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
import theano.tensor as T
import loaddata

def create_NN(n_hidden, mean, var, nn_input, nn_output, X_train, Y_train, conv=False):
	if conv is False: # Create BNN
		# Initialize random weights between each layer
		init_1 = np.random.randn(X_train.shape[1], n_hidden).astype(floatX)
		init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)
		init_out = np.random.randn(n_hidden, 10).astype(floatX)

		# Initialize random biases in each layer
		init_b_1 = np.random.randn(n_hidden).astype(floatX)
		init_b_2 = np.random.randn(n_hidden).astype(floatX)
		init_b_out = np.random.randn(1).astype(floatX)
		
		with pm.Model() as neural_network:
			# Weights from input to hidden layer
			weights_in_1 = pm.Normal('w_in_1', mu=mean, sd=np.sqrt(var),
									shape=(X_train.shape[1], n_hidden),
									testval=init_1)

			# Add bias to first hidden layer
			weights_in_b1 = pm.Normal('b_1', mu=mean, sd=np.sqrt(var), 
									shape=(n_hidden), testval=init_b_1)
			
			# Weights from 1st to 2nd layer
			weights_1_2 = pm.Normal('w_1_2', mu=mean, sd=np.sqrt(var), 
									shape=(n_hidden, n_hidden), 
									testval=init_2)

			# Add bias to first hidden layer
			weights_in_b2 = pm.Normal('b_2', mu=mean, sd=np.sqrt(var), 
									shape=(n_hidden), testval=init_b_2)
			
			# Weights from hidden layer to output
			weights_2_out = pm.Normal('w_2_out', mu=mean, sd=np.sqrt(var), 
									shape=(n_hidden, 10), testval=init_out)

			# Add bias to first hidden layer
			weights_in_b_out = pm.Normal('b_out', mu=mean, sd=np.sqrt(var), 
									shape=(10), testval=init_b_out)

			# Build neural-network using tanh activation function
			act_1 = pm.math.tanh(pm.math.dot(nn_input, weights_in_1)+ weights_in_b1)
			act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2) + weights_in_b2)
			act_out = T.nnet.softmax(pm.math.dot(act_2, weights_2_out)+ weights_in_b_out)
			
			out = pm.Categorical('out', act_out, observed=nn_output, total_size=Y_train.shape[0]) # IMPORTANT for minibatches
	
	else:
		from keras.models import Sequential
		from keras.layers import Dense, Conv2D, Flatten

		model = Sequential()
        
	return neural_network
		