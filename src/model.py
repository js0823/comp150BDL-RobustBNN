import theano
floatX = theano.config.floatX
import pymc3 as pm
from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
import theano.tensor as T
import loaddata
import math
import lasagne

# For Bayesian CNN
import keras
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D

# For adding gaussian weights to Bayesian CNN
class GaussWeights(object):
    def __init__(self):
        self.count = 0
    def __call__(self, shape):
        self.count += 1
        return pm.Normal('w%d' % self.count, mu=0, sd=.1,
                         testval=np.random.normal(size=shape).astype(np.float32),
                         shape=shape)

def create_NN(n_hidden, mean, var, nn_input, nn_output, X_train, Y_train, conv=False, init=GaussWeights()):

	def build_bcnn(init):
		l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
										input_var=nn_input)

		# Add a fully-connected layer of 800 units, using the linear rectifier, and
		# initializing weights with Glorot's scheme (which is the default anyway):
		n_hid1 = 800
		l_hid1 = lasagne.layers.DenseLayer(
			l_in, num_units=n_hid1,
			nonlinearity=lasagne.nonlinearities.tanh,
			b=init,
			W=init
		)

		n_hid2 = 800
		# Another 800-unit layer:
		l_hid2 = lasagne.layers.DenseLayer(
			l_hid1, num_units=n_hid2,
			nonlinearity=lasagne.nonlinearities.tanh,
			b=init,
			W=init
		)

		# Finally, we'll add the fully-connected output layer, of 10 softmax units:
		l_out = lasagne.layers.DenseLayer(
			l_hid2, num_units=10,
			nonlinearity=lasagne.nonlinearities.softmax,
			b=init,
			W=init
		)
		
		prediction = lasagne.layers.get_output(l_out)
		
		# 10 discrete output classes -> pymc3 categorical distribution
		out = pm.Categorical('out', 
							prediction,
							observed=target_var)
		
		return out

	if conv is False: # Create BNN
		# Initialize random weights between each layer
		init_1 = np.random.randn(X_train.shape[1], n_hidden).astype(floatX)
		init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)
		init_out = np.random.randn(n_hidden, 10).astype(floatX)

		# Initialize random biases in each layer
		init_b_1 = np.random.randn(n_hidden).astype(floatX)
		init_b_2 = np.random.randn(n_hidden).astype(floatX)
		init_b_out = np.random.randn(1).astype(floatX)
		
		with pm.Model() as model:
			# Weights from input to hidden layer
			weights_in_1 = pm.Normal('w_in_1', mu=mean, sd=math.sqrt(var/n_hidden),
									shape=(X_train.shape[1], n_hidden),
									testval=init_1)

			# Add bias to first hidden layer
			weights_in_b1 = pm.Normal('b_1', mu=mean, sd=math.sqrt(var/n_hidden), 
									shape=(n_hidden), testval=init_b_1)
			
			# Weights from 1st to 2nd layer
			weights_1_2 = pm.Normal('w_1_2', mu=mean, sd=math.sqrt(var/n_hidden), 
									shape=(n_hidden, n_hidden), 
									testval=init_2)

			# Add bias to second hidden layer
			weights_in_b2 = pm.Normal('b_2', mu=mean, sd=math.sqrt(var/n_hidden), 
									shape=(n_hidden), testval=init_b_2)
			
			# Weights from 2nd layer to output
			weights_2_out = pm.Normal('w_2_out', mu=mean, sd=math.sqrt(var/n_hidden), 
									shape=(n_hidden, 10), testval=init_out)

			# Add bias to last hidden layer
			weights_in_b_out = pm.Normal('b_out', mu=mean, sd=math.sqrt(var/n_hidden), 
									shape=(10), testval=init_b_out)

			# Build neural-network using tanh activation function
			act_1 = pm.math.sigmoid(pm.math.dot(nn_input, weights_in_1) + weights_in_b1)
			act_2 = pm.math.sigmoid(pm.math.dot(act_1, weights_1_2) + weights_in_b2)
			act_out = T.nnet.softmax(pm.math.dot(act_2, weights_2_out) + weights_in_b_out)
			
			out = pm.Categorical('out', act_out, observed=nn_output, total_size=Y_train.shape[0]) # IMPORTANT for minibatches
	
	else: # Bayesian Convolutional neural network (Lenet)
		'''
		model = Sequential()
		# first set of CONV => RELU => POOL
		model.add(Conv2D(20, kernel_size=(3, 3), padding="same", activation='relu',
							input_shape=(X_train.shape[1], X_train.shape[2], 1)))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# second set of CONV => RELU => POOL
		model.add(Conv2D(50, kernel_size=(5, 5), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))

		# softmax classifier
		model.add(Dense(10))
		model.add(Activation("softmax"))
		'''

		'''
		with pm.Model() as model:
			i = Input(tensor=nn_input, shape=(X_train.shape[1], X_train.shape[2]))
			layer1 = Conv2D(20, kernel_size=(5, 5), kernel_initializer=init, activation='relu')(i)
			layer1Pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer1)

			layer2 = Conv2D(50, kernel_size=(3, 3), kernel_initializer=init, activation='relu')(layer1Pool)
			layer2Pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer2)

			flatten = Flatten()(layer2Pool)
			dense1 = Dense(500, kernel_initializer=init, activation='relu')(flatten)

			dense2 = Dense(10, kernel_initializer=init, activation='softmax')(dense1)

			out = pm.Categorical('out', dense2, observed=nn_output, total_size=Y_train.shape[0])
		'''
		
		# TODO: I think I need to use lasagne. See bayesian_cnn.py
	
	return model