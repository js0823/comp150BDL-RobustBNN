import theano
import pymc3 as pm
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
import pandas as pd
filterwarnings('ignore')
sns.set_style('white')
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import sys, os
import theano.tensor as T


def construct_nn(ann_input, ann_output):
	n_hidden = 50
	
	# Initialize random weights between each layer
	init_1 = np.random.randn(X_train.shape[1], n_hidden).astype(theano.config.floatX)
	init_2 = np.random.randn(n_hidden, n_hidden).astype(theano.config.floatX)
	init_out = np.random.randn(n_hidden,10).astype(theano.config.floatX)
		
	with pm.Model() as neural_network:
		# Weights from input to hidden layer
		weights_in_1 = pm.Normal('w_in_1', 0, sd=1, 
								 shape=(X_train.shape[1], n_hidden), 
								 testval=init_1)
		
		# Weights from 1st to 2nd layer
		weights_1_2 = pm.Normal('w_1_2', 0, sd=1, 
								shape=(n_hidden, n_hidden), 
								testval=init_2)
		
		# Weights from hidden layer to output
		weights_2_out = pm.Normal('w_2_out', 0, sd=1, 
								  shape=(n_hidden,10), 
								  testval=init_out)
		
		# Build neural-network using tanh activation function
		act_1 = pm.math.tanh(pm.math.dot(ann_input, 
										 weights_in_1))
		act_2 = pm.math.tanh(pm.math.dot(act_1, 
										 weights_1_2))
		act_out = T.nnet.softmax(pm.math.dot(act_2, 
											  weights_2_out))

		out = pm.Categorical('out', 
						   act_out,
						   observed=ann_output,
						   total_size=Y_train.shape[0] # IMPORTANT for minibatches
						  )
	return neural_network

def load_dataset():
	# We first define a download function, supporting both Python 2 and 3.
	if sys.version_info[0] == 2:
		from urllib import urlretrieve
	else:
		from urllib.request import urlretrieve

	def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
		print("Downloading %s" % filename)
		urlretrieve(source + filename, filename)

	# We then define functions for loading MNIST images and labels.
	# For convenience, they also download the requested files if needed.
	import gzip

	def load_mnist_images(filename):
		if not os.path.exists(filename):
			download(filename)
		# Read the inputs in Yann LeCun's binary format.
		with gzip.open(filename, 'rb') as f:
			data = np.frombuffer(f.read(), np.uint8, offset=16)
		# The inputs are vectors now, we reshape them to monochrome 2D images,
		# following the shape convention: (examples, channels, rows, columns)
		data = data.reshape(-1, 1, 28, 28)
		# The inputs come as bytes, we convert them to float32 in range [0,1].
		# (Actually to range [0, 255/256], for compatibility to the version
		# provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
		return data / np.float32(256)

	def load_mnist_labels(filename):
		if not os.path.exists(filename):
			download(filename)
		# Read the labels in Yann LeCun's binary format.
		with gzip.open(filename, 'rb') as f:
			data = np.frombuffer(f.read(), np.uint8, offset=8)
		# The labels are vectors of integers now, that's exactly what we want.
		return data

	# We can now download and read the training and test set images and labels.
	X_train = load_mnist_images('train-images-idx3-ubyte.gz')
	y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
	X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
	y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

	# We reserve the last 10000 training examples for validation.
	X_train, X_val = X_train[:-10000], X_train[-10000:]
	y_train, y_val = y_train[:-10000], y_train[-10000:]

	# We just return all the arrays in order, as expected in main().
	# (It doesn't matter how we do this as long as we can read them again.)
	return X_train, y_train, X_val, y_val, X_test, y_test


def tracesToNP(traces):
	posteriorArrays = []
	for trace in traces:
		pArray = []
		for layer in trace:
			pArray.append(trace[layer])
		posteriorArrays.append(np.asarray(pArray))
	return posteriorArrays

if __name__ == "__main__":
	print("Loading data...")
	X_train, Y_train, X_val, Y_val, X_test, y_test = load_dataset()

	X_train = np.asarray([entry.flatten() for entry in X_train])
	X_val = np.asarray([entry.flatten() for entry in X_val])
	X_test = np.asarray([entry.flatten() for entry in X_test])
	# Building a theano.shared variable with a subset of the data to make construction of the model faster.
	# We will later switch that out, this is just a placeholder to get the dimensionality right.
	ann_input = theano.shared(X_train.astype(np.float64))
	ann_output = theano.shared(Y_train.astype(np.float64))

	neural_network = construct_nn(ann_input, ann_output)
	minibatch_x = pm.Minibatch(X_train.astype(np.float64), batch_size=500)
	minibatch_y = pm.Minibatch(Y_train.astype(np.float64), batch_size=500)


	# from pymc3.theanof import set_tt_rng, MRG_RandomStreams
	# set_tt_rng(MRG_RandomStreams(42))

	with neural_network:
		 inference = pm.ADVI()
		 approx = pm.fit(n=10, method=inference, more_replacements={ann_input:minibatch_x, ann_output:minibatch_y})

	# neural_network_minibatch = construct_nn(minibatch_x, minibatch_y)
	# with neural_network_minibatch:
	#     inference = pm.ADVI()
	#     approx = pm.fit(40000, method=inference)
	trace = approx.sample(draws=500)
	
	# print(trace.varnames)
	# print(trace[0])
	# print("type")
	# print(type(trace[0]['w_2_out']))
	# print("summary")
	# print(pm.summary(trace))

	#Given n, spit out n model files or np arrays posterior samples
	plt.plot(-inference.hist)
	plt.ylabel('ELBO')
	plt.xlabel('iteration');
	plt.show()

	# # create symbolic input
	# x = T.matrix('X')
	# # symbolic number of samples is supported, we build vectorized posterior on the fly
	# n = T.iscalar('n')
	# # Do not forget test_values or set theano.config.compute_test_value = 'off'
	# x.tag.test_value = np.empty_like(X_train[:10])
	# n.tag.test_value = 100
	# _sample_proba = approx.sample_node(neural_network_minibatch.out.distribution.p,
	#                                    size=n,
	#                                    more_replacements={ann_input: x})
	# # It is time to compile the function
	# # No updates are needed for Approximation random generator
	# # Efficient vectorized form of sampling is used
	# sample_proba = theano.function([x, n], _sample_proba)

	#set shared var to test data
	ann_input.set_value(X_test)
	ann_output.set_value(y_test)

	with neural_network:
		ppc = pm.sample_ppc(trace, samples=100)
	y_pred = mode(ppc['out'], axis=0).mode[0, :]

	print('Accuracy on test data = {}%'.format(accuracy_score(y_test, y_pred) * 100))

	# pm.traceplot(trace);
	# plt.show()