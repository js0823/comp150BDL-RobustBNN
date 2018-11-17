import theano
floatX = theano.config.floatX
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
    init_1 = np.random.randn(X_train.shape[1], n_hidden).astype(floatX)
    init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)
    init_out = np.random.randn(n_hidden,10).astype(floatX)
        
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

        out = pm.Categorical('out', act_out, observed=ann_output,  total_size=Y_train.shape[0])

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

def fit_and_eval_bnn(X_train, X_test, Y_train, Y_test, bnn_func, bnn_kwargs=None, sample_kwargs=None):
    if bnn_kwargs is None:
        bnn_kwargs = {}
    
    if sample_kwargs is None:
        #sample_kwargs = {'cores': 1, 'draws': 500, 'init': 'auto'} // advi+adapt_diag is faster
        sample_kwargs = {'cores': 1, 'init': 'auto', 'draws': 500}
    
    ann_input = theano.shared(X_train.astype(floatX))
    ann_output = theano.shared(Y_train.astype(floatX))

    model = bnn_func(ann_input, ann_output, **bnn_kwargs)

    with model:
        # pm.sample = Default draw is 500
        trace = pm.sample(**sample_kwargs)

        #ppc_train = pm.sample_ppc(trace, samples=100)
        #pred_train = mode(ppc_train['out'], axis=0).mode[0, :]

    ann_input.set_value(X_test)
    ann_output.set_value(Y_test)

    with model:
        ppc_test = pm.sample_ppc(trace, samples=100)
        pred_test = mode(ppc_test['out'], axis=0).mode[0, :]
    
    #return pred_train, pred_test, trace
    return pred_test, trace

if __name__ == "__main__":
    print("Loading data...")
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_dataset()

    X_train = np.asarray([entry.flatten() for entry in X_train])
    X_val = np.asarray([entry.flatten() for entry in X_val])
    X_test = np.asarray([entry.flatten() for entry in X_test])

    # fit and eval
    #pred_train, pred_test, trace = fit_and_eval_bnn(X_train, X_test, Y_train, Y_test, construct_nn)
    pred_test, trace = fit_and_eval_bnn(X_train, X_test, Y_train, Y_test, construct_nn)

    # Print train accuracy
    #print ("Train accuracy = {:.2f}%".format(100 * np.mean(pred_train == Y_train)))
    #print('Train accuracy = {}%'.format(accuracy_score(Y_train, pred_train) * 100))
    # Print test accuracy
    #print ("Test accuracy = {:.2f}%".format(100 * np.mean(pred_test == Y_test)))
    print('Test accuracy = {}%'.format(accuracy_score(Y_test, pred_test) * 100))
    pm.traceplot(trace)
    plt.savefig('nuts-trace.png')
    pm.energyplot(trace)
    plt.savefig('nuts-trace-energyplot.png')
    pm.plot_posterior(trace)
    plt.savefig('nuts-trace-plot_posterior.png')


'''
    with neural_network:
        trace = pm.sample()
    
    ann_input.set_value(X_test)
    ann_output.set_value(y_test)

    with neural_network:
        ppc = pm.sample_ppc(trace, samples=100)
    
    y_pred = mode(ppc['out'], axis=0).mode[0, :]
    
    print('Accuracy on test data = {}%'.format(accuracy_score(y_test, y_pred) * 100))
    #plt.plot(-step.hist)
    #plt.ylabel('ELBO')
    #plt.xlabel('iteration')
    #plt.show()
'''