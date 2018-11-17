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
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import sys, os
import theano.tensor as T

# Package related to cleverhans
import logging
import tensorflow as tf
from cleverhans.loss import CrossEntropy
from cleverhans.utils_tf import model_eval
from cleverhans.train import train
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.utils import AccuracyReport, set_log_level

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

        out = pm.Categorical('out', act_out, observed=ann_output,
                           total_size=Y_train.shape[0]) # IMPORTANT for minibatches
        
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

# Fit and evaluate bnn
def fit_and_eval_bnn(X_train, X_test, Y_train, Y_test, bnn_func, bnn_kwargs=None, sample_kwargs=None):
    if bnn_kwargs is None:
        bnn_kwargs = {}
    
    if sample_kwargs is None:
        #sample_kwargs = {'chains': 1, 'draws': 500, 'init': 'auto'}
        sample_kwargs = {'draws': 500, 'init': 'auto'}
    
    ann_input = theano.shared(X_train.astype(floatX))
    ann_output = theano.shared(Y_train.astype(floatX))
    
    minibatch_x = pm.Minibatch(X_train.astype(floatX), batch_size=500)
    minibatch_y = pm.Minibatch(Y_train.astype(floatX), batch_size=500)

    model = bnn_func(ann_input, ann_output, **bnn_kwargs)

    with model:
        inference = pm.ADVI()
        approx = pm.fit(n=50000, method=inference, more_replacements={ann_input:minibatch_x, ann_output:minibatch_y})
        trace = approx.sample(draws=500)

    #ppc_train = pm.sample_ppc(trace, samples=100)
    #pred_train = mode(ppc_train['out'], axis=0).mode[0, :]

    ann_input.set_value(X_test)
    ann_output.set_value(Y_test)

    with model:
        ppc_test = pm.sample_ppc(trace, samples=100)
        pred_test = mode(ppc_test['out'], axis=0).mode[0, :]
    
    return inference, pred_test, trace

'''
def adversarial_attack(X_train, X_test, Y_train, Y_test, model, num_threads=None):
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Create TF session
    if num_threads:
        config_args = dict(intra_op_parallelism_threads=1)
    else:
        config_args = {}
    sess = tf.Session(config=tf.ConfigProto(**config_args))

    # Use Image Parameters
    img_rows, img_cols, nchannels = X_train.shape[1:4]
    nb_classes = Y_train.shape[1]

    # TF placeholders
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    bim_params = {
        'eps': 0.3,
        'eps_iter': 0.1,
        'clip_min': 0.,
        'clip_max': 1.
    }
    
    bim = BasicIterativeMethod(model, sess=sess)
    adv_x = bim.generate(x, **bim_params)
    preds_adv = model.get_logits(adv_x)
'''

if __name__ == "__main__":
    print("Loading data...")
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_dataset()

    X_train = np.asarray([entry.flatten() for entry in X_train])
    X_val = np.asarray([entry.flatten() for entry in X_val])
    X_test = np.asarray([entry.flatten() for entry in X_test])

    # fit and eval
    #inference, pred_train, pred_test, trace = fit_and_eval_bnn(X_train, X_test, Y_train, Y_test, construct_nn)
    inference, pred_test, trace = fit_and_eval_bnn(X_train, X_test, Y_train, Y_test, construct_nn)

    # Print train accuracy
    #print ("Train accuracy = {:.2f}%".format(100 * np.mean(pred_train == Y_train)))
    #print('Train accuracy = {}%'.format(accuracy_score(Y_train, pred_train) * 100))
    # Print test accuracy
    #print ("Test accuracy = {:.2f}%".format(100 * np.mean(pred_test == Y_test)))
    print('Test accuracy = {}%'.format(accuracy_score(Y_test, pred_test) * 100))

    plt.figure(1)
    plt.plot(-inference.hist)
    plt.ylabel('ELBO')
    plt.xlabel('iteration')
    plt.savefig('advi-elbo.png')
    
    plt.figure(2)
    pm.traceplot(trace)
    plt.savefig('advi-trace.png')