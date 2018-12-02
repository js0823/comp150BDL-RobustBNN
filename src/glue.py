import pickle
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, average, Input
from keras.layers import Conv2D, MaxPooling2D
import keras

class BNN:
    def __init__(self, path=None, ISMNIST=True, num_labels=10, LeNet=False,
                 num_samples=50, burnin=100):
        """
        Creates a BNN from a set of posterior samples stored as a PyMC3 
        Multitrace object, allowing for predictions and an interface that
        complements Carlini's attack algorithm code. Note that the returned
        Keras models do not perform the softmax activation on the final layer.
        
        Args:
        - path: String filename for a pickled multitrace object
        - ISMNIST: Whether the model is classifying MNIST, only relevant if
        the model requested is LeNet (i.e. LeNet=True).
        - num_labels: Number of labels (default=10)
        - LeNet: Boolean for whether we trained a LeNet model (default=False)
        - num_samples: Number of posterior samples to use for BNN (default=50)
        - burnin: Length of burn-in phase, key in MCMC inference (default=100)
        """
        self.num_channels = 1 if ISMNIST else 3
        self.image_size = 28 if ISMNIST else 32
        self.num_labels = num_labels

        with open(path, 'rb') as f:
            # Randomly choose posterior samples after burnin phase
            # and create a Keras model for each
            trace = pickle.load(f)['trace']
            ids = np.random.choice(range(burnin, len(trace)), 
                                          num_samples, replace=False)
            models = ([create_lenet(trace.point(i), ISMNIST) for i in ids]
                     if LeNet else [create_model(trace.point(i)) for i in ids])
            
        def average_preds(models, data):
            """
            Takes in a list of posterior samples and data as a Numpy array
            and returns a Keras model instance that averages the input models'
            predictions.

            Args:
            - models: A list of Keras models
            - data: Observations as a Numpy array
            """
            preds = [model(data) for model in models]
            avg = average(preds)
            avg_model = Model(inputs=data, outputs=avg)
            return avg_model

        # Save model returned by average_preds function as model
        inp = Input(shape=(self.num_channels * self.image_size**2,))
        model = average_preds(models, inp)
        self.model = model
        self.model_list = models
    
    def predict(self, data):
        """ Prediction function that wraps average_preds for convenience. """
        return self.model(data)


def create_model(weights):
    """
    Given a set of weights, constructs a simple BNN with only dense layers.
    Returns a Keras model.

    Args:
    - weights: A dict with layer names as keys and Numpy arrays of float 
    weights as values.
    """
    model = Sequential()
    num_layers = len(weights)
    layers = []
    
    # We will infer NN architecture from the shapes of the weight arrays.
    for i, (name, data) in enumerate(list(weights.items())[0::2]):
        
        # Set input_dims only for the first layer
        if i == 0:
            input_dim = data.shape[0]
            layers.append(Dense(data.shape[1], name=name, input_dim=input_dim))
            layers.append(Activation('tanh'))
        else:
            layers.append(Dense(data.shape[1], name=name))
            
            # Only add an activation if it's not the last layer
            if i != num_layers//2 - 1:
                layers.append(Activation('tanh'))

    # Construct model architecture
    for layer in layers:
        model.add(layer)
    
    # Initialize model weights
    model.set_weights(list(weights.values()))
    return model


def create_lenet(weights, ISMNIST):
    model = Sequential()
    input_shape = (28, 28, 1) if ISMNIST else (32,32,3)
    layers = [Conv2D(20, 5, padding='same', input_shape=input_shape),
              Activation('relu'), 
              MaxPooling2D(pool_size=(2,2), strides=(2,2)), 
              Conv2D(50, 5, padding="same"),
              Activation('relu'),
              MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
              Flatten(),
              Dense(500),
              Activation('relu'),
              Dense(10)]

    for layer in layers:
        model.add(layer)

    # print(model.summary())
    model.set_weights(weights.values())
    return model


def check_model_dims(path):
    """
    Small test function to check the dimensions of the weight arrays for
    a posterior sample. Prints out the shapes of one posterior.

    Args:
    - path: String filename, assumed to be a pickled Multitrace object
    """
    with open(path, 'rb') as f:
        trace = pickle.load(f)['trace']
        one_iter = trace.point(40)
        for arr in list(one_iter.values()):
            print(arr.shape)

if __name__ == '__main__':
    path = 'advi-bnn-MNIST-cpurun.pkl'
    BNN = BNN(path)
    fake_data = np.random.rand(64, 784)
    model = BNN.model
    models = BNN.model_list
    print(model.summary())
    print(model.predict(fake_data))