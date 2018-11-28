import pickle
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras

class BNN:
    def __init__(self, path=None, ISMNIST=True, num_labels=10, LeNet=True):
        self.num_channels = 1 if ISMNIST else 3
        self.image_size = 28 if ISMNIST else 32
        self.num_labels = num_labels

        with open(path, 'rb') as f:
            posteriors = pickle.load(f)
            models = ([create_lenet(p, ISMNIST) for p in posteriors] if LeNet 
                        else [create_model(p) for p in posteriors])

        
        def average_preds(x):
            preds = np.mean([model.predict(x) for model in models])
            return tf.convert_to_tensor(preds)

        self.model = keras.core.layers.Lambda(average_preds)
        self.model_list = models
    
    def predict(self, data):
        return self.model(data)


def create_model(weights):
    model = Sequential()
    num_layers = len(weights)
    layers = []
    
    for i, (layer, data) in enumerate(weights.items()):
        if i == 0:
            input_dim = data.shape[0]
            layers.append(Dense(data.shape[1], input_dim=input_dim))
            layers.append(Activation('tanh'))
        else:
            model.add(Dense(data.shape[1]))
            if i != num_layers - 1:
                layers.append(Activation('tanh'))

    for layer in layers:
        model.add(layer)
    
    model.set_weights(weights.values())
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
    with open(path, 'rb') as f:
        trace = pickle.load(f)
        for x in trace[0]:
            for i in x.values():
                print(i.shape)

# if __name__ == '__main__':
#   with open('example_trace.pkl', 'rb') as f:
#       trace = pickle.load(f)
#       models = theano_to_keras(trace)
#       print(models)