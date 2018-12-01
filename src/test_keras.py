import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import urllib.request

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils import np_utils
from keras.models import load_model

model = Sequential()

nb_filters = 64
num_labels = 10
layers = [Conv2D(nb_filters, (5, 5), strides=(2, 2), padding="same",
                 input_shape=(28, 28, 1)),
          Activation('relu'),
          Conv2D(nb_filters, (3, 3), strides=(2, 2), padding="valid"),
          Activation('relu'),
          Conv2D(nb_filters, (3, 3), strides=(1, 1), padding="valid"),
          Activation('relu'),
          Flatten(),
          Dense(32),
          Activation('relu'),
          Dropout(.5),
          Dense(num_labels)]

for layer in layers:
    model.add(layer)

keras.backend.set_learning_phase(False)
print("Model type is: ", type(model))
fake_data = np.random.rand(28, 28, 1, 32)
print("Data type is: ", type(fake_data))