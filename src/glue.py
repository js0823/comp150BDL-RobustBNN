import pickle
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras


def theano_to_keras(posterior_samples):
	return [create_keras(s) for s in posterior_samples]

def create_keras(weights):
	model = Sequential()
	num_layers = len(weights)
	for i, (layer, data) in enumerate(weights.items()):
		if i == 0:
			input_dim = data.shape[0]
			model.add(Dense(data.shape[1], input_dim=input_dim))
			model.add(Activation('tanh'))
		else:
			model.add(Dense(data.shape[1]))
			if i == num_layers - 1:
				model.add(Activation('softmax'))
			else:
				model.add(Activation('tanh'))
	return model


if __name__ == '__main__':
	with open('example_trace.pkl', 'rb') as f:
		trace = pickle.load(f)
		models = theano_to_keras(trace)
		print(models)