import pickle
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras

class BNN:
    def __init__(self, path=None, num_labels=10):
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = num_labels

        with open(path, 'rb') as f:
        	posteriors = pickle.load(f)
        	models = [create_model(posterior) for posterior in posteriors]
        
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


# if __name__ == '__main__':
# 	with open('example_trace.pkl', 'rb') as f:
# 		trace = pickle.load(f)
# 		models = theano_to_keras(trace)
# 		print(models)