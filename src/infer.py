import pymc3 as pm
import theano
import pickle
floatX = theano.config.floatX
from scipy.stats import mode

def save_posterior(model, trace, filename):
	with open(filename, 'wb') as buff:
		#pickle.dump({'model': model, 'trace': trace}, buff)
		pickle.dump({'trace': trace}, buff)
	print("Saving model and trace done.")

def load_posterior(filename):
	with open(filename, 'rb') as buff:
		data = pickle.load(buff)
	
	#basic_model, trace = data['model'], data['trace']
	trace = data['trace']
	print("Loading model and trace done.")
	return trace

def train_model(inference_alg, model, num_posterior, nn_input, nn_output, X_train, Y_train, X_test, Y_test):
	#inference_alg.fit(n, method, data)
	#return posterior_samples

	if inference_alg is 'advi':
		minibatch_x = pm.Minibatch(X_train.astype(floatX), batch_size=500)
		minibatch_y = pm.Minibatch(Y_train.astype(floatX), batch_size=500)
		with model:
			inference = pm.ADVI()
			approx = pm.fit(n=50000, method=inference, more_replacements={nn_input:minibatch_x, nn_output:minibatch_y})
			#db = pm.backends.Text('advi-backend')
			trace = approx.sample(draws=num_posterior)

			#ppc_train = pm.sample_ppc(trace, samples=100)
			#pred_train = mode(ppc_train['out'], axis=0).mode[0, :]

		nn_input.set_value(X_test)
		nn_output.set_value(Y_test)

		with model:
			ppc_test = pm.sample_ppc(trace, samples=100)
			pred_test = mode(ppc_test['out'], axis=0).mode[0, :]

	elif inference_alg is 'nuts':
		with model:
			#db = pm.backends.Text('nuts-backend')
			sample_kwargs = {'cores': 1, 'init': 'advi+adapt_diag', 'draws': num_posterior}
			trace = pm.sample(**sample_kwargs)

			#ppc_train = pm.sample_ppc(trace, samples=100)
			#pred_train = mode(ppc_train['out'], axis=0).mode[0, :]

		nn_input.set_value(X_test)
		nn_output.set_value(Y_test)

		with model:
			ppc_test = pm.sample_ppc(trace, samples=100)
			pred_test = mode(ppc_test['out'], axis=0).mode[0, :]
	
	return pred_test, trace