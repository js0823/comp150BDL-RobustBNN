import pymc3 as pm
import theano
floatX = theano.config.floatX
from scipy.stats import mode

def train_model(inference_alg, model, num_posterior, X_train, Y_train, minibatch_x=None, minibatch_y=None):
	#inference_alg.fit(n, method, data)
	#return posterior_samples
	
	nn_input = theano.shared(X_train.astype(floatX))
	nn_output = theano.shared(Y_train.astype(floatX))

	if inference_alg is 'advi':
		minibatch_x = pm.Minibatch(X_train.astype(floatX), batch_size=500)
		minibatch_y = pm.Minibatch(Y_train.astype(floatX), batch_size=500)
		with model:
			inference = pm.ADVI()
			approx = pm.fit(n=50000, method=inference, more_replacements={nn_input:minibatch_x, nn_output:minibatch_y})
			trace = approx.sample(draws=num_posterior)
	elif inference_alg is 'nuts':
		with model:
			sample_kwargs = {'cores': 1, 'init': 'auto', 'draws': num_posterior}
			trace = pm.sample(**sample_kwargs)
	
	ppc_train = pm.sample_ppc(trace, samples=100)
	pred_train = mode(ppc_train['out'], axis=0).mode[0, :]
	
	return pred_train, trace