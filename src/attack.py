import numpy as np
import tensorflow as tf

from cleverhans.loss import CrossEntropy
from cleverhans.dataset import MNIST
from cleverhans.dataset import CIFAR10
from cleverhans.train import train
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans_tutorials.tutorial_models import ModelBasicCNN


def init_hyperparams(attack): #, x_train, y_train):
	# hyperparams = dict()
	#attack specific params
	if attack == "BasicIterativeMethod":
		hyperparams = { #['attack_hp'] = {
			'eps': 0.2, #maximum distortion of adversarial examples compared to orig, default = 0.3
			'eps_iter': 0.05, #step size for each attack iteration, default = 0.1
			'clip_min': 0., #min input component val, default = 0
			'clip_max': 1.,  #max input component val, default = 1
			#y #tensor with model labels
			#y_target #tensor with labels to target
			#ord #order of the norm (1,2,np.inf)
			'nb_iter': 20 #number of attack iterations, default = 10
			#x #model's symbolic inputs
		}
	elif attack == "CarliniWagnerL2":
		hyperparams = { #['attack_hp'] = {
			#'x': 0.18,
			#'y': ,
			#'y_target': ,
			'confidence': 0,
			'batch_size': 1,
			'learning_rate': 0.005,
			'binary_search_steps': 5,
			'max_iterations': 1000,
			'abort_early': True,
			'initial_const': 0.01,
			'clip_min': 0,
			'clip_max': 1
		}
	else:
		print("Unsupported attack {}".format(attack))
		exit()
	# #data specific params
	# img_rows, img_cols, nchannels = x_train.shape[1:4]
	# nb_classes = y_train.shape[1]
	# hyperparams['data_hp'] = {
	# 	'img_rows' = img_rows,
	# 	'img_cols' = img_cols,
	# 	'nchannels' = nchannels,
	# 	'nb_classes' = nb_classes
	# }
	return hyperparams

def gray_box(model, attack_string, data, sess):
	hp = init_hyperparams(attack_string)
	# sess = tf.Session()
	attack = generate_attack(attack_string, model, sess)
	adv_x = attack.generate_np(data, **hp)
	return adv_x

def white_box(models, attack, data):
	pass
	#clean = load_dataset(data)
	#hp = init_hyperparams(attack, data)
	#model = avg(models)
	#adv_x = attack_multiple(models, clean, **hp)
	#return adv_x

def eval_model(models, x, adv_x, adv_y, x_test, y_test, sess=None):
	pass
	a = len(adv_x)
	x_combined = np.concatenate((adv_x, x_test))
	#What to write here depends on what is returned from glue code
	preds_probs = np.asarray([model.get_probs(x).eval(feed_dict={x: x_combined}, session=sess) for model in models])
	adv_preds_probs = np.mean(preds_probs[:,:a], axis=0)
	adv_preds = np.argmax(adv_preds_probs, axis=1)
	true_labels = np.argmax(adv_y, axis=1)
	accuracy = np.sum(adv_preds==true_labels) / float(a)
	print("Predicted labels on adversarial examples:")
	print(adv_preds)
	print("True labels of adversarial examples:")
	print(true_labels)
	print("Accuracy: {}".format(accuracy))

	#MI
	#TODO: Ugly, clean up
	#for every class
	class_sums = np.zeros(len(preds_probs[0]))
	for c in range(10):
		p_hat = np.mean(preds_probs[:,:,c], axis = 0).flatten()
		# print("p_hat")
		# print(p_hat)
		model_sums = []
		#for every example get p_ij sum term
		for i in range(len(preds_probs[0])):
			p_ijs = preds_probs[:,i,c]
			# print("p_ijs")
			# print(p_ijs)
			model_sums.append(np.sum(np.square(p_ijs))/float(len(preds_probs)))
		# print("model_sums")
		# print(model_sums)
		# print(np.square(p_hat))
		class_sums += (np.asarray(model_sums) - np.square(p_hat))
	uncertainties = class_sums / 10.0
	# print(uncertainties)
	return accuracy, uncertainties

def load_dataset(data_string, train_start=0, train_end=60000, test_start=0, test_end=10000):
	if data_string == "MNIST":
		data = MNIST(train_start=train_start, train_end=train_end,
				  test_start=test_start, test_end=test_end)
	elif data_string == "CIFAR10":
		data = CIFAR10(train_start=train_start, train_end=train_end,
				  test_start=test_start, test_end=test_end)
	else:
		print("Unsupported dataset {}".format(data))
	x_train, y_train = data.get_set('train')
	x_test, y_test = data.get_set('test')
	return x_train, y_train, x_test, y_test

def generate_attack(attack_string, model, sess):
	if attack_string == "BasicIterativeMethod":
		attack = BasicIterativeMethod(model, sess)
	elif attack_string == "CarliniWagnerL2":
		attack = CarliniWagnerL2(model, sess)
	else:
		print("Unsupported attack {}".format(attack_string))
	return attack

# def generate_adv_x(attack, hp):
# 	x = tf.placeholder(tf.float32, shape=(None, hp.data_hp.img_rows, hp.data_hp.img_cols,
# 										   hp.data_hp.nchannels))
# 	return attack.generate(x, **hp.attack_hp)

if __name__ == '__main__':
	#TODO: handle CIFAR10 having 3 channels
	x_train, y_train, x_test, y_test = load_dataset("MNIST")

	#TEMP
	#have to figure out what will be spit out from glue code and how to use
	#and setup tf session
	sess=tf.Session()
	train_params = {
	  'nb_epochs': 3,
	  'batch_size': 128,
	  'learning_rate': 0.001
	}
	img_rows, img_cols, nchannels = x_train.shape[1:4]
	nb_classes = y_train.shape[1]

	x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
									nchannels))
	y = tf.placeholder(tf.float32, shape=(None, nb_classes))
	model = ModelBasicCNN('model1', nb_classes, 64)
	loss = CrossEntropy(model, smoothing=0.1)
	train(sess, loss, x_train, y_train, args = train_params,
		  rng=np.random.RandomState([2018,11,18]), var_list=model.get_params())
	#models = glue()...
	adv_x = gray_box(model, "BasicIterativeMethod", x_test[0:10], sess)
	accuracy, uncertainties = eval_model([model, model], x, adv_x, y_test[0:10], x_test[10:20], y_test[10:20], sess=sess)