def init_hyperparams(attack, dataset): #e.g. dataset = "MNIST"
	return hyperparams

def gray_box(model, attack, data):
	pass
	#clean = load_dataset(data)
	#hp = init_hyperparams(attack, data)
	#attack = cleverhans.attack(model, sess)
	#adv_x = attack.generate(clean, **hp)
	#return adv

def white_box(models, attack, data):
	pass
	#clean = load_dataset(data)
	#hp = init_hyperparams(attack, data)
	#model = avg(models)
	#adv_x = attack_multiple(models, clean, **hp)
	#return adv_x

def eval_model(models, adv, data):
	pass
	#preds = [model.predict(adv+data) for model in models]
	#accuracy = np.sum(np.mean(preds[0:a]) == label) / num_samples
	#uncertainty = MI (on adv+data)
	#return accuracy, uncertainty