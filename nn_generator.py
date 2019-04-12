import os
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from keras import optimizers
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Flatten

import utils as utils


################## architecture sampling #############################################

class NeuralNetwork:

	def __init__(self,target_classes):
		self.target_classes = target_classes
		self.optimizer = 'Adam'
		self.lr = 0.001
		self.decay = 0.0
		self.momentum = 0.0
		self.loss_func = 'categorical_crossentropy'
		self.metrics = ['accuracy']
		self.weights_file = 'logdir/shared_weights{}.pkl'.format(datetime.now().strftime("%H%M"))
		self.shared_weights = pd.DataFrame({'bigram_id': [], 'weights': []})

		## initialise weights file with an empty dictionary
		if not os.path.exists(self.weights_file):
			print("initalising shared weights dictionary...")
			self.shared_weights.to_pickle(self.weights_file)

	def update_weights(self, model):
		## get layer configs, turn them into the tuple format (nb of nodes, activation)
		layer_configs = ['input']
		for layer in model.layers:
			if 'flatten' in layer.name:
				layer_configs.append(('flatten'))
			elif 'dropout' not in layer.name:
				layer_configs.append((layer.get_config()['units'],layer.get_config()['activation']))
		## generate bigram tuples of layers as found in the model
		config_ids = []
		for i in range(1,len(layer_configs)):
			config_ids.append((layer_configs[i-1], layer_configs[i]))
		## update dictionary using said bigram tuples as keys
		j=0
		for i, layer in enumerate(model.layers):
			if 'dropout' not in layer.name:
				warnings.simplefilter(action='ignore', category=FutureWarning)
				bid = self.shared_weights['bigram_id'].values
				srch_ind = []
				for i in range(len(bid)):
					if config_ids[j] == bid[i]:
						srch_ind.append(i)
				if len(srch_ind)==0:
					self.shared_weights = self.shared_weights.append({'bigram_id': config_ids[j],
											'weights': layer.get_weights()},
											ignore_index=True)
				else:
					self.shared_weights.at[srch_ind[0], 'weights'] = layer.get_weights()
				j+=1
		## pickle the dictionary
		self.shared_weights.to_pickle(self.weights_file)

	def set_model_weights(self, model):
		## get layer configs, turn them into the tuple format (nb of nodes, activation)
		layer_configs = ['input']
		for layer in model.layers:
			if 'flatten' in layer.name:
				layer_configs.append(('flatten'))
			elif 'dropout' not in layer.name:
				layer_configs.append((layer.get_config()['units'],layer.get_config()['activation']))
		## generate bigram tuples of layers as found in the model
		config_ids = []
		for i in range(1,len(layer_configs)):
			config_ids.append((layer_configs[i-1], layer_configs[i]))
		## set weights using the bigrams and the weights dict
		j=0
		for i, layer in enumerate(model.layers):
			if 'dropout' not in layer.name:
				warnings.simplefilter(action='ignore', category=FutureWarning)
				bid = self.shared_weights['bigram_id'].values
				srch_ind = []
				for i in range(len(bid)):
					if config_ids[j] == bid[i]:
						srch_ind.append(i)
				if len(srch_ind)>0:
					print("transferring weights for layer:", config_ids[j])
					layer.set_weights(self.shared_weights['weights'].values[srch_ind[0]])
				j+=1

	def create_model(self, pred_sequence, inp_shape):
		## change sequence to its decoded value
		pred_sequence = utils.decode_sequence(utils.vocab_dict(self.target_classes), pred_sequence)
		## set optimizer parameters
		if self.optimizer == 'sgd':
			optim = optimizers.SGD(lr=self.lr,decay=self.decay,momentum=self.momentum)
		else:
			optim = getattr(optimizers, self.optimizer)(lr=self.lr,decay=self.decay)
		## generate a sequential architecture for the sequence
		## add flatten if data is 3d or more
		if len(inp_shape) > 1:
			model = Sequential()
			model.add(Flatten(name = 'flatten', input_shape=inp_shape))
			for i in range(len(pred_sequence)):
				if pred_sequence[i] is 'dropout':
					model.add(Dropout(0.2))
				else:
					model.add(Dense(units = pred_sequence[i][0], activation = pred_sequence[i][1]))
			model.compile(loss = self.loss_func, optimizer = optim, metrics = self.metrics)
			return model
		else:
			model = Sequential()
			for i in range(len(pred_sequence)):
				if i == 0:
					model.add(Dense(units = pred_sequence[i][0], activation = pred_sequence[i][1], input_shape=inp_shape))
				elif pred_sequence[i] is 'dropout':
					model.add(Dropout(0.2))
				else:
					model.add(Dense(units = pred_sequence[i][0], activation = pred_sequence[i][1]))
			model.compile(loss = self.loss_func, optimizer = optim, metrics = self.metrics)
			return model

	def train_model(self, model, x_data, y_data, nb_epochs, validation_split=0.1, callbacks=None, update_shared_weights=True):
		if self.target_classes > 2:
			y_data = to_categorical(y_data)
		self.set_model_weights(model)
		history = model.fit(x_data, y_data, epochs=nb_epochs, validation_split=validation_split, callbacks=callbacks)
		if update_shared_weights:
			self.update_weights(model)
		return history
