import os
import shutil
import pickle
import numpy as np
from datetime import datetime
import keras.backend as K
from keras import optimizers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences

import generate_nn as gn
import controller as lstm
import utils as utils


################### neural architecture search ####################

class NAS:

	def __init__(self, x, y, target_classes, mdfile):

		self.x_data = x
		self.y_data = y
		self.target_classes = target_classes
		self.max_len = 3
		self.cntrl_epochs = 13
		self.mc_samples = 10
		self.hybrid_model_epochs = 10
		self.nn_epochs = 1
		self.nb_final_archs = 10
		self.final_nn_train_epochs = 100
		self.alpha1 = 5
		self.mdfile = mdfile
		self.pre_train_epochs = 1000
		self.data = []
		self.vocab = utils.vocab_dict(self.target_classes)
		self.nb_classes = len(self.vocab.keys())+1

	def pre_train_loss(self, target, output):
		with open(self.mdfile, 'rb') as f:
			mdata = pickle.load(f)
		self.reward = np.array([item[1] for item in mdata]).reshape(len(mdata),1)
		loss =  - K.sum(target * K.log(output/K.sum(output)), axis=-1)
		loss -= (self.reward - np.median(self.reward))*self.alpha1
		return loss

	def custom_loss(self, target, output):
		self.reward = np.array([item[1] for item in self.data[-self.mc_samples:]]).reshape(self.mc_samples,1)
		loss =  - K.sum(target * K.log(output/K.sum(output)), axis=-1)
		loss -= (self.reward - np.median(self.reward))*self.alpha1
	 	## add predictor accuracy term
		# self.alpha2 = 3
		# self.pred_acc = np.array([item[2] for item in self.data]).reshape(len(self.data),1)
		# loss += (self.reward - self.pred_acc)*self.alpha2
		return loss

	## best archs is list of sequences sorted by their validation accuracy in descending order
	def train_best_architectures(self, best_archs, use_shared_weights=False, earlyStopping=True):
		if use_shared_weights and earlyStopping:
			mode = 'sw_eS'
		elif use_shared_weights:
			mode = 'sw'
		elif earlyStopping:
			mode = 'eS'
		else:
			mode = 'full'
		val_accs = []
		max_val_acc = 0.
		for seq in best_archs[:self.nb_final_archs]:
			if self.target_classes == 2:
				self.nn.loss_func = 'binary_crossentropy'
			## train every model
			print("architecture:", utils.decode_sequence(self.vocab,seq))
			model = self.nn.create_model(seq, np.shape(self.x_data[0]))
			## use early stopping
			if earlyStopping:
				callbacks = [EarlyStopping(monitor='val_acc', patience=0)]
			else:
				callbacks=None
			if use_shared_weights:
				## use pre-trained shared weights without updating them
				history = self.nn.train_model(model,self.x_data,self.y_data,
							  self.final_nn_train_epochs,
							  validation_split=0.1,
							  update_shared_weights=False,
							  callbacks=callbacks)
			else:
				history = model.fit(self.x_data,self.y_data,
						epochs=self.final_nn_train_epochs,
						validation_split=0.1,
						callbacks=callbacks)
			val_accs.append(np.ma.average(history.history['val_acc'],
			weights=np.arange(1,len(history.history['val_acc'])+1),axis=-1))
			## store model, model_weights if mean weighted rolling
			## validation accuracy better than previous models
			if val_accs[-1] > max_val_acc:
				best_arch_vals = {}
				best_arch_vals.update({tuple(seq): model.get_weights()})
				max_val_acc = val_accs[-1]
		## return validation accuracy of all trained architectures
		## return best architecture, it's weights
		best_archs_dict = {}
		for i in range(self.nb_final_archs):
			 best_archs_dict.update({tuple(best_archs[i]) : val_accs[i]})
		top_arch = utils.decode_sequence(self.vocab,
				   list(list(best_arch_vals.keys())[0]))
		print("top {} architectures:".format(self.nb_final_archs),
		 	  best_archs[:self.nb_final_archs])
		print("corresponding validation accuracies:", val_accs)
		print("best architecture:", top_arch)
		print("it's validation accuracy:", max_val_acc)
		print("saving best weights...")
		best_weights_file = 'logdir/best_arch_weights{}{}.pkl'.format(mode,datetime.now().strftime("%H%M"))
		with open(best_weights_file, 'wb') as file:
			pickle.dump(best_arch_vals, file)
		print("saving top architectures and their validation accuracies...")
		best_archs_file = 'logdir/top{}archs{}{}.pkl'.format(self.nb_final_archs,mode,
								 datetime.now().strftime("%H%M"))
		with open(best_archs_file, 'wb') as file:
			pickle.dump(best_archs_dict, file)
		return val_accs, top_arch

	def pre_train_controller(self):
		print("pre-training...")
		with open(self.mdfile, 'rb') as file:
			md = pickle.load(file)
		seqs = [item[0] for item in md]
		seqs = [utils.encode_sequence(self.vocab, x) for x in seqs]
		cntrl_sequences = pad_sequences(seqs, maxlen = self.max_len, padding='post')
		xc = cntrl_sequences[:,:-1].reshape(len(cntrl_sequences),1,self.max_len-1)
		yc = to_categorical(cntrl_sequences[:,-1],self.nb_classes)
		val_acc_target = [item[1] for item in md]
		self.cntrl.train_hybrid_model(xc,yc,
					 val_acc_target,
					 self.pre_train_loss,
					 len(md),
					 self.pre_train_epochs)

	def architecture_search(self, nn_optimizer='Adam'):
		## initialise network modelling and controller instances
		self.nn = gn.NeuralNetwork(self.target_classes, 
                	                   optimizer=nn_optimizer)
		self.cntrl = lstm.LSTMController(self.max_len,
						self.nb_classes,
						self.target_classes,
						(1,self.max_len-1),
						len(self.data))
		## pretraining using metadata
		self.pre_train_controller()
		## start architecture search
		for n in range(self.cntrl_epochs):
			self.pre_training = False
			print("Controller epoch:", n+1)
			self.curr_epoch = n
			## generate sequences using random probabilistic sampling
			sequences = self.cntrl.sample_arch_sequences(self.mc_samples)
			## train predictor and get predicted accuracies for new sequences
			pred_val_acc = self.cntrl.get_predicted_accuracies_hybrid_model(sequences)
			## for each randomly generated sample
			for i in range(len(sequences)):
				print("probabilistic sampling. model no:", i+1)
				print(utils.decode_sequence(self.vocab,sequences[i]))
				## create model. train model
				print("training model...")
				if self.target_classes == 2:
					self.nn.loss_func = 'binary_crossentropy'
				model = self.nn.create_model(sequences[i], np.shape(self.x_data[0]))
				print("predicted validation accuracy:", pred_val_acc[i])
				history = self.nn.train_model(model,self.x_data,self.y_data,self.nn_epochs)
				## condition to avoid error for nn_epochs = 1
				if len(history.history['val_acc']) == 1:
					self.data.append([sequences[i],
							 history.history['val_acc'][0],
							 pred_val_acc[i]])
				else:
					self.data.append([sequences[i],
						 	  np.ma.average(history.history['val_acc'],
							  weights=np.arange(1,len(history.history['val_acc'])+1),
							  axis=-1),pred_val_acc[i]])
			cntrl_sequences = pad_sequences(sequences, maxlen = self.max_len, padding='post')
			xc = cntrl_sequences[:,:-1].reshape(len(cntrl_sequences),1,self.max_len-1)
			yc = to_categorical(cntrl_sequences[:,-1],self.nb_classes)
			## sequence, validation accuracy data sorted by validation accuracy
			print("[sequence, val acc, predicted val acc]")
			for data in self.data:
				print(data)
			## train the controller
			val_acc_target = [item[1] for item in self.data]
			self.cntrl.train_hybrid_model(xc,yc,
						  val_acc_target[-self.mc_samples:],
						  self.custom_loss,
						  len(self.data),
						  self.hybrid_model_epochs)
		val_accs = [item[1] for item in self.data]
		sorted_idx = np.argsort(val_accs)[::-1]
		self.data = [self.data[x] for x in sorted_idx]
		print("saving tested architectures, their validation accuracy and predicted accuracy...")
		with open('logdir/tested_archs_data{}.pkl'.format(datetime.now().strftime("%H%M")), 'wb') as file:
			pickle.dump(self.data, file)
		print("saving encoding-decoding dictionary...")
		with open('logdir/encode_decode_dict.pkl', 'wb') as file:
			pickle.dump(self.vocab, file)
		return self.data
