import os
import numpy as np
from datetime import datetime
from keras import optimizers
from keras.layers import Dense, LSTM
from keras.models import Model, Sequential
from keras.engine.input_layer import Input
from keras.preprocessing.sequence import pad_sequences

import utils as utils

from keras_attention.models.custom_recurrents import AttentionDecoder


################# lstm controller ###########################

class LSTMController:

	def __init__(self, max_len, nb_classes, target_classes, inp_shape, batch_shape):
		self.max_len = max_len
		self.nb_classes = nb_classes
		self.target_classes = target_classes
		self.inp_shape = inp_shape
		self.batch_shape = batch_shape
		self.vocab = utils.vocab_dict(target_classes)
		self.lstm_dim = 100
		self.use_attention = True
		self.optimizer = 'sgd'
		self.lr = 0.01
		self.decay = 0.0
		self.momentum = 0.9
		self.hybrid_weights = 'logdir/hybrid_weights{}.h5'.format(datetime.now().strftime("%H%M"))
		self.model = self.hybrid_cntrl_model(self.inp_shape, self.batch_shape)
		self.seq_data = []

	def sample_arch_sequences(self, rand_samples):
		final_layer_id = len(self.vocab)
		samples = []
		print("generating architecture samples...")
		## loop for generating rand_samples number of sample architectures
		while len(samples) < rand_samples:
			seed = []
			## generating architecture sequence
			while len(seed) < self.max_len:
				sequence = pad_sequences([seed], maxlen = self.max_len-1, padding='post')
				sequence = sequence.reshape(1,1,self.max_len-1)
				# probab = self.model.predict(sequence)
				(probab, _) = self.model.predict(sequence)
				## remove zero value
				probab = probab[0]
				probab = np.delete(probab, 0)
				if len(seed) == 0:
					probab[-1] = 0 				## first layer can't be the last one
					probab[-2] = 0				## first layer can't be dropout
				## smoothing using a gaussian kernel
				probab = utils.gaussian_smoothing(probab, np.array(list(self.vocab.keys())), 0.02)
				probab = probab/np.sum(probab)
				## given the previous elements in a sequence, generating the following ones
				next = np.random.choice(list(self.vocab.keys()),size=1, p = probab)[0]
				## considering the constraints
				if not next == 0:
					seed.append(next)
				if next == final_layer_id:
					break
				if len(seed) == self.max_len - 1:
					seed.append(final_layer_id)
					break
			if seed not in self.seq_data:
				samples.append(seed)
				self.seq_data.append(seed)
		return samples

	def hybrid_cntrl_model(self, inp_shape, batch_size):
		main_input = Input(shape=inp_shape, batch_shape=batch_size, name='main_input')
		x = LSTM(self.lstm_dim, return_sequences=True)(main_input)
		predictor_output = Dense(1, activation='sigmoid', name='predictor_output')(x)
		if self.use_attention:
			main_output = AttentionDecoder(self.lstm_dim, self.nb_classes, name='main_output')(x)
		else:
			main_output = Dense(self.nb_classes, activation = 'softmax', name='main_output')(x)
		model = Model(inputs=[main_input], outputs=[main_output, predictor_output])
		return model

	def train_hybrid_model(self, x_data, y_data, pred_target, loss_func, batch_size, nb_epochs):
		if self.optimizer == 'sgd':
			optim = optimizers.SGD(lr=self.lr,decay=self.decay,momentum=self.momentum)
		else:
			optim = getattr(optimizer, self.optimizer)(lr=self.lr,decay=self.decay)
		self.model.compile(optimizer=optim,
	              	  loss={'main_output': loss_func, 'predictor_output': 'mse'},
					  loss_weights = {'main_output': 1, 'predictor_output': 1})
		if os.path.exists(self.hybrid_weights):
			self.model.load_weights(self.hybrid_weights)
		print("training controller...")
		self.model.fit({'main_input': x_data},
	              {'main_output': y_data.reshape(len(y_data),1,self.nb_classes),
			       'predictor_output': np.array(pred_target).reshape(len(pred_target),1,1)},
	          	   epochs=nb_epochs, batch_size=batch_size)
		self.model.save_weights(self.hybrid_weights)

	def get_predicted_accuracies_hybrid_model(self, seqs):
		pred_accuracies = []
		for seq in seqs:
			cntrl_sequences = pad_sequences([seq], maxlen = self.max_len, padding='post')
			xc = cntrl_sequences[:,:-1].reshape(len(cntrl_sequences),1,self.max_len-1)
			(_, pred_accuracy) = [x[0][0] for x in self.model.predict(xc)]
			pred_accuracies.append(pred_accuracy[0])
		return pred_accuracies
