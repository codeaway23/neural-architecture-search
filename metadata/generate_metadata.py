import sys
sys.path.append('..')

import numpy as np

from hyperopt import Trials, STATUS_OK, tpe

from sklearn.preprocessing import LabelEncoder
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice

import os
import pickle

import pandas as pd

import utils as utils


def create_model(x_train, y_train, nb_inputs, nb_classes, arch_list, filename):
	layer = []
	if len(nb_inputs) == 1:
		layer.append(Dense({{choice([8,16,32,64,128,256,512])}}, input_shape=nb_inputs))
	else:
		layer.append(Dense({{choice([8,16,32,64,128,256,512])}}))
	layer.append(Dense({{choice([8,16,32,64,128,256,512])}}))
	act = []
	act.append(Activation({{choice(['relu', 'elu', 'tanh', 'sigmoid'])}}))
	act.append(Activation({{choice(['relu', 'elu', 'tanh', 'sigmoid'])}}))
	dropout = []
	dropout.append({{choice([0,1])}})
	dropout.append({{choice([0,1])}})
	model = Sequential()
	if len(nb_inputs) > 1:
		model.add(Flatten(input_shape=(nb_inputs)))
	model.add(layer[0])
	model.add(act[0])
	hidden_layers = {{choice(range(1,3))}}  ## maximum 3 hidden layers. 
	for i in range(1,hidden_layers):
		model.add(layer[i])
		model.add(act[i])
		if dropout[i]==1:
			model.add(Dropout(0.2))
	if nb_classes == 2:
		model.add(Dense(nb_classes-1))
		model.add(Activation('sigmoid'))
		loss_func = 'binary_crossentropy'
	else:
		model.add(Dense(nb_classes))
		model.add(Activation('softmax'))
		loss_func = 'categorical_crossentropy'
	model.compile(loss=loss_func,
			metrics=['accuracy'],
			optimizer='Adam')
	print(model.summary())
	x, y = utils.unison_shuffled_copies(x_train, y_train)
	result = model.fit(x, y,
			epochs=10,
			validation_split=0.1)
	validation_acc = np.ma.average(result.history['val_acc'], 
			weights=np.arange(1, len(result.history['val_acc'])+1))
	arch = []
	for layer in model.layers:
		if 'dense' in layer.name:
			units = layer.units
			arch.append(units)
		if 'activation' in layer.name:
			act = layer.get_config()['activation']
			arch.append(act)
		if 'dropout' in layer.name:
			arch.append('dropout')
	arch_final = []
	for i,x in enumerate(arch):
		if type(x) == int:
			arch_final.append(tuple((x, arch[i+1])))
		if x == 'dropout':
			arch_final.append('dropout')
	print(arch_final)
	arch_list.append([arch_final, validation_acc])
	file = 'metadatafiles/hyperas_{}_arch_list.pkl'.format(filename[:-4])
	with open(file,'wb') as f:
		pickle.dump(arch_list, f)
	print('validation accuracy:', validation_acc)
	return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

def data():
	datasets = '../datasets'
	filename = 'bank.csv'
	csvfile = datasets+'/{}'.format(filename)
	x = pd.read_csv(csvfile, encoding='iso-8859-1')
	x.fillna(0.0, inplace=True)
	x_train = x.values[:, :-1]
	y_train = x.values[:,-1]
	nb_classes = len(np.unique(y_train))
	if nb_classes > 2:
		y_train = np_utils.to_categorical(y_train)
	nb_inputs = x_train[0].shape
	arch_list = []
	return x_train, y_train, nb_inputs, nb_classes, arch_list, filename

best_run, best_model = optim.minimize(model=create_model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=100,
                                      trials=Trials())


