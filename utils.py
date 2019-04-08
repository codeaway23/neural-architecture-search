import os
import shutil
import pickle
import numpy as np
from datetime import datetime


############## util functions ##############################


## layer encodings
def vocab_dict(target_classes):
	nodes = [8,16,32,64,128,256,512]
	act_funcs = ['sigmoid','tanh','relu','elu']
	layer_params = []
	layer_id = []
	for i in range(len(nodes)):
		for j in range(len(act_funcs)):
			layer_params.append((nodes[i],act_funcs[j]))
			layer_id.append(len(act_funcs)*i+j+1)
	vocab = dict(zip(layer_id, layer_params))
	vocab[len(vocab)+1] = (('dropout'))
	if target_classes == 2:
		vocab[len(vocab)+1] = (target_classes-1,'sigmoid')
	else:
		vocab[len(vocab)+1] = (target_classes,'softmax')
	return vocab

def encode_sequence(vocab, sequence):
	keys = list(vocab.keys())
	values = list(vocab.values())
	encoded_sequence = []
	for value in sequence:
		encoded_sequence.append(keys[values.index(value)])
	return encoded_sequence

def decode_sequence(vocab, sequence):
	keys = list(vocab.keys())
	values = list(vocab.values())
	decoded_sequence = []
	for key in sequence:
		decoded_sequence.append(values[keys.index(key)])
	return decoded_sequence

def gaussian_smoothing(x, y, sigma):
	smoothed_x = np.zeros(x.shape)
	for i in y:
		kernel = np.exp(-(y - i) ** 2 / (2 * sigma ** 2))
		kernel = kernel / sum(kernel)
		smoothed_x[i-1] = sum(x * kernel)
	return smoothed_x

def clean_log():
	filelist = os.listdir('logdir')
	for file in filelist:
		if os.path.isfile('logdir/{}'.format(file)):
			os.remove('logdir/{}'.format(file))

def logevent():
	dest = 'logdir'
	while os.path.exists(dest):
		dest = 'logdir/event{}'.format(np.random.randint(10000))
	os.mkdir(dest)
	filelist = os.listdir('logdir')
	for file in filelist:
		if os.path.isfile('logdir/{}'.format(file)):
			shutil.move('logdir/{}'.format(file),dest)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def get_latest_event_id():
    all_subdirs = ['logdir/'+ d for d in os.listdir('logdir') if os.path.isdir('logdir/'+ d)]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    return int(latest_subdir.replace('logdir/event',''))

