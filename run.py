import time
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

import mlpnas as nas
import utils as utils


utils.clean_log()

start = time.time()

## data preprocessing
data = pd.read_csv('datasets/22.csv')
y = data['label'].values
data.drop('label', axis=1, inplace=True)
x = data.values
x, y = utils.unison_shuffled_copies(x,y)

target_classes = len(np.unique(y))

## intialising the NAS object
mlpnas = nas.NAS(x, y, target_classes)

mlpnas.max_len = 3
mlpnas.cntrl_epochs = 10
mlpnas.mc_samples = 5
mlpnas.hybrid_model_epochs = 10
mlpnas.nn_epochs = 1
mlpnas.nb_final_archs = 5
mlpnas.final_nn_train_epochs = 20
mlpnas.alpha = 0.1
mlpnas.lstm_dim = 100
mlpnas.controller_attention = True
mlpnas.controller_optim = 'sgd'
mlpnas.controller_lr = 0.01
mlpnas.controller_decay = 0.0
mlpnas.controller_momentum = 0.9
mlpnas.nn_optim = 'Adam'
mlpnas.nn_lr = 0.001
mlpnas.nn_decay = 0.0
mlpnas.dropout = 0.2

sorted_archs = mlpnas.architecture_search()
nastime = time.time()

seqsinorder = [item[0] for item in sorted_archs]
valaccsinorder = [item[1] for item in sorted_archs]
predvalaccsinorder = [item[2] for item in sorted_archs]

## best architectures analysed in 4 settings: 
## with/without pre-trained shared weights and/or early stopping
best_archs_valacc, best = mlpnas.train_best_architectures(seqsinorder,
						  use_shared_weights=True,
						  earlyStopping=True)
sw_estime = time.time()

best_archs_valacc, best = mlpnas.train_best_architectures(seqsinorder,
						  use_shared_weights=True,
						  earlyStopping=False)
swtime = time.time()

best_archs_valacc, best = mlpnas.train_best_architectures(seqsinorder,
						  use_shared_weights=False,
						  earlyStopping=True)
estime = time.time()
best_archs_valacc, best = mlpnas.train_best_architectures(seqsinorder,
						  use_shared_weights=False,
						  earlyStopping=False)
end = time.time()

utils.logevent()

print("time spent in seconds:")
print("NAS:", nastime - start)
print("sw_es mode training:", sw_estime - nastime)
print("sw mode training:", swtime - sw_estime)
print("es mode training:", estime - swtime)
print("full mode training:", end - estime)
print("total time:", end - start)

event_id = utils.get_latest_event_id()
print("event id:", event_id)
