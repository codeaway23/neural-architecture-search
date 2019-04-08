import time
import pandas as pd
from keras.datasets import fashion_mnist, mnist
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

import sample_architecture as sa
import lstm_nn as lstm
import nas as nas
import utils as utils

utils.clean_log()

start = time.time()

# mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x = x_train
y = to_categorical(y_train)

x, y = utils.unison_shuffled_copies(x,y)

target_classes = 10

mlpnas = nas.NAS(x, y, target_classes)

mlpnas.max_len = 5
mlpnas.cntrl_epochs = 20
mlpnas.mc_samples = 15
mlpnas.hybrid_model_epochs = 15
mlpnas.nn_epochs = 1
mlpnas.nb_final_archs = 10
mlpnas.final_nn_train_epochs = 20
mlpnas.alpha1 = 5

# make all controller, nn parameters accessible for modification here.
# lstm, nn optimizers.
# lstm loss weights.

sorted_archs = mlpnas.architecture_search()
nastime = time.time()

seqsinorder = [item[0] for item in sorted_archs]
valaccsinorder = [item[1] for item in sorted_archs]
predvalaccsinorder = [item[2] for item in sorted_archs]

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
