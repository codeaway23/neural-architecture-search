# Neural Architecture Search for Multi Layer Perceptrons 

Insights drawn from the following papers:

1. [ENAS](https://arxiv.org/abs/1802.03268)
2. [SeqGAN](https://arxiv.org/abs/1609.05473) 
3. [NAO](https://arxiv.org/abs/1808.07233)
4. An evaluation of different NAS algorithms' search phases can be found [here](https://arxiv.org/abs/1902.08142)


## Features

The code incorporates an LSTM controller to generate sequences that represent neural network architectures, and an accuracy predictor for the generated architectures. these architectures are built into keras models, trained for certain number of epochs, evaluated, the validation accuracy being used to update the controller for better architecture search. 

1. LSTM controller with REINFORCE gradient
2. Accuracy predictor that shares weights with the above mentioned LSTM controller.
3. Weight sharing in all the architectures generated during the search phase.  
4. Attention mechanism used available in this [repository](https://github.com/datalogue/keras-attention)


## Usage

To run the architecture search:
1. Add the dataset in the datasets directory.
2. add dataset path in run.py
3. run the following command from the main directory.

```bash
python3 run.py
```

To change dataset preprocessing, the NAS/controller/mlp training parameters, open the run.py file and edit. defaults mentioned below.

```python

mlpnas = nas.NAS(x, y, target_classes)

mlpnas.max_len = 3
mlpnas.cntrl_epochs = 20
mlpnas.mc_samples = 15
mlpnas.hybrid_model_epochs = 15
mlpnas.nn_epochs = 1
mlpnas.nb_final_archs = 10
mlpnas.final_nn_train_epochs = 20
mlpnas.alpha = 5
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
```

to vary the search space, edit the vocab_dict() function in utils.py file. defaults mentioned below.

```python
nodes = [8,16,32,64,128,256,512]
act_funcs = ['sigmoid','tanh','relu','elu']
```

