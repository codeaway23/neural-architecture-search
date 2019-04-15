# Neural Architecture Search for Multi Layer Perceptrons 

Insights drawn from the following papers:
[ENAS paper](https://arxiv.org/abs/1802.03268), 
[SeqGAN paper](https://arxiv.org/abs/1609.05473), 
[NAO paper](https://arxiv.org/abs/1808.07233). An evaluation of different NAS algorithms' search phases can be found [here](https://arxiv.org/abs/1902.08142).


## Features

The code incorporates an LSTM controller to generate sequences that represent neural network architectures, and an accuracy predictor for the generated architectures. these architectures are built into keras models, trained for certain number of epochs, evaluated, the validation accuracy being used to update the controller for better architecture search. 

1. LSTM controller with REINFORCE policy gradient
2. Accuracy predictor that shares weights with the above mentioned LSTM controller.
3. Weight sharing in all the architectures generated during the search phase. 
4. Meta-learning using dataset properties extraction, comparison and pre-training of the LSTM model. 
5. Attention mechanism used available in this [repository](https://github.com/datalogue/keras-attention)


## Usage

To run the architecture search:
1. Add the dataset in the datasets directory.
2. add dataset path in run.py
3. run the following command from the main directory.

```bash
python3 run.py
```

To generate metadata:
1. Add dataset in the datasets directory.
2. Run the following command from the metadata directory:
```bash
python3 generate_metadata.py
```

To change dataset preprocessing, the NAS/controller/mlp training parameters, open the run.py file and edit. 

