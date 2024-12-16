import numpy as np
import pandas as pd

from sklearn import preprocessing
from neuron import SingleNeuron

data_fr = pd.read_csv('../seeds_Classification.csv')
data_fr = data_fr.sample(frac=1)

dsx = data_fr[['perimeter', 'lengthOfKernel', 'widthOfKernel']]
dsy = data_fr[['seedType']]
data_x = np.array(dsx)

scaler = preprocessing.MinMaxScaler()
dsy = scaler.fit_transform(dsy)
data_y = dsy.reshape(1, -1)[0]

neuron = SingleNeuron(input_size=3)
neuron.train(data_x, data_y)

neuron.save_weights('neuron_weights.txt')