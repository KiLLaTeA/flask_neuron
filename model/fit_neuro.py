import numpy as np
import pandas as pd

from sklearn import preprocessing
from neuron import SingleNeuron

data_fr = pd.read_csv('seeds3.csv')
data_fr = data_fr.sample(frac=1)

dsx = data_fr[['perimeter', 'lengthOfKernel', 'widthOfKernel']]
dsy = data_fr[['seedType']]
data_x = np.array(dsx)

scaler = preprocessing.MinMaxScaler()
dsy = scaler.fit_transform(dsy)
data_y = dsy.reshape(1, -1)[0]

# print(data_x)
# print(data_y)
#
#
# Инициализация и обучение нейрона
neuron = SingleNeuron(input_size=3) # было 2
neuron.train(data_x, data_y, epochs=600, learning_rate=0.04)

# Сохранение весов в файл
neuron.save_weights('neuron_weights.txt')