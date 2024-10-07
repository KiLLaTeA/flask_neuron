import numpy as np
from neuron import SingleNeuron

new_neuron = SingleNeuron(input_size=3)
new_neuron.load_weights('neuron_weights.txt')

# 1 класс = 0; 2 класс = 1;
test_data = np.array([16.99, 6.581, 3.785])  #2 тип
predictions = new_neuron.forward(test_data)
print("Предсказанное значение:", predictions)

test_data = np.array([15.25, 5.718, 3.525])  #2 тип
predictions = new_neuron.forward(test_data)
print("Предсказанное значение:", predictions)

test_data = np.array([13.85, 5.348, 3.156])  #1 тип
predictions = new_neuron.forward(test_data)
print("Предсказанное значение:", predictions)

test_data = np.array([14.56, 5.57, 3.377])  #1 тип
predictions = new_neuron.forward(test_data)
print("Предсказанное значение:", predictions)
