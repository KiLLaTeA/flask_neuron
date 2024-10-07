import numpy as np
from neuron import SingleNeuron

# Загрузка весов из файла и тестирование
new_neuron = SingleNeuron(input_size=3)
new_neuron.load_weights('neuron_weights.txt')

# Пример использования
test_data = np.array([14.49, 5.563, 3.259])
predictions = new_neuron.forward(test_data)
print("Предсказанное значение: %.3f" % predictions)