import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

class SingleNeuron:
    def __init__(self, input_size):
        self.weight = np.random.rand(input_size)
        self.bias = np.random.rand(1)

    def forward(self, X):
        self.input = X
        self.z = np.dot(X, self.weight) + self.bias
        self.output = sigmoid(self.z)
        return self.output

    def backward(self, y, learning_rate=0.001):
        error = self.output - y

        d_output = error * sigmoid_derivative(self.output)
        d_weight = np.dot(self.input.T, d_output)
        d_bias = np.sum(d_output)

        self.weight -= learning_rate * d_weight
        self.bias -= learning_rate * d_bias

        return error

    def train(self, X, y, epochs=20000, learning_rate=0.001):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(y, learning_rate)

    def save_weights(self, filename):
        np.savetxt(filename, np.hstack((self.weight, self.bias)))

    def load_weights(self, filename):
        data = np.loadtxt(filename)

        self.weight = data[:-1]
        self.bias = data[-1]