import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, url_for, request, jsonify
from model.neuron import SingleNeuron

app = Flask(__name__)

menu = [{"name": "Нейронная сеть", "url": "neural_network"}]

new_neuron = SingleNeuron(input_size=3)
new_neuron.load_weights('model/neuron_weights.txt')

@app.route("/")
def index():
    return render_template('index.html', title="Костин М.М.ИСТ-301", menu=menu)

@app.route("/neural_network", methods=['POST', 'GET'])
def neural_network():
    if request.method == 'GET':
        return render_template('neural_network.html', title="Нейронная сеть", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           ])
        predictions = new_neuron.forward(X_new)
        Y_res = "Первый сорт" if predictions <= 0.5 else "Второй сорт"
        return render_template('neural_network.html', title="Первый нейрон", menu=menu,
                               class_model="Результат: " + str(predictions) + "\n" + "Это " + str(Y_res))

if __name__ == "__main__":
    app.run(debug=True)
