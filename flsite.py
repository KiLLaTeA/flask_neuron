import pickle
import pandas as pd

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, url_for, request, jsonify
from model.neuron import SingleNeuron

app = Flask(__name__)

menu = [
            {"name": "(Lab_14) Нейронная сеть", "url": "neural_network"},
            {"name": "(Lab_16) API классификации", "url": "api_class"},
            {"name": "(Lab_16) API регрессии", "url": "api_reg"}
       ]

new_neuron = SingleNeuron(input_size=3)
new_neuron.load_weights('model/neuron_weights.txt')
model_reg = tf.keras.models.load_model('model/regression_model.h5')
model_class = tf.keras.models.load_model('model/classification_model.h5')

@app.route("/")
def index():
    return render_template('index.html', title="Костин М.М. - ИСТ-301", menu=menu)


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


@app.route('/api_reg', methods=['get'])
def predict_regression():
    # http://localhost:5000/api_reg?LengthOfKernel=6.303
    input_data = np.array([float(request.args.get('LengthOfKernel'))])
    print(input_data)

    predictions = model_reg.predict(input_data)
    print(predictions)

    return jsonify(WidthOfKernel=str(predictions[0][0]))


@app.route('/api_class', methods=['get'])
def predict_classification():
    # http://localhost:5000/api_class?Perimeter=16.72&LengthOfKernel=6.303&WidthOfKernel=3.791
    # http://localhost:5000/api_class?Perimeter=13.85&LengthOfKernel=5.348&WidthOfKernel=3.156

    input_data = np.array([[float(request.args.get('Perimeter')),
                           float(request.args.get('LengthOfKernel')),
                           float(request.args.get('WidthOfKernel'))]])

    print(input_data)

    predictions = model_class.predict(input_data)
    print(predictions)
    result = "First sort" if predictions[0][0] <= 0.5 else "Second sort"
    print(result)

    app.config['JSON_AS_ASCII'] = False
    return jsonify(ov = str(result))


if __name__ == "__main__":
    app.run(debug=True)
