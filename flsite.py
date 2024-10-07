import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, url_for, request, jsonify
from model.neuron import SingleNeuron

app = Flask(__name__)

# menu = [{"name": "Лаба 1", "url": "p_knn"},
#         {"name": "Лаба 2", "url": "p_lab2"},
#         {"name": "Лаба 3", "url": "p_lab3"},
#         {"name": "Лаба 4", "url": "p_lab4"},
#         {"name": "Нейронная сеть", "url": "neural_network"}]

menu = [{"name": "Нейронная сеть", "url": "neural_network"}]

# loaded_model_knn = pickle.load(open('model/Iris_pickle_file', 'rb'))
# Загрузка весов из файла
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
        print("Предсказанное значение: %.3f" % predictions)
        # print("Предсказанные значения:", predictions, *np.where(predictions >= 0.5, 'Помидор', 'Огурец'))
        return render_template('neural_network.html', title="Первый нейрон", menu=menu,
                               class_model="Это: " + predictions)





# @app.route('/api', methods=['get'])
# def get_sort():
#     X_new = np.array([[float(request.args.get('sepal_length')),
#                        float(request.args.get('sepal_width')),
#                        float(request.args.get('petal_length')),
#                        float(request.args.get('petal_width'))]])
#     pred = loaded_model_knn.predict(X_new)
#
#     return щ(sort=pred[0])
#
# @app.route('/api_v2', methods=['get'])
# def get_sort_v2():
#     request_data = request.get_json()
#     X_new = np.array([[float(request_data['sepal_length']),
#                        float(request_data['sepal_width']),
#                        float(request_data['petal_length']),
#                        float(request_data['petal_width'])]])
#     pred = loaded_model_knn.predict(X_new)
#
#     return jsonify(sort=pred[0])

if __name__ == "__main__":
    app.run(debug=True)
