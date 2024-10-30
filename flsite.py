import pickle
import pandas as pd
import os
import PIL

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, url_for, request, jsonify, redirect
from model.neuron import SingleNeuron

from keras.src.legacy.preprocessing import image

app = Flask(__name__)

menu = [
            {"name": "(Lab_14) Нейронная сеть", "url": "neural_network"},
            {"name": "(Lab_16) API классификации", "url": "api_class?Perimeter=16.72&LengthOfKernel=6.303&WidthOfKernel=3.791"},
            {"name": "(Lab_16) API регрессии", "url": "api_reg?LengthOfKernel=6.303"},
            {"name": "(Lab_17) Одежда", "url": "mnist_fashion"},
            {"name": "(Lab_18) Одежда", "url": "mnist_CNN"}
       ]

fashion_classes = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress",
    4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker",
    8: "Bag", 9: "Ankle boot"
}

UPLOAD_FOLDER = 'model/clothes'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

new_neuron = SingleNeuron(input_size=3)
new_neuron.load_weights('model/neuron_weights.txt')
model_reg = tf.keras.models.load_model('model/regression_model.h5')
model_class = tf.keras.models.load_model('model/classification_model.h5')
model_fashion = tf.keras.models.load_model('model/fashion.h5')
model_fashion_CNN = tf.keras.models.load_model('model/fashion_CNN.h5')

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


@app.route("/mnist_fashion", methods=['POST', 'GET'])
def upload_clothes():
    if request.method == 'GET':
        return render_template('fashion.html', title="Классификация одежды", menu=menu, fashion_model='')
    if request.method == 'POST':
        if 'image' not in request.files:
            return "Нету пути файла"

        file = request.files['image']

        if file.filename == '':
            return "Не выбран файл"

        if file:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

            img = image.image_utils.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
            x = image.image_utils.img_to_array(img)
            x = x.reshape(1, 784)
            x = 255 - x
            x /= 255

            predictions = model_fashion.predict(x)
            predictions = np.argmax(predictions)
            print("Номер класса:", predictions)

            return render_template('fashion.html', title="Классификация одежды", menu=menu,
                                   fashion_model="Результат: " + str(predictions) + "\n"
                                                 + "Это " + fashion_classes[predictions])


@app.route("/mnist_CNN", methods=['POST', 'GET'])
def upload_CNN():
    if request.method == 'GET':
        return render_template('fashion_CNN.html', title="Классификация одежды 2", menu=menu, CNN_model='')
    if request.method == 'POST':
        if 'image' not in request.files:
            return "Нету пути файла"

        file = request.files['image']

        if file.filename == '':
            return "Не выбран файл"

        if file:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

            img = image.image_utils.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
            x = image.image_utils.img_to_array(img)
            np.shape(x)
            # x = x.reshape(1, 784)
            x = 255 - x
            x = np.expand_dims(x, axis=0)
            np.shape(x)
            # x /= 255

            predictions = model_fashion_CNN.predict(x)
            predictions = np.argmax(predictions)
            print("Номер класса:", predictions)

            return render_template('fashion_CNN.html', title="Классификация одежды 2", menu=menu,
                                   CNN_model="Результат: " + str(predictions) + "\n"
                                                 + "Это " + fashion_classes[predictions])


if __name__ == "__main__":
    app.run(debug=True)
