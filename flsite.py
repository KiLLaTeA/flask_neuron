import os

import torch
import cv2

from PIL import Image

from googletrans import Translator

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from model.Lab14.neuron import SingleNeuron

import keras

from keras.src.legacy.preprocessing import image
from keras import utils, models

app = Flask(__name__)

menu = [
    {"name": "(Lab_14) Нейронная сеть", "url": "neural_network"},
    {"name": "(Lab_16) API классификации", "url": "api_class?Perimeter=16.72&LengthOfKernel=6.303&WidthOfKernel=3.791"},
    {"name": "(Lab_16) API регрессии", "url": "api_reg?LengthOfKernel=6.303"},
    {"name": "(Lab_17) Одежда", "url": "mnist_fashion"},
    {"name": "(Lab_18) Одежда", "url": "mnist_CNN"},
    {"name": "(Lab_19) Аугментация", "url": "Lab_19"},
    {"name": "(Lab_20) Дообучение", "url": "Lab_20"},
    {"name": "(Lab_21) Детектирование", "url": "Lab_21"}
]

fashion_classes = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress",
    4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker",
    8: "Bag", 9: "Ankle boot"
}

UPLOAD_FOLDER = 'model/clothes'
LAB19_FOLDER = 'model/Lab19'
LAB20_FOLDER = 'model/Lab20'
DETECT_FOLDER = './static/images/Lab21/detect'
ORIGINAL_FOLDER = './static/images/Lab21/original_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['LAB19_FOLDER'] = LAB19_FOLDER
app.config['LAB20_FOLDER'] = LAB20_FOLDER
app.config['DETECT_FOLDER'] = DETECT_FOLDER
app.config['ORIGINAL_FOLDER'] = ORIGINAL_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(DETECT_FOLDER):
    os.makedirs(DETECT_FOLDER)

if not os.path.exists(ORIGINAL_FOLDER):
    os.makedirs(ORIGINAL_FOLDER)

new_neuron = SingleNeuron(input_size=3)
new_neuron.load_weights('model/neuron_weights.txt')
model_reg = tf.keras.models.load_model('model/Lab16/regression_model.h5')
model_class = tf.keras.models.load_model('model/Lab16/classification_model.h5')
model_fashion = tf.keras.models.load_model('model/Lab17/fashion.h5')
model_fashion_CNN = tf.keras.models.load_model('model/Lab18/fashion_CNN.h5')
model_CATSDOGS = tf.keras.models.load_model('model/Lab19/cats_dogs_augment.h5')
model_trans = tf.keras.models.load_model('./model/Lab20/trans_learn.keras')
model_YOLO = torch.hub.load(repo_or_dir='./model/Lab21/yolov5', model='yolov5s', source = 'local')

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


@app.route("/Lab_19", methods=['POST', 'GET'])
def animals():
    if request.method == 'GET':
        return render_template('animals.html', title="Классификация с аугментацией", menu=menu, animal_model='')
    if request.method == 'POST':
        if 'image' not in request.files:
            return "Нету пути файла"

        file = request.files['image']

        if file.filename == '':
            return "Не выбран файл"

        if file:
            translator = Translator()

            img_height = 180
            img_width = 180
            class_names = ['cats', 'dogs']

            file.save(os.path.join(app.config['LAB19_FOLDER'], file.filename))
            img_path = os.path.join(app.config['LAB19_FOLDER'], file.filename)

            img = utils.load_img(
                img_path, target_size=(img_height, img_width)
            )

            img_array = utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            predictions = model_CATSDOGS.predict(img_array)
            print(predictions)
            score = tf.nn.softmax(predictions[0])

            class_detect = class_names[np.argmax(score)]

            translated = translator.translate(class_detect[0:3], src='en', dest='ru')

            return render_template('animals.html', title="Классификация с аугментацией", menu=menu,
                animal_model=f"Похоже на этом фото есть {translated.text} с вероятностью {100 * np.max(score)} процентов.")


@app.route("/Lab_20", methods=['POST', 'GET'])
def trans_learning():
    if request.method == 'GET':
        return render_template('trans.html', title="Классификация c дообучением", menu=menu, trans_model='')
    if request.method == 'POST':
        if 'image' not in request.files:
            return "Нету пути файла"

        file = request.files['image']

        if file.filename == '':
            return "Не выбран файл"

        if file:
            translator = Translator()

            class_names = ['cats', 'dogs']

            file.save(os.path.join(app.config['LAB20_FOLDER'], file.filename))
            img_path = os.path.join(app.config['LAB20_FOLDER'], file.filename)

            img = utils.load_img(
                img_path, target_size=(160, 160)
            )
            img_array = utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            predictions = model_trans.predict_on_batch(img_array)
            predictions = model_trans.predict_on_batch(img_array).flatten()
            predictions = tf.nn.sigmoid(predictions)
            predictions = tf.where(predictions < 0.5, 0, 1)

            class_detect = class_names[int(predictions)]
            translated = translator.translate(class_detect[0:3], src='en', dest='ru')

            return render_template('trans.html', title="Классификация c дообучением", menu=menu,
                trans_model=f"На этом изображении -  {translated.text}")


@app.route("/Lab_21", methods=['POST', 'GET'])
def upload_21():
    if request.method == 'GET':
        return render_template('detection.html', title="Детектирование объектов", menu=menu, detected_images='')
    if request.method == 'POST':
        if 'image' not in request.files:
            return "Нету пути файла"

        file = request.files['image']

        if file.filename == '':
            return "Не выбран файл"

        if file:
            img_path = os.path.join(app.config['ORIGINAL_FOLDER'], file.filename)
            file.save(img_path)

            results = model_YOLO(img_path)
            results.save(labels=True, save_dir=os.path.join(app.config['DETECT_FOLDER']), exist_ok=True)

            desired_classes = ['person']
            confidence_threshold = 0.75

            filtered_results = results.pandas().xyxy[0]
            filtered_results = filtered_results[(filtered_results['name'].isin(desired_classes)) & (
                        filtered_results['confidence'] >= confidence_threshold)]

            img = Image.open(os.path.join(app.config['ORIGINAL_FOLDER'], file.filename))

            filtered_image = np.array(img)
            filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_RGB2BGR)

            for _, row in filtered_results.iterrows():
                label = row['name']
                conf = row['confidence']
                xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']]
                cv2.rectangle(filtered_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
                cv2.rectangle(filtered_image, (int(xmin), int(ymin) - 20), (int(xmax), int(ymin)), (255, 0, 0), -1)
                cv2.putText(filtered_image, f'{label} {conf:.2f}', (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 1)

            cv2.imwrite(os.path.join(app.config['DETECT_FOLDER'], '2_detected_'+file.filename), filtered_image)

            # Фильтрация результатов по классам и вероятности
            filtered_results = []
            for *box, conf, cls in results.xyxy[0]:
                label = model_YOLO.names[int(cls)]
                if label in desired_classes and conf >= confidence_threshold:
                    filtered_results.append((box, conf, cls))
            image_np = np.array(img)

            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            paths = [
                "/static/images/Lab21/original_images/" + file.filename,
                "/static/images/Lab21/detect/" + file.filename,
                "/static/images/Lab21/detect/" + '2_detected_'+file.filename,
            ]

            cropped_images = []
            count = 0
            for box, conf, cls in filtered_results:
                count += 1
                xmin, ymin, xmax, ymax = map(int, box)
                cropped_image = image_np[ymin:ymax, xmin:xmax]
                cropped_images.append(cropped_image)
                path_classes = os.path.join(app.config['DETECT_FOLDER'], '3_class'+ str(count) +'_' + file.filename)
                cv2.imwrite(path_classes, cropped_image)
                paths.append("/static/images/Lab21/detect/" + '3_class'+ str(count) +'_' + file.filename)

            print(paths)

            return render_template('detection.html', title="Детектирование объектов", menu=menu,
                                   detected_images=paths)


if __name__ == "__main__":
    app.run(debug=True)
