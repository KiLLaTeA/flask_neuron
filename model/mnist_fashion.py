import tensorflow as tf

from keras.api.datasets import fashion_mnist
from keras import utils
from keras.src.models import Sequential
from keras.src.layers import Dense
from keras.src.layers import Dropout

# Попробуйте использовать разное количество нейронов на входном слое: 400, 600, 800, 1200.
# Добавьте в нейронную сеть скрытый слой с разным количеством нейронов: 200, 300, 400, 600, 800.
# Добавьте несколько скрытых слоев в сеть с разным количеством нейронов в каждом слое.
# Используйте разное количество эпох: 10, 15, 20, 25, 30.
# Используйте разные размеры мини-выборки (batch_size): 10, 50, 100, 200, 500.

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

nb_classes = 10
Y_train = utils.to_categorical(y_train, nb_classes)
Y_test = utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Dense(512, input_shape=(784,), activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(512, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(10, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
# optimizer=tf.keras.optimizers.Adam(0.0001)
# optimizer='adam'

model.fit(X_train, Y_train, batch_size=256, epochs=20, verbose=1)

scores = model.evaluate(X_test, Y_test, verbose=1)
print(f'Значение функции потерь (loss) на тестовых данных: {scores[0]}')
print(f'Доля верных ответов на тестовых данных, в процентах (accuracy): {round(scores[1] * 100, 4)}')

model.save('fashion.h5')