import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn import preprocessing

data_fr = pd.read_csv('../seeds_Classification.csv')
data_fr = data_fr.sample(frac=1)

dsx = data_fr[['perimeter', 'lengthOfKernel', 'widthOfKernel']]
dsy = data_fr[['seedType']]
data_x = np.array(dsx)

scaler = preprocessing.MinMaxScaler()
dsy = scaler.fit_transform(dsy)
data_y = dsy.reshape(1, -1)[0]

model_class = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_shape=(3,)),
    tf.keras.layers.Dense(3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_class.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001))
# binary_crossentropy
# mean_squared_error

model_class.fit(data_x, data_y, epochs=1000, batch_size=32)

print(model_class.predict(np.array([[16.72, 6.303, 3.791]]))) # Второй сорт (1)
print(model_class.predict(np.array([[13.85, 5.348, 3.156]]))) # Первый сорт (0)

model_class.save('classification_model.h5')