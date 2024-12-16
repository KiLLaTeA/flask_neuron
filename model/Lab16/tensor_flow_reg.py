import tensorflow as tf
import pandas as pd
import numpy as np

data_fr = pd.read_csv('../seeds_Classification.csv')
data_fr = data_fr.sample(frac=1)

dsx = data_fr[['lengthOfKernel']]
dsy = data_fr[['widthOfKernel']]
data_x = np.array(dsx).reshape(1, -1)[0]
data_y = np.array(dsy).reshape(1, -1)[0]

print(data_x)
print(data_y)

l0 = tf.keras.layers.Dense(units=3, input_shape=(1,))
l1 = tf.keras.layers.Dense(units=3)
l2 = tf.keras.layers.Dense(units=1, activation='linear')
model_reg = tf.keras.Sequential([l0, l1, l2])
model_reg.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.0001))
model_reg.fit(data_x, data_y, epochs=1000)

print(model_reg.predict(np.array([6.303])))
print(model_reg.predict(np.array([5.348])))

# Сохранение модели для регрессии
model_reg.save('regression_model.h5')