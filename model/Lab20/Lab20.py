import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from keras import utils, layers, losses, optimizers, applications
from keras import Input, Model


from keras.src.models import Sequential

from PIL import Image
import pathlib

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
print(path_to_zip)
PATH = os.path.join(path_to_zip, 'cats_and_dogs_filtered')
print(PATH)

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
print(train_dir)
print(validation_dir)

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train_dataset = utils.image_dataset_from_directory(train_dir,
                                                        shuffle=True,
                                                        batch_size=BATCH_SIZE,
                                                        image_size=IMG_SIZE)

validation_dataset = utils.image_dataset_from_directory(validation_dir,
                                                         shuffle=True,
                                                         batch_size=BATCH_SIZE,
                                                         image_size=IMG_SIZE)

class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    # plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.savefig('plt_data.png')

val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = Sequential([
  layers.RandomFlip('horizontal'),
  layers.RandomRotation(0.2),
])

for image, _ in train_dataset.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
    # plt.imshow(augmented_image[0] / 255)
    plt.axis('off')
plt.savefig('plt_rotate.png')

preprocess_input = applications.mobilenet_v2.preprocess_input

IMG_SHAPE = IMG_SIZE + (3,)
print(IMG_SHAPE)

base_model = applications.MobileNetV2(input_shape=IMG_SHAPE,
                                       include_top=True,
                                       weights='imagenet')

img = utils.load_img("2k.png", target_size=(160, 160))
img_array = utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

preprocessed_img = applications.mobilenet_v2.preprocess_input(img_array)

predictions = base_model.predict(preprocessed_img)

decoded_predictions = applications.mobilenet_v2.decode_predictions(predictions, top=5)

print(decoded_predictions)

# ПЕРЕНОС ОБУЧЕНИЯ

base_model = applications.MobileNetV2(input_shape=IMG_SHAPE,
                                       include_top=False,
                                       weights='imagenet')

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)          # 333

base_model.trainable = False
base_model.summary()                # Удалить

global_average_layer = layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = layers.Dense(1, activation = 'sigmoid')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

inputs = Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(optimizer=optimizers.Adam(learning_rate=base_learning_rate),
              loss=losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()     # тоже удалить

history = model.fit(train_dataset,
                    epochs=15,
                    validation_data=validation_dataset)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
plt.savefig('plt_accuracy.jpg')

loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)

model.save("trans_learn.keras")
# model.save("trans_learn.h5")