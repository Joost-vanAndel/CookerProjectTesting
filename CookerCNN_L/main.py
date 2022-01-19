import pathlib

import cv2
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

data_dir = tf.keras.utils.get_file('cooker_dataset', origin='', untar=False)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# AUTOTUNE = tf.data.AUTOTUNE
#
# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
#
# normalization_layer = layers.Rescaling(1./255)
#
# normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# image_batch, labels_batch = next(iter(normalized_ds))
# first_image = image_batch[0]
# # Notice the pixel values are now in `[0,1]`.
# print(np.min(first_image), np.max(first_image))
#
# num_classes = len(class_names)
#
# model = Sequential([
#   layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Flatten(),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(num_classes)
# ])
#
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# model.summary()
#
# epochs=50
# history = model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=epochs
# )
#
# model.save('CookerCNN_Model')

reconstructed_model = tf.keras.models.load_model("CookerCNN_Model")

# imgTest = cv2.imread('C:/Users/joost/.keras/datasets/beantest.jpg');
# imgTest = cv2.resize(imgTest, dsize=(180, 180), interpolation=cv2.INTER_CUBIC)
#
# np_image_data = np.asarray(imgTest)
# np_image_data=cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
# np_final = np.expand_dims(np_image_data, axis=0)

img = cv2.imread('C:/Users/joost/Documents/GitHub/CookerProjectTesting/ValidationImages/carrots_boiling/175IMG_6682.JPEG.jpg')
inp = cv2.resize(img, (img_width, img_height))
rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.float32)

rgb_tensor = tf.expand_dims(rgb_tensor, 0)

probability_model = tf.keras.Sequential([reconstructed_model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(rgb_tensor, steps=1)

# predict_path = tf.keras.utils.get_file('beantest.jpg', origin='')
#
# img = tf.keras.utils.load_img(
#     predict_path, target_size=(img_height, img_width)
# )
# img_array = tf.keras.utils.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0) # Create a batch

# predictions = reconstructed_model.predict(img_array)

score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
