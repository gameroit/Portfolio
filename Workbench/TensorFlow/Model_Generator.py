# Import packages
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import datetime

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard

# Define log directory for TensorBoard
model_name = "SimpleClassificator"
log_dir = "logs/tensorboard/" + model_name + "_" + datetime.datetime.now().strftime("%d_%m_%Y-%H_%M")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Load dataset from tensorflow_datasets
dataset, info = tfds.load('tf_flowers', with_info=True, as_supervised=True)

# Define training dataset and validation dataset
train_ds, val_ds = dataset['train'], dataset['train']
train_size = int(0.8 * info.splits['train'].num_examples)
val_size = int(0.2 * info.splits['train'].num_examples)
train_ds = train_ds.take(train_size)
val_ds = val_ds.skip(train_size).take(val_size)

# Define of image dimensions
img_height = 180
img_width = 180

# Image formatting function
def format_image(image, label):
    image = tf.image.resize(image, (img_height, img_width))
    image = image /255.0
    return image, label

# Data augmentation function
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    image = tf.image.random_crop(image, size=[img_height, img_width, 3])
    return image, label

# Image format application
train_ds = train_ds.map(format_image, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(format_image, num_parallel_calls=tf.data.AUTOTUNE)

# Data augmentation application
train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

# Mix the trainning dataset
train_ds = train_ds.shuffle(buffer_size=1000)

# Batch creation
batch_size = 32
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

# Pipeline optimization
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Model creation
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.7),
    layers.Dense(32, activation='relu'),
    layers.Dense(info.features['label'].num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Model training
epochs = 25
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[tensorboard_callback] # Tensorboard plots
)

# Matplot plots (static, faster and more exportable)
"""
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()
"""
model.save(model_name + '.h5')