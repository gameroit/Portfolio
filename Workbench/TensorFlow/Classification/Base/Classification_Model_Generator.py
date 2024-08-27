# Import packages
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard

# Define parameters
model_name = 'Classificator_mnist'

# Define log directory for TensorBoard
tensorboard_callback = TensorBoard(log_dir='logs/tensorboard/' + model_name , histogram_freq=1)

# Load dataset from tensorflow_datasets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() # Images(X) and Labels(Y)


# Image format application
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

y_train = to_categorical(y_train, num_classes = 10)
y_test = to_categorical(y_test, num_classes = 10)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

datagen = ImageDataGenerator(
    rotation_range = 30,
    width_shift_range = 0.25,
    height_shift_range = 0.25,
    zoom_range = [0.5, 1.5]
)

datagen.fit(x_train)

# Model creation
model = models.Sequential([
    layers.Conv2D(32, (2,2), input_shape=(28, 28, 1), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (2,2), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

train_dg = datagen.flow(
    x_train,
    y_train,
    batch_size = 32
)

# Model training
history = model.fit(
    train_dg,
    epochs = 60,
    validation_data = (x_test, y_test),
    batch_size = 32,
    callbacks=[tensorboard_callback] # Tensorboard plots
)

# Save data with Panda
data = {
    'Epoch': list(range(1, len(history.history['accuracy']) + 1)),
    'Accuracy': history.history['accuracy'],
    'Loss': history.history['loss'],
    'Validation Accuracy': history.history['val_accuracy'],
    'Validation Loss': history.history['val_loss']
}

df = pd.DataFrame(data)
df.to_csv('logs/csv/' + model_name + '.csv', index=False)

model.save(model_name + '.keras')