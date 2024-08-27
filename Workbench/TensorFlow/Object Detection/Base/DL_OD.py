import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow_datasets as tfds
import pandas as pd
from tensorflow.keras.callbacks import TensorBoard

# Model name
model_name = 'VOC2007'

# Define log directory for TensorBoard
log_dir = "logs/tensorboard/" + model_name 
tensorboard_callback = TensorBoard(log_dir=log_dir)

# Load dataset Pascal VOC 2012
train_dataset, info = tfds.load('voc/2007', split='train', with_info=True)
val_dataset = tfds.load('voc/2007', split='validation', with_info=False)

# Preprocess
def preprocess_data(example):
    image = example['image']
    bbox = example['objects']['bbox'][0] 
    label = example['objects']['label'][0] 

    # Resize to 128x128
    image = tf.image.resize(image, (128, 128))

    # Normalize
    image = image / 255.0

    # Scale the bounding box to the new dimension
    bbox = tf.stack([
        bbox[0] * 128,  # ymin
        bbox[1] * 128,  # xmin
        bbox[2] * 128,  # ymax
        bbox[3] * 128   # xmax
    ], axis=0)

    # Transform label to tensor of type int32 for class prediction
    label = tf.cast(label, tf.int32)

    return image, (bbox, label)

# Apply preprocess to dataset
train_dataset = train_dataset.map(preprocess_data).batch(32).prefetch(1)
val_dataset = val_dataset.map(preprocess_data).batch(32).prefetch(1)

# Define CNN model with 2 outputs
def create_cnn_model():
    inputs = Input(shape=(128, 128, 3))
    
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    
    # Bounding box output
    bbox_output = Dense(4, activation='sigmoid', name='bbox_output')(x)
    
    # Class output
    class_output = Dense(info.features['objects']['label'].num_classes, activation='softmax', name='class_output')(x)
    
    model = Model(inputs, [bbox_output, class_output])
    return model

# Create and compile the model
model = create_cnn_model()
model.compile(optimizer=Adam(), 
              loss={
                  'bbox_output': 'mean_squared_error', 
                  'class_output': 'sparse_categorical_crossentropy'
                  },
              metrics={'bbox_output': 'mse', 'class_output': 'accuracy'}
              )

# Train the model
history = model.fit(
    train_dataset,
    validation_data = val_dataset, 
    epochs=100,
    callbacks=[tensorboard_callback] 
)

# Save data with Panda
data = {
    'Epoch': list(range(1, len(history.history['class_output_accuracy']) + 1)),
    'Train_Accuracy': history.history['class_output_accuracy'],
    'Train_Loss': history.history['loss'],
    'Val_Accuracy': history.history['val_class_output_accuracy'],
    'Val_Loss': history.history['val_loss'],
}

df = pd.DataFrame(data)
df.to_csv('./logs/csv/' + model_name + '.csv', index=False)

# Save the model
model.save(model_name + ".keras")


