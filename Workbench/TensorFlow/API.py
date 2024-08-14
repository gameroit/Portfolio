from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

dataset, info = tfds.load('tf_flowers', with_info=True, as_supervised=True)
class_names = info.features['label'].names

# Cargar el modelo entrenado
model = tf.keras.models.load_model('SimpleClassificator.h5')

# Dimensiones de las imágenes de entrada
img_height = 180
img_width = 180

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classes')
def get_classes():
    return jsonify(class_names)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file'].read()
    img = Image.open(io.BytesIO(file))

    if img.mode == 'RGBA':
        img = img.convert('RGB')

    img = img.resize((img_width, img_height))
    img_array = np.array(img) / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)  # Crear un batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = int(np.argmax(score))
    confidence = float(np.max(score))

    predicted_class_name = class_names[predicted_class]

    response = {
        'predicted_class': predicted_class_name,
        'confidence': confidence
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
