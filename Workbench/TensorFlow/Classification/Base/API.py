from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('Classificator_mnist.keras')

# Main page
@app.route('/')
def index():
    return render_template('index.html')

# Call to the model
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file'].read()
    img = Image.open(io.BytesIO(file))
    
    if img.mode == 'RGBA':
        img = img.convert("L")
    
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch

    # Predict
    score = tf.nn.softmax(model.predict(img_array)[0])
    predicted_class = int(np.argmax(score))
    confidence = float(np.max(score))

    response = {
        'predicted_class': predicted_class,
        'confidence': confidence
    }

    return jsonify(response)

# Start Flask server
if __name__ == '__main__':
    app.run(debug=True)
