import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io

app = Flask(__name__)

# Load dataset classes
dataset, info = tfds.load('voc/2007', split='train', with_info=True)
class_names = info.features['objects']['label'].names

# Load model
model = tf.keras.models.load_model('VOC2007.keras') 

# Main page
@app.route('/')
def index():
    return render_template('index.html')

# List classes
@app.route('/classes')
def get_classes():
    return jsonify(class_names)

# Call to the model
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        image = image.convert('RGB')  
        image_array = image.resize((128, 128))  
        image_array = np.array(image_array) / 255.0  
        image_array = image_array.reshape((1, 128, 128, 3))  

        # Predict
        predictions = model.predict(image_array)
        predictions = predictions[1][0]  
        bounding_box = [
            predictions[0] * image.width,
            predictions[1] * image.height,
            predictions[2] * image.width,
            predictions[3] * image.height
        ]
        print(class_names[int(np.argmax(predictions[4:]))])
        result = {
            'bounding_box': bounding_box,  
            'class_label': class_names[int(np.argmax(predictions[4:]))]  
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Start Flask server
if __name__ == '__main__':
    app.run(debug=True)
