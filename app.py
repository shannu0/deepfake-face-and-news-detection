import os
import nltk
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import base64
import numpy as np
from model import predict_deepfake, predict_news
import time
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Set NLTK data path to the directory where your datasets are stored
nltk_data_path = 'nltk_data'
nltk.data.path.append(nltk_data_path)

# Function to check for the presence of NLTK datasets
def check_nltk_data():
    datasets = {
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'omw-1.4': 'corpora/omw-1.4'
    }
    for dataset, path in datasets.items():
        try:
            nltk.data.find(path)
            print(f"{dataset} is present.")
        except LookupError:
            print(f"{dataset} is not found. Please ensure it is installed.")

# Check for NLTK data
check_nltk_data()

@app.route('/predict_deepfake', methods=['POST'])
def deepfake_prediction():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            image = Image.open(file.stream)
            predictions, face_with_mask = predict_deepfake(image)

            real_prob = predictions['real']
            fake_prob = predictions['fake']
            final_output = predictions['final_result']

            # Prepare face_with_mask for JSON response
            if isinstance(face_with_mask, np.ndarray):
                face_with_mask = Image.fromarray(face_with_mask)

            buffered = BytesIO()
            face_with_mask.save(buffered, format="JPEG")
            face_with_mask_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return jsonify({
                'real_prob': real_prob,
                'fake_prob': fake_prob,
                'final_output': final_output,
                'face_with_mask': face_with_mask_b64
            })
    return jsonify({'error': 'No image provided'}), 400

@app.route('/predict_news', methods=['POST'])
def news_prediction():
    if request.method == 'POST':
        text = request.form['news_text']
        prediction = predict_news(text)
        return jsonify({'prediction': prediction})
    return jsonify({'error': 'No text provided'}), 400

if __name__ == '__main__':
    app.run(debug=False)
