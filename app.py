import os
import pickle
import numpy as np
import librosa
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename


import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model




app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'ogg'}

# Load your model
model = load_model("Deep Fake Voice Recognition/model.h5")

# Load your model
# model = pickle.load(open("C:/Users/Akshay Ransure/Documents/fahad/Deep Fake Voice Recognition/build.pkl", 'rb'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', prediction_text="No file part")
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction_text="No selected file")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Call the detect_fake function
        prediction = detect_fake(file_path)
        
        os.remove(file_path)  # Clean up the file after processing
        
        if prediction == 0:
            return render_template('index.html', prediction_text="Audio is FAKE")
        else:
            return render_template('index.html', prediction_text="Audio is REAL")
    else:
        return render_template('index.html', prediction_text="File type not allowed")


def detect_fake(filename):
    # Load the audio file
    sound_signal, sample_rate = librosa.load(filename, res_type="kaiser_fast")
    
    # Extract MFCC features
    mfcc_features = librosa.feature.mfcc(y=sound_signal, sr=sample_rate, n_mfcc=40)
    
    # Scale the features
    mfccs_features_scaled = np.mean(mfcc_features.T, axis=0)
    mfccs_features_scaled = mfccs_features_scaled.reshape(1, -1)
    
    # Predict
    result_array = model.predict(mfccs_features_scaled)
    print(result_array)  # For debugging purposes
    
    result_classes = ["FAKE", "REAL"]
    result = np.argmax(result_array[0])
    
    print("Result:", result_classes[result])  # For debugging purposes
    
    return result

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host="0.0.0.0", port=8080, debug=True)
