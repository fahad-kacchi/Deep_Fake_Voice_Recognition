import os
import uuid
import logging
import numpy as np
import librosa
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from keras.models import load_model




BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'ogg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB maximum upload limit

# Ensure the upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Handle oversized file uploads
@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    return render_template('index.html', prediction_text="File size exceeds the 16 MB limit"), 413

# Verify model file exists before attempting to load
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at: '{MODEL_PATH}'. "
        "Please ensure 'model.h5' is located in the application directory."
    )

# Load your model
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model from '{MODEL_PATH}': {e}") from e

def allowed_file(filename):
    if not filename or '.' not in filename:
        return False
    return filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', prediction_text="No file part in request"), 400

    file = request.files['file']
    if not file or file.filename == '':
        return render_template('index.html', prediction_text="No selected file"), 400

    if not allowed_file(file.filename):
        return render_template('index.html', prediction_text="File type not allowed. Supported formats: .wav, .mp3, .ogg"), 400

    original_filename = secure_filename(file.filename)
    if not original_filename or '.' not in original_filename:
        return render_template('index.html', prediction_text="Invalid filename"), 400

    # Use UUID to prevent path traversal, filename collisions, and race conditions
    ext = original_filename.rsplit('.', 1)[1].lower()
    safe_filename = f"{uuid.uuid4().hex}.{ext}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)

    try:
        file.save(file_path)

        # Check for empty file
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return render_template('index.html', prediction_text="Uploaded audio file is empty"), 400

        # Perform detection
        prediction = detect_fake(file_path)

        if prediction == 0:
            return render_template('index.html', prediction_text="Audio is FAKE")
        else:
            return render_template('index.html', prediction_text="Audio is REAL")

    except ValueError as ve:
        # Client-facing audio decoding or feature validation error
        return render_template('index.html', prediction_text=str(ve)), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return render_template('index.html', prediction_text="An error occurred while processing the audio file"), 500
    finally:
        # Guarantee cleanup of temporary uploaded audio file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError as err:
                logger.warning(f"Failed to remove temporary file {file_path}: {err}")


def detect_fake(filename):
    # Load and decode the audio file safely
    try:
        sound_signal, sample_rate = librosa.load(filename, res_type="kaiser_fast")
    except Exception as e:
        raise ValueError("Invalid or corrupted audio file. Could not decode audio.") from e

    if sound_signal is None or len(sound_signal) == 0:
        raise ValueError("Audio file contains no audible data.")

    # Extract MFCC features
    mfcc_features = librosa.feature.mfcc(y=sound_signal, sr=sample_rate, n_mfcc=40)

    # Scale the features
    mfccs_features_scaled = np.mean(mfcc_features.T, axis=0)

    if np.isnan(mfccs_features_scaled).any():
        raise ValueError("Unable to extract valid audio features from the provided file.")

    mfccs_features_scaled = mfccs_features_scaled.reshape(1, -1)

    # Predict
    result_array = model.predict(mfccs_features_scaled, verbose=0)
    print(result_array)  # For debugging purposes

    result_classes = ["FAKE", "REAL"]
    result = np.argmax(result_array[0])

    print("Result:", result_classes[result])  # For debugging purposes

    return result

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    # Secure local development configuration: localhost only, port 8080, debug disabled
    app.run(host="127.0.0.1", port=8080, debug=False)
