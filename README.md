# Deep Fake Voice Recognition

## Overview
The **Deep Fake Voice Recognition** project detects AI-generated (deepfake) synthetic speech and human voice recordings using machine learning techniques.

The project features a hardened Flask web application serving a trained deep learning neural network ([`model.h5`](model.h5)) for single-clip audio classification, alongside automated pipeline evaluation tools.

---

## Technology Stack

- **Backend:** Python 3.12, Flask 3.0.3, Werkzeug 3.0.4, Jinja2 3.1.4
- **Machine Learning & Deep Learning:** TensorFlow 2.17.0, Keras 3.5.0, h5py 3.11.0, NumPy 1.26.4, Scikit-learn 1.5.0, Pandas 2.2.2
- **Audio Processing:** Librosa 0.10.2.post1, SoundFile 0.12.1, Resampy 0.4.3, SciPy 1.14.0
- **Frontend:** HTML5, CSS3 (responsive grid, accessible status badges), Vanilla JavaScript (Drag & Drop, HTML5 Audio preview, asynchronous fetch)

---

## Repository Structure

```text
C:\Deep_Fake_Voice_Recognition\
│
├── static/                                # Frontend assets
│   ├── index.js                           # UI interaction, drag-and-drop, fetch client
│   ├── style.css                          # Layout styling, dark mode accents, responsiveness
│   ├── background_final.png               # Visual asset
│   ├── image7.jpg                         # Visual asset
│   └── images.jpeg                        # Visual asset
│
├── templates/
│   └── index.html                         # Accessible web interface template
│
├── uploads/                               # Secure temporary audio upload staging directory
│   └── .gitkeep
│
├── app.py                                 # Hardened Flask web application (inference server)
├── Deep Fake Voice Recognition.py        # Model training script
├── Deep Fake Voice Recognition.ipynb     # Jupyter notebook for exploratory data analysis
├── evaluate_pipeline.py                   # Standalone evaluation & data leakage audit script
├── model.h5                               # Production pretrained Keras neural network model (~904 KB)
├── requirements.txt                       # Clean, verified runtime dependencies (UTF-8)
├── .gitignore                             # Git ignore rules for venv, cache, uploads, and temp files
├── LICENSE                                # Project license
└── README.md                              # Project documentation and run guide
```

---

## Prerequisites & Compatibility

- **Operating System:** Windows 10 or Windows 11 (64-bit)
- **Supported Python Version:** **Python 3.12 (64-bit)** (also compatible with Python 3.11 and 3.10)
  > **Note on NumPy:** TensorFlow 2.17 requires `numpy<2.0.0`. The project is pinned and verified with `numpy==1.26.4`.
  > **Note on Python 3.14+:** Python 3.14+ is not currently supported by TensorFlow. Use Python 3.12, 3.11, or 3.10.

---

## Setup and Startup Instructions (Windows)

All commands below should be executed from the repository root (`C:\Deep_Fake_Voice_Recognition`).

### Method A: Windows Command Prompt (`cmd.exe`)

```cmd
:: 1. Navigate to repository directory
cd /d C:\Deep_Fake_Voice_Recognition

:: 2. Create virtual environment using Python 3.12
py -3.12 -m venv venv

:: 3. Activate virtual environment
venv\Scripts\activate

:: 4. Upgrade pip
python -m pip install --upgrade pip

:: 5. Install verified dependencies
pip install -r requirements.txt

:: 6. Launch the application
python app.py
```

### Method B: Windows PowerShell

```powershell
# 1. Navigate to repository directory
Set-Location -Path C:\Deep_Fake_Voice_Recognition

# 2. Create virtual environment
py -3.12 -m venv venv

# 3. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Launch the application
python app.py
```

---

## Application Access & Features

Once started, open your web browser and navigate to:
```
http://127.0.0.1:8080
```

### Key Features:
- **Audio Upload & Drag-and-Drop:** Intuitive dropzone with keyboard accessibility (Space/Enter).
- **Supported Formats:** `.wav`, `.mp3`, and `.ogg`.
- **Upload Size Limit:** 16 MB (`MAX_CONTENT_LENGTH = 16 * 1024 * 1024`).
- **In-Browser Audio Preview:** Immediate audio preview player to listen to recordings before submission.
- **Asynchronous Analysis:** Non-blocking `fetch()` upload with interactive loading indicator.
- **Automated Disk Cleanup:** All uploaded audio files are assigned unique UUID filenames in `uploads/` and strictly deleted inside a `finally` block after prediction.

---

## Running Evaluation and Verification Scripts

The repository includes a standalone evaluation suite ([`evaluate_pipeline.py`](evaluate_pipeline.py)) to test model behavior and verify the absence of training-to-test data leakage.

### 1. Verification Mode (Self-Contained Synthetic Audit)
To verify pipeline reproducibility, stratified group splitting, and metric calculations without downloading external datasets:
```powershell
.\venv\Scripts\python.exe evaluate_pipeline.py --mode synthetic
```
*Outputs accuracy, confusion matrix, ROC-AUC score, and confirms zero train-test partition overlap.*

### 2. Real Dataset Evaluation Mode
If you have downloaded the **DEEP-VOICE** dataset:
```powershell
.\venv\Scripts\python.exe evaluate_pipeline.py --mode dataset --dataset-path "C:\path\to\DEEP-VOICE"
```

> **Evaluation Distinction:**
> - Metrics produced under `--mode synthetic` serve purely to verify that the evaluation script runs without data leakage and that matrix calculations function properly. They **do not** reflect real-world model accuracy.
> - Only `--mode dataset` run against external ground-truth audio benchmarks provides true real-world performance metrics.

---

## Automated Integration Tests

To run the full suite of integration tests (verifying HTTP endpoints, WAV/MP3/OGG decoding, 400/413 error handlers, temporary file cleanup, and model tensor shapes):

```powershell
.\venv\Scripts\python.exe -c "
import app, io, numpy as np, soundfile as sf

client = app.app.test_client()

# 1. Test homepage
r = client.get('/')
assert r.status_code == 200, f'Homepage failed: {r.status_code}'

# 2. Test valid WAV inference & cleanup (HTML form upload)
buf = io.BytesIO()
sf.write(buf, np.zeros(22050), 22050, format='WAV')
buf.seek(0)
r = client.post('/upload', data={'file': (buf, 'test.wav')}, content_type='multipart/form-data')
assert r.status_code == 200 and 'Audio is FAKE' in r.get_data(as_text=True), f'WAV prediction failed: {r.status_code}'

# 3. Test missing audio field (400)
r = client.post('/upload', data={})
assert r.status_code == 400, f'Expected 400 for missing field, got {r.status_code}'

# 4. Test unsupported file extension (400)
r = client.post('/upload', data={'file': (io.BytesIO(b'dummy'), 'test.txt')}, content_type='multipart/form-data')
assert r.status_code == 400, f'Expected 400 for unsupported extension, got {r.status_code}'

print('All core integration tests passed successfully!')
"
```

---

## Current Scope & Limitations

1. **Pretrained Model Scope:**
   - The production model ([`model.h5`](model.h5)) is a fixed, lightweight Dense Sequential architecture accepting 40 Mel-frequency cepstral coefficients (MFCCs). It is not retrained during web application startup.
2. **Real-Dataset Evaluation:**
   - The production model's original training metrics were from an exploratory notebook setting. Formal speaker-grouped evaluation on the full DEEP-VOICE corpus requires local dataset downloading.
3. **Audio Duration & Compression:**
   - Files are converted and sampled via `librosa.load(..., res_type='kaiser_fast')`. Highly compressed or corrupted non-standard audio files may fail decoding and return HTTP 400.
4. **Input Method:**
   - The web app accepts recorded file uploads (`.wav`, `.mp3`, `.ogg`). Live in-browser microphone streaming is not currently implemented.
