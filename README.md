# Deep_Fake_Voice_Recognition

## Overview
The **Deep_Fake_Voice_Recognition** project aims to build a robust model to detect AI-generated (deepfake) speech using machine learning techniques. This repository includes scripts and resources to train, validate, and deploy the model for real-time detection of AI-generated voices. The project is designed to identify subtle differences between human and AI-generated speech patterns.

## Dataset
The dataset used for this project is the **DEEP-VOICE** dataset, available on Kaggle. It contains real and AI-generated voice samples for training and evaluation.
- [Download the DEEP-VOICE dataset](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition/data)

## Prerequisites
Make sure to install the required Python libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Project Structure
- **data/**: Contains the training and testing audio samples.
- **models/**: Contains pre-trained and trained models for detecting deepfake voices.
- **notebooks/**: Jupyter notebooks for data exploration, feature extraction, and model training.
- **src/**: Source code for data preprocessing, model architecture, and utility functions.
- **README.md**: Project documentation.

## Usage
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your_username/Deep_Fake_Voice_Recognition.git
   ```
2. **Download the dataset** and place it in the `data/` folder.
3. **Run the training script** to start training the model:
   ```bash
   python src/train_model.py
   ```
4. **Evaluate the model**:
   ```bash
   python src/evaluate_model.py
   ```
5. **Use the real-time detection script** for testing on new audio files:
   ```bash
   python src/real_time_detection.py --file path_to_audio.wav
   ```

## Features
- **Audio Preprocessing**: Includes feature extraction like Mel spectrogram, MFCC, and audio augmentation.
- **Modeling**: Uses deep learning models such as CNNs and RNNs for audio classification.
- **Real-Time Detection**: Provides a script for detecting AI-generated speech from input audio files.
- **Model Evaluation**: Evaluates models using accuracy, precision, recall, and F1-score.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with detailed changes.

## Contact
For any queries or suggestions, feel free to reach out:

- **Fahad Kacchi**  
  Email: [fahadkacchi1998@gmail.com](mailto:fahadkacchi1998@gmail.com)
```
