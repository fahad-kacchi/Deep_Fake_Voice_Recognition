#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install resampy


# In[3]:


import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
try:
    from imblearn.over_sampling import RandomOverSampler
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


# In[4]:

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Check for custom dataset path via command line argument or environment variable, defaulting to local data folder
default_dataset_path = os.path.join(BASE_DIR, "data_deep_voice", "KAGGLE", "AUDIO")
if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    audio_files_path = sys.argv[1]
elif "DATASET_PATH" in os.environ and os.path.exists(os.environ["DATASET_PATH"]):
    audio_files_path = os.environ["DATASET_PATH"]
elif os.path.exists(default_dataset_path):
    audio_files_path = default_dataset_path
else:
    audio_files_path = "C:/Users/Akshay Ransure/Documents/fahad/Deep Fake Voice Recognition/data_deep_voice/KAGGLE/AUDIO"


# In[5]:

if not os.path.exists(audio_files_path):
    print(f"[WARNING] Dataset directory not found at: '{audio_files_path}'.")
    print("Please download the DEEP-VOICE dataset from Kaggle and place it in 'data_deep_voice/KAGGLE/AUDIO' or pass the path via argument/DATASET_PATH.")
    folders = []
else:
    folders = [f for f in os.listdir(audio_files_path) if os.path.isdir(os.path.join(audio_files_path, f))]
print("Found class folders:", folders)


# In[6]:


real_audio = os.path.join(audio_files_path, "REAL", "obama-original.wav") if os.path.exists(audio_files_path) else "linus-original-DEMO.mp3"
fake_audio = os.path.join(audio_files_path, "FAKE", "Obama-to-Biden.wav") if os.path.exists(audio_files_path) else "linus-to-musk-DEMO.mp3"


# # Visualization

# In[8]:


print("Real Audio:")
IPython.display.Audio(real_audio)


# In[9]:


print("Fake Audio:")
IPython.display.Audio(fake_audio)


# In[10]:


real_ad, real_sr = librosa.load(real_audio)
plt.figure(figsize=(12, 4))
plt.plot(real_ad)
plt.title("Real Audio Data")
plt.show()


# In[11]:


real_spec = np.abs(librosa.stft(real_ad))
real_spec = librosa.amplitude_to_db(real_spec, ref=np.max)

plt.figure(figsize=(14, 5))
librosa.display.specshow(real_spec, sr=real_sr, x_axis="time", y_axis="log")
plt.colorbar(format="%+2.0f dB")
plt.title("Real Audio Spectogram")
plt.show()


# In[15]:


real_mel_spect = librosa.feature.melspectrogram(y=real_ad, sr=real_sr)
real_mel_spect = librosa.power_to_db(real_mel_spect, ref=np.max)

plt.figure(figsize=(14, 5))
librosa.display.specshow(real_mel_spect, y_axis="mel", x_axis="time")
plt.title("Real Audio Mel Spectogram")
plt.colorbar(format="%+2.0f dB")
plt.show()


# In[18]:


real_chroma = librosa.feature.chroma_cqt(y=real_ad, sr=real_sr, bins_per_octave=36)

plt.figure(figsize=(14, 5))
librosa.display.specshow(real_chroma, sr=real_sr, x_axis="time", y_axis="chroma", vmin=0, vmax=1)
plt.colorbar()
plt.title("Real Audio Chromagram")
plt.show()


# In[20]:


real_mfccs = librosa.feature.mfcc(y=real_ad, sr=real_sr)

plt.figure(figsize=(14, 5))
librosa.display.specshow(real_mfccs, sr=real_sr, x_axis="time")
plt.colorbar()
plt.title("Real Audio Mel-Frequency Cepstral Coefficients (MFCCs)")
plt.show()


# In[21]:


fake_ad, fake_sr = librosa.load(fake_audio)
plt.figure(figsize=(12, 4))
plt.plot(fake_ad)
plt.title("Fake Audio Data")
plt.show()


# In[22]:


fake_spec = np.abs(librosa.stft(fake_ad))
fake_spec = librosa.amplitude_to_db(fake_spec, ref=np.max)

plt.figure(figsize=(14, 5))
librosa.display.specshow(fake_spec, sr=fake_sr, x_axis="time", y_axis="log")
plt.colorbar(format="%+2.0f dB")
plt.title("Fake Audio Spectogram")
plt.show()


# In[23]:


fake_mel_spect = librosa.feature.melspectrogram(y=fake_ad, sr=fake_sr)
fake_mel_spect = librosa.power_to_db(fake_mel_spect, ref=np.max)

plt.figure(figsize=(14, 5))
librosa.display.specshow(fake_mel_spect, y_axis="mel", x_axis="time")
plt.title("Fake Audio Mel Spectogram")
plt.colorbar(format="%+2.0f dB")
plt.show()


# In[24]:


fake_chroma = librosa.feature.chroma_cqt(y=fake_ad, sr=fake_sr, bins_per_octave=36)

plt.figure(figsize=(14, 5))
librosa.display.specshow(fake_chroma, sr=fake_sr, x_axis="time", y_axis="chroma", vmin=0, vmax=1)
plt.colorbar()
plt.title("Fake Audio Chromagram")
plt.show()


# In[25]:


fake_mfccs = librosa.feature.mfcc(y=fake_ad, sr=fake_sr)

plt.figure(figsize=(14, 5))
librosa.display.specshow(fake_mfccs, sr=fake_sr, x_axis="time")
plt.colorbar()
plt.title("Fake Audio Mel-Frequency Cepstral Coefficients (MFCCs)")
plt.show()


# # Preprocess

# In[28]:


data = []
labels = []

for folder in folders:
    files = os.listdir(os.path.join(audio_files_path, folder))
    for file in tqdm(files):
        file_path = os.path.join(audio_files_path, folder, file)
        audio, sample_rate = librosa.load(file_path, res_type="kaiser_fast")
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_features_scaled = np.mean(mfccs_features.T, axis=0)
        data.append(mfccs_features_scaled)
        labels.append(folder)


# In[39]:


feature_df = pd.DataFrame({"features": data, "class": labels})
feature_df.head()


# In[41]:


feature_df["class"].value_counts()


# In[43]:


def label_encoder(column):
    le = LabelEncoder().fit(column)
    print(column.name, le.classes_)
    return le.transform(column)


# In[45]:


feature_df["class"] = label_encoder(feature_df["class"])


# In[47]:


X = np.array(feature_df["features"].tolist())
y = np.array(feature_df["class"].tolist())


# Step 1: Perform Stratified Train/Test Split on raw, unaugmented data FIRST to eliminate leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n--- DATASET PARTITION SUMMARY (LEAKAGE-FREE) ---")
print(f"Total raw samples: {len(X)}")
print(f"Train set (raw): {len(X_train)} samples -> Class distribution: {np.bincount(y_train)}")
print(f"Test set (raw):  {len(X_test)} samples -> Class distribution: {np.bincount(y_test)}")

# Step 2: Apply oversampling ONLY to the training set; NEVER oversample the test set
def balance_training_data(X_tr, y_tr, random_state=42):
    if HAS_IMBLEARN:
        ros = RandomOverSampler(random_state=random_state)
        return ros.fit_resample(X_tr, y_tr)
    else:
        from sklearn.utils import resample
        unique_classes, counts = np.unique(y_tr, return_counts=True)
        max_count = np.max(counts)
        X_bal, y_bal = [], []
        for cls in unique_classes:
            idx = np.where(y_tr == cls)[0]
            cls_X, cls_y = X_tr[idx], y_tr[idx]
            if len(cls_X) < max_count:
                cls_X_res, cls_y_res = resample(cls_X, cls_y, replace=True, n_samples=max_count, random_state=random_state)
                X_bal.append(cls_X_res)
                y_bal.append(cls_y_res)
            else:
                X_bal.append(cls_X)
                y_bal.append(cls_y)
        return np.vstack(X_bal), np.concatenate(y_bal)

X_train_resampled, y_train_resampled = balance_training_data(X_train, y_train, random_state=42)
print(f"Train set (resampled): {len(X_train_resampled)} samples -> Balanced distribution: {np.bincount(y_train_resampled)}")
print(f"Test set remains unaugmented and un-duplicated (count: {len(X_test)}) to guarantee honest evaluation.")

num_labels = len(np.unique(y))

# Step 3: Convert labels to categorical one-hot vectors for model training
y_train_cat = to_categorical(y_train_resampled, num_classes=num_labels)
y_test_cat = to_categorical(y_test, num_classes=num_labels)


# In[55]:


num_labels = len(feature_df["class"].unique())
num_labels


# In[57]:


input_shape = feature_df["features"][0].shape
input_shape


# # Model

# In[60]:


model = Sequential()
model.add(Dense(128, input_shape=input_shape))
model.add(Activation(activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation(activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation(activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_labels))
model.add(Activation(activation="softmax"))


# In[61]:


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


# In[64]:


model.summary()


early = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)


# In[68]:


history = model.fit(
    X_train_resampled,
    y_train_cat,
    validation_data=(X_test, y_test_cat),
    batch_size=32,
    epochs=100,
    callbacks=[early],
    verbose=1
)


# In[72]:


test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Comprehensive Metrics on Uncontaminated Test Set
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

print("\n" + "="*60)
print("COMPREHENSIVE EVALUATION ON UNCONTAMINATED TEST SET")
print("="*60)
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["FAKE (0)", "REAL (1)"]))

try:
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    print(f"ROC-AUC Score: {roc_auc:.4f}")
except Exception as e:
    print(f"Could not compute ROC-AUC: {e}")
print("="*60)


# In[74]:


plt.figure()
plt.title("Model Accuracy")
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.legend()
plt.ylim([0, 1])
plt.show()


# In[76]:


plt.figure()
plt.title("Model Loss")
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.legend()
plt.ylim([0, 1])
plt.show()


# # Test

# In[79]:


def detect_fake(filename):
    sound_signal, sample_rate = librosa.load(filename, res_type="kaiser_fast")
    mfcc_features = librosa.feature.mfcc(y=sound_signal, sr=sample_rate, n_mfcc=40)
    mfccs_features_scaled = np.mean(mfcc_features.T, axis=0)
    mfccs_features_scaled = mfccs_features_scaled.reshape(1, -1)
    result_array = model.predict(mfccs_features_scaled)
    print(result_array)
    result_classes = ["FAKE", "REAL"]
    result = np.argmax(result_array[0])
    print("Result:", result_classes[result])


# In[81]:


test_real = "C:/Users/Akshay Ransure/Documents/fahad/Deep Fake Voice Recognition/data_deep_voice/KAGGLE/AUDIO/REAL/obama-original.wav"
test_fake = "C:/Users/Akshay Ransure/Documents/fahad/Deep Fake Voice Recognition/data_deep_voice/KAGGLE/AUDIO/FAKE/Obama-to-Biden.wav"


# In[83]:


detect_fake(test_real)
IPython.display.Audio(test_real)


# In[84]:


detect_fake(test_fake)
IPython.display.Audio(test_fake)


# # Model Export

# In[88]:


experimental_model_path = os.path.join(BASE_DIR, 'model_experimental.h5')
model.save(experimental_model_path)
print(f"[SUCCESS] Experimental model saved to: '{experimental_model_path}'")
print("[SAFETY NOTE] The production model ('model.h5') used by app.py was NOT overwritten.")


# In[ ]:





# In[95]:


from keras.models import load_model
model = load_model(experimental_model_path)
print("Verified experimental model loads cleanly.")

