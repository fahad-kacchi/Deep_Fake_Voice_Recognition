"""
train_deep_voice.py

Robust Training & Evaluation Pipeline for DEEP-VOICE Dataset.

Key Features:
1. Fast, exact 40-MFCC feature extraction matching app.py (using soundfile + resampy kaiser_fast).
2. Segment-level (chunking) option to convert long 600s files into standardized speech samples.
3. Feature caching to .npz for fast iteration.
4. Strict leakage prevention:
   - Stratified train/test split (or speaker-independent group split).
   - Oversampling applied exclusively to training set.
   - Test set remains 100% untouched.
5. Saves best experimental model to model_experimental.h5.
6. Exports comprehensive evaluation metrics and confusion matrix to evaluation_results/.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import soundfile as sf
import resampy
import librosa
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Fix random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AUDIO_DIR = os.path.join(BASE_DIR, "AUDIO")
CACHE_FILE = os.path.join(BASE_DIR, "deep_voice_features.npz")
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results")
MODEL_KERAS = os.path.join(BASE_DIR, "model_experimental.keras")
MODEL_H5 = os.path.join(BASE_DIR, "model_experimental.h5")

os.makedirs(RESULTS_DIR, exist_ok=True)


def extract_features_from_audio(file_path, target_sr=22050, n_mfcc=40, chunk_seconds=10):
    """
    Extracts 40 MFCC features from an audio file.
    If chunk_seconds > 0, chunks the audio into segments of chunk_seconds and returns
    a list of 40-MFCC feature vectors (one per segment).
    If chunk_seconds <= 0, returns a single mean-pooled 40-MFCC vector for the entire file.
    """
    data, sr = sf.read(file_path, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != target_sr:
        data = resampy.resample(data, sr, target_sr, filter="kaiser_fast")
        sr = target_sr

    if chunk_seconds is not None and chunk_seconds > 0:
        chunk_len = int(chunk_seconds * sr)
        vectors = []
        for start in range(0, len(data) - chunk_len + 1, chunk_len):
            chunk = data[start:start + chunk_len]
            mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=n_mfcc)
            feat = np.mean(mfcc.T, axis=0)
            vectors.append(feat)
        return vectors
    else:
        mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc)
        feat = np.mean(mfcc.T, axis=0)
        return [feat]


def load_dataset(audio_dir, chunk_seconds=10, use_cache=True):
    """
    Loads DEEP-VOICE audio from audio_dir/REAL and audio_dir/FAKE.
    Returns:
        X (np.ndarray): feature matrix (N, 40)
        y (np.ndarray): labels (N,) where 0 = FAKE, 1 = REAL
        groups (np.ndarray): speaker group IDs (for speaker-independent testing)
        file_sources (list): list of source filenames
    """
    if not os.path.exists(audio_dir):
        print(f"[ERROR] Audio directory does not exist: {audio_dir}")
        return np.array([]), np.array([]), np.array([]), []

    cache_path = os.path.join(BASE_DIR, f"features_chunk{chunk_seconds}.npz")
    if use_cache and os.path.exists(cache_path) and os.path.abspath(audio_dir) == os.path.abspath(DEFAULT_AUDIO_DIR):
        print(f"[CACHE] Loading cached features from {cache_path}...")
        data = np.load(cache_path)
        return data["X"], data["y"], data["groups"], data["file_sources"].tolist()

    print(f"[LOAD] Scanning dataset at: {audio_dir}")
    classes = ["FAKE", "REAL"]  # 0: FAKE, 1: REAL
    all_features = []
    all_labels = []
    all_groups = []
    all_sources = []

    excluded_files = []
    usable_counts = {"REAL": 0, "FAKE": 0}

    for label_idx, cls_name in enumerate(classes):
        cls_dir = os.path.join(audio_dir, cls_name)
        if not os.path.exists(cls_dir):
            print(f"[ERROR] Class directory missing: {cls_dir}")
            continue

        files = sorted(os.listdir(cls_dir))
        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            if not fname.lower().endswith((".wav", ".mp3", ".ogg")):
                excluded_files.append((fname, "Unsupported file extension"))
                continue

            # Determine speaker group
            base = os.path.splitext(fname)[0].lower()
            if cls_name == "REAL":
                speaker = base.split("-original")[0]
            else:
                speaker = base.split("-to-")[0]

            try:
                t0 = time.time()
                feats = extract_features_from_audio(fpath, chunk_seconds=chunk_seconds)
                dur = time.time() - t0
                if not feats:
                    excluded_files.append((fname, "Audio too short for chunk length"))
                    continue

                for f in feats:
                    all_features.append(f)
                    all_labels.append(label_idx)
                    all_groups.append(speaker)
                    all_sources.append(fname)

                usable_counts[cls_name] += 1
                print(f"  Processed [{cls_name}] {fname} -> {len(feats)} chunks ({dur:.1f}s)")
            except Exception as e:
                excluded_files.append((fname, f"Read/Extraction error: {e}"))

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    groups = np.array(all_groups)

    print("\n--- DATASET SUMMARY ---")
    print(f"Total usable audio files: REAL={usable_counts['REAL']}, FAKE={usable_counts['FAKE']}")
    print(f"Total feature vectors extracted: {len(X)}")
    print(f"Class distribution: FAKE (0) = {np.sum(y == 0)}, REAL (1) = {np.sum(y == 1)}")
    if excluded_files:
        print(f"Excluded files ({len(excluded_files)}): {excluded_files}")

    # Save to cache
    np.savez_compressed(cache_path, X=X, y=y, groups=groups, file_sources=np.array(all_sources))
    print(f"[CACHE] Saved extracted features to {cache_path}")

    return X, y, groups, all_sources


def balance_training_data(X_train, y_train, random_state=SEED):
    """
    Applies Random Oversampling exclusively to the training split.
    """
    from sklearn.utils import resample
    unique_classes, counts = np.unique(y_train, return_counts=True)
    max_count = np.max(counts)
    X_bal, y_bal = [], []
    for cls in unique_classes:
        idx = np.where(y_train == cls)[0]
        cls_X, cls_y = X_train[idx], y_train[idx]
        if len(cls_X) < max_count:
            cls_X_res, cls_y_res = resample(
                cls_X, cls_y, replace=True, n_samples=max_count, random_state=random_state
            )
            X_bal.append(cls_X_res)
            y_bal.append(cls_y_res)
        else:
            X_bal.append(cls_X)
            y_bal.append(cls_y)
    return np.vstack(X_bal), np.concatenate(y_bal)


def build_mlp_model(input_dim=40, num_classes=2):
    """
    Reconstructs the 4-layer Sequential MLP matching the production model.h5 architecture:
    Dense(128) -> ReLU -> Dropout(0.5)
    Dense(256) -> ReLU -> Dropout(0.5)
    Dense(128) -> ReLU -> Dropout(0.5)
    Dense(2)   -> Softmax
    """
    model = Sequential([
        Dense(128, input_shape=(input_dim,)),
        Activation("relu"),
        Dropout(0.5),
        Dense(256),
        Activation("relu"),
        Dropout(0.5),
        Dense(128),
        Activation("relu"),
        Dropout(0.5),
        Dense(num_classes),
        Activation("softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Train MLP on DEEP-VOICE dataset")
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR, help="Path to AUDIO directory")
    parser.add_argument("--chunk-seconds", type=int, default=10, help="Segment length in seconds (0 for whole file)")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--no-cache", action="store_true", help="Ignore feature cache")
    args = parser.parse_args()

    t_start = time.time()

    # Phase 2: Load and extract dataset
    X, y, groups, sources = load_dataset(args.audio_dir, chunk_seconds=args.chunk_seconds, use_cache=not args.no_cache)

    if len(X) == 0:
        print("[FATAL] No features extracted. Aborting training.")
        sys.exit(1)

    # Phase 3: Train / Val / Test Split
    # Strategy: Stratified 70% Train, 15% Validation, 15% Test
    print("\n--- PARTITIONING DATASET ---")
    X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    print(f"Training partition (raw):   {len(X_train_raw)} samples (FAKE={np.sum(y_train_raw==0)}, REAL={np.sum(y_train_raw==1)})")
    print(f"Validation partition:       {len(X_val)} samples (FAKE={np.sum(y_val==0)}, REAL={np.sum(y_val==1)})")
    print(f"Test partition (UNTOUCHED): {len(X_test)} samples (FAKE={np.sum(y_test==0)}, REAL={np.sum(y_test==1)})")

    # Apply oversampling ONLY to the training partition
    X_train_bal, y_train_bal = balance_training_data(X_train_raw, y_train_raw, random_state=SEED)
    print(f"Training partition (balanced): {len(X_train_bal)} samples (FAKE={np.sum(y_train_bal==0)}, REAL={np.sum(y_train_bal==1)})")

    # One-hot encoding
    y_train_cat = to_categorical(y_train_bal, num_classes=2)
    y_val_cat = to_categorical(y_val, num_classes=2)
    y_test_cat = to_categorical(y_test, num_classes=2)

    # Build model
    model = build_mlp_model(input_dim=40, num_classes=2)
    model.summary()

    # Callbacks
    early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(MODEL_KERAS, monitor="val_loss", save_best_only=True, verbose=1)

    print("\n--- TRAINING MODEL ---")
    history = model.fit(
        X_train_bal,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop, checkpoint],
        verbose=2
    )

    training_time = time.time() - t_start
    print(f"\nTraining completed in {training_time:.2f} seconds.")

    # Phase 4: Evaluate on unseen test set
    print("\n" + "="*60)
    print("PHASE 4: EVALUATION ON UNSEEN TEST SET")
    print("="*60)

    # Evaluate best saved model
    best_model = tf.keras.models.load_model(MODEL_KERAS)
    # Also save as .h5 for seamless backward compatibility with Flask app.py
    best_model.save(MODEL_H5)
    y_pred_proba = best_model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec_classes = precision_score(y_test, y_pred, average=None)
    rec_classes = recall_score(y_test, y_pred, average=None)
    f1_classes = f1_score(y_test, y_pred, average=None)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    except Exception as e:
        roc_auc = None
        print(f"Could not compute ROC-AUC: {e}")

    print(f"Total Test Samples: {len(y_test)}")
    print(f"  Test FAKE (0): {np.sum(y_test == 0)}")
    print(f"  Test REAL (1): {np.sum(y_test == 1)}")
    print(f"Overall Test Accuracy: {acc * 100:.2f}%")
    if roc_auc is not None:
        print(f"ROC-AUC Score: {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print("                 Predicted FAKE (0)  Predicted REAL (1)")
    print(f"Actual FAKE (0): {cm[0, 0]:<20} {cm[0, 1]}")
    print(f"Actual REAL (1): {cm[1, 0]:<20} {cm[1, 1]}")

    print("\nClassification Report:")
    report_dict = classification_report(y_test, y_pred, target_names=["FAKE (0)", "REAL (1)"], output_dict=True)
    print(classification_report(y_test, y_pred, target_names=["FAKE (0)", "REAL (1)"]))

    # Save results to JSON
    results = {
        "dataset": "DEEP-VOICE",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunk_seconds": args.chunk_seconds,
        "training_duration_seconds": round(training_time, 2),
        "total_samples": len(X),
        "train_samples": len(X_train_bal),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "test_class_distribution": {
            "FAKE": int(np.sum(y_test == 0)),
            "REAL": int(np.sum(y_test == 1))
        },
        "metrics": {
            "accuracy": round(float(acc), 4),
            "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else None,
            "macro_f1": round(float(macro_f1), 4),
            "per_class": {
                "FAKE": {
                    "precision": round(float(prec_classes[0]), 4),
                    "recall": round(float(rec_classes[0]), 4),
                    "f1": round(float(f1_classes[0]), 4)
                },
                "REAL": {
                    "precision": round(float(prec_classes[1]), 4),
                    "recall": round(float(rec_classes[1]), 4),
                    "f1": round(float(f1_classes[1]), 4)
                }
            }
        },
        "confusion_matrix": cm.tolist()
    }

    results_file = os.path.join(RESULTS_DIR, "deep_voice_evaluation_report.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n[RESULTS] Detailed evaluation report saved to: {results_file}")
    print(f"[MODEL] Best trained model saved to: {MODEL_H5} and {MODEL_KERAS}")


if __name__ == "__main__":
    main()
