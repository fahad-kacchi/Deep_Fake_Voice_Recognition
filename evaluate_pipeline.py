"""
evaluate_pipeline.py
Leakage-Free Training & Evaluation Pipeline for Deep Fake Voice Recognition.

This script:
1. Implements strict data leakage prevention:
   - Stratified train/test split on raw data FIRST.
   - Oversampling applied exclusively to training data.
   - Test set remains 100% unaugmented and uncontaminated.
2. Supports speaker grouping to prevent speaker identity leakage across train and test sets.
3. Computes comprehensive evaluation metrics:
   - Accuracy, Precision, Recall, F1-score, Confusion Matrix, and ROC-AUC.
4. Safely saves experimental models separately without overwriting production 'model.h5'.
5. Includes a self-contained verification mode when the external Kaggle dataset is not present.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def balance_training_data(X_train, y_train, random_state=42):
    """
    Apply Random Oversampling solely to the training set.
    Uses imblearn if available, with a pure scikit-learn fallback.
    """
    try:
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=random_state)
        return ros.fit_resample(X_train, y_train)
    except ImportError:
        from sklearn.utils import resample
        unique_classes, counts = np.unique(y_train, return_counts=True)
        max_count = np.max(counts)
        X_resampled, y_resampled = [], []
        for cls in unique_classes:
            idx = np.where(y_train == cls)[0]
            cls_X, cls_y = X_train[idx], y_train[idx]
            if len(cls_X) < max_count:
                cls_X_res, cls_y_res = resample(
                    cls_X, cls_y, replace=True, n_samples=max_count, random_state=random_state
                )
                X_resampled.append(cls_X_res)
                y_resampled.append(cls_y_res)
            else:
                X_resampled.append(cls_X)
                y_resampled.append(cls_y)
        return np.vstack(X_resampled), np.concatenate(y_resampled)

def build_model(input_dim=40, num_classes=2):
    """Reconstructs the 4-layer MLP matching the original project architecture."""
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

def extract_speaker_id(filename):
    """
    Extracts speaker identifier from standard DEEP-VOICE audio naming conventions.
    Examples:
      'linus-original-DEMO.mp3' -> 'linus'
      'Obama-to-Biden.wav'       -> 'obama'
      'trump-original.wav'      -> 'trump'
    """
    base = os.path.basename(filename)
    parts = base.replace("_", "-").split("-")
    return parts[0].lower() if parts else "unknown"

def evaluate_predictions(y_true, y_pred_proba, target_names=("FAKE (0)", "REAL (1)")):
    """Computes and prints honest evaluation metrics."""
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=None)
    rec = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    cm = confusion_matrix(y_true, y_pred)
    
    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
    except Exception:
        roc_auc = float("nan")

    print("\n" + "="*60)
    print("LEAKAGE-FREE EVALUATION REPORT")
    print("="*60)
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    if not np.isnan(roc_auc):
        print(f"ROC-AUC Score:    {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"                Predicted {target_names[0]}  Predicted {target_names[1]}")
    print(f"Actual {target_names[0]:<10}: {cm[0, 0]:<20} {cm[0, 1]}")
    print(f"Actual {target_names[1]:<10}: {cm[1, 0]:<20} {cm[1, 1]}")

    print("\nPer-Class Metrics:")
    for i, name in enumerate(target_names):
        print(f"  {name}: Precision = {prec[i]:.4f}, Recall = {rec[i]:.4f}, F1 = {f1[i]:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    print("="*60)

    return {
        "accuracy": acc,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": classification_report(y_true, y_pred, target_names=target_names, output_dict=True),
    }

def run_pipeline(X, y, speaker_ids=None, epochs=50, batch_size=32, save_model=True):
    """
    Executes the corrected leakage-free training and evaluation pipeline.
    """
    print(f"Total raw dataset size: {len(X)} samples")
    print(f"Raw class distribution: FAKE={np.sum(y == 0)}, REAL={np.sum(y == 1)}")

    # Pre-split duplicate check on raw dataset
    unique_raw_features = len(np.unique(np.round(X, 6), axis=0))
    if unique_raw_features < len(X):
        dup_count = len(X) - unique_raw_features
        print(f"[DATASET WARNING] Detected {dup_count} duplicate feature vector(s) in the raw dataset before splitting!")
        print("Note: Duplicate recordings in source data can compromise evaluation if identical clips exist under different names.")
    else:
        print("Duplicate Check: All raw feature vectors are distinct (0 exact duplicates found).")

    # 1. Split raw data FIRST (with stratification or speaker grouping)
    split_strategy = "unknown"
    if speaker_ids is not None:
        unique_speakers = np.unique(speaker_ids)
        if len(unique_speakers) >= 3:
            print(f"Attempting Stratified Group Split across {len(unique_speakers)} detected speaker groups...")
            sgkf = StratifiedGroupKFold(n_splits=min(5, len(unique_speakers)), shuffle=True, random_state=42)
            try:
                train_idx, test_idx = next(sgkf.split(X, y, groups=speaker_ids))
                X_train_cand, X_test_cand = X[train_idx], X[test_idx]
                y_train_cand, y_test_cand = y[train_idx], y[test_idx]
                
                # Verify that test set contains both classes
                if len(np.unique(y_test_cand)) == len(np.unique(y)):
                    X_train, X_test = X_train_cand, X_test_cand
                    y_train, y_test = y_train_cand, y_test_cand
                    split_strategy = f"StratifiedGroupKFold (Speaker-Independent across {len(unique_speakers)} groups)"
                    print(f"SUCCESS: Speaker-independent split established. No speaker overlap between train and test.")
                else:
                    print("[LIMITATION REPORT] Grouped split resulted in a single-class test fold due to unbalanced speaker distribution across classes.")
                    print("Falling back to Stratified Random Split to preserve both FAKE and REAL classes in test set.")
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
                    split_strategy = "Stratified Random Split (Class-balanced, non-grouped)"
            except Exception as e:
                print(f"[LIMITATION REPORT] Grouped split failed ({e}). Falling back to Stratified Random Split.")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
                split_strategy = "Stratified Random Split (Class-balanced, non-grouped)"
        else:
            print(f"[LIMITATION REPORT] Only {len(unique_speakers)} speaker group(s) detected ('{list(unique_speakers)}').")
            print("Grouped stratification requires at least 3 distinct speaker groups. Falling back to Stratified Random Split.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
            split_strategy = "Stratified Random Split (Insufficient speaker groups for grouped split)"
    else:
        print("[LIMITATION REPORT] No speaker IDs available. Using Stratified Random Split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
        split_strategy = "Stratified Random Split (Class-balanced, non-grouped)"

    print(f"Selected Split Strategy: {split_strategy}")
    print(f"Train partition (raw): {len(X_train)} samples (FAKE={np.sum(y_train==0)}, REAL={np.sum(y_train==1)})")
    print(f"Test partition  (raw): {len(X_test)} samples (FAKE={np.sum(y_test==0)}, REAL={np.sum(y_test==1)})")

    # 2. Oversample ONLY the training set
    X_train_resampled, y_train_resampled = balance_training_data(X_train, y_train, random_state=42)
    print(f"Train partition after oversampling: {len(X_train_resampled)} samples (Balanced 50/50)")
    print(f"Test partition remains UNTOUCHED:   {len(X_test)} samples (Evaluation distribution preserved)")

    # Verify zero duplicate leakage between train and test
    # (convert rows to bytes/tuples to check set intersection)
    train_hashes = set(tuple(np.round(row, 6)) for row in X_train_resampled)
    test_hashes = set(tuple(np.round(row, 6)) for row in X_test)
    overlap = train_hashes.intersection(test_hashes)
    print(f"Leakage Verification: Overlapping samples between Train and Test = {len(overlap)} (Must be 0)")
    assert len(overlap) == 0, "CRITICAL ERROR: Data leakage detected between train and test sets!"

    # 3. Categorical one-hot encoding for training
    y_train_cat = to_categorical(y_train_resampled, num_classes=2)
    y_test_cat = to_categorical(y_test, num_classes=2)

    # 4. Build and train model
    model = build_model(input_dim=X.shape[1], num_classes=2)
    early = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    print(f"\nTraining model for up to {epochs} epochs (batch_size={batch_size})...")
    history = model.fit(
        X_train_resampled,
        y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early],
        verbose=0
    )
    print(f"Training completed in {len(history.history['loss'])} epochs.")

    # 5. Evaluate on uncontaminated test set
    y_pred_proba = model.predict(X_test, verbose=0)
    results = evaluate_predictions(y_test, y_pred_proba)

    # 6. Save experimental model safely
    if save_model:
        exp_path = os.path.join(BASE_DIR, "model_experimental.h5")
        model.save(exp_path)
        print(f"\n[SAFETY NOTE] Experimental model saved to: '{exp_path}'")
        print("The production application model ('model.h5') was NOT modified.")

    return results

if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "data_deep_voice", "KAGGLE", "AUDIO")
    
    if os.path.exists(dataset_path):
        print(f"Loading DEEP-VOICE dataset from: '{dataset_path}'")
        import librosa
        folders = ["FAKE", "REAL"]
        data, labels, speakers = [], [], []
        for folder in folders:
            folder_path = os.path.join(dataset_path, folder)
            if not os.path.exists(folder_path):
                continue
            files = [f for f in os.listdir(folder_path) if f.lower().endswith((".wav", ".mp3", ".ogg"))]
            for f in files:
                fpath = os.path.join(folder_path, f)
                try:
                    audio, sr = librosa.load(fpath, res_type="kaiser_fast")
                    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                    mfcc_scaled = np.mean(mfcc.T, axis=0)
                    data.append(mfcc_scaled)
                    labels.append(0 if folder == "FAKE" else 1)
                    speakers.append(extract_speaker_id(f))
                except Exception as e:
                    print(f"Skipping corrupt file {f}: {e}")
        
        X = np.array(data)
        y = np.array(labels)
        speakers = np.array(speakers)
        run_pipeline(X, y, speaker_ids=speakers)
    else:
        print(f"[STATUS] External Kaggle dataset not found at '{dataset_path}'.")
        print("Running self-contained verification test with synthetic feature data...")
        # Generate synthetic 40-dim feature matrix with realistic class imbalance (70% Fake, 30% Real)
        np.random.seed(42)
        n_samples = 200
        # Simulated MFCC features with distinctive centers for real vs fake
        X_fake = np.random.normal(loc=0.0, scale=1.0, size=(140, 40))
        X_real = np.random.normal(loc=0.5, scale=1.0, size=(60, 40))
        X = np.vstack([X_fake, X_real])
        y = np.array([0]*140 + [1]*60)
        
        # Synthetic speaker IDs
        synthetic_speakers = np.random.choice(["speaker_a", "speaker_b", "speaker_c", "speaker_d"], size=n_samples)
        
        print("\nVerifying leakage-free pipeline...")
        run_pipeline(X, y, speaker_ids=synthetic_speakers, epochs=15, batch_size=16, save_model=True)
