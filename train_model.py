"""
Speech Emotion Recognition — Model Training Script
===================================================
Supports two backends:
  1. Deep Learning (CNN-LSTM) via TensorFlow/Keras  [preferred]
  2. Gradient Boosting via scikit-learn             [fallback]

Dataset: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
Download: https://zenodo.org/record/1188976

Directory structure expected:
  data/
    Actor_01/
      03-01-01-01-01-01-01.wav
      ...
    Actor_02/
      ...

RAVDESS filename encoding:
  Modality-Vocal_Channel-Emotion-Emotional_Intensity-Statement-Repetition-Actor.wav
  Emotion codes: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry,
                 06=fearful, 07=disgust, 08=surprised
"""

import os
import glob
import pickle
import warnings
import numpy as np
import librosa
from tqdm import tqdm
warnings.filterwarnings('ignore')

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR   = "C:\\Users\\preet\\.cache\\kagglehub\\datasets\\dmitrybabko\\speech-emotion-recognition-en\\versions\\1\\"           # RAVDESS root
N_MFCC     = 40
SR         = 22050
BACKEND    = "both"           # "deep_learning" | "sklearn" | "both"

EMOTION_MAP = {
    "01": 0,  # neutral
    "02": 1,  # calm
    "03": 2,  # happy
    "04": 3,  # sad
    "05": 4,  # angry
    "06": 5,  # fearful
    "07": 6,  # disgust
    "08": 7,  # surprised
}

LABEL_NAMES = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


# ─── Feature Extraction ──────────────────────────────────────────────────────
def extract_features(file_path: str, n_mfcc: int = 40) -> np.ndarray | None:
    try:
        y, sr = librosa.load(file_path, sr=SR, mono=True, duration=10)
        if len(y) < 2048:
            return None
        features = []

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        stft = np.abs(librosa.stft(y))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))

        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        features.extend([np.mean(mel), np.std(mel)])

        contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
        features.extend(np.mean(contrast, axis=1))
        features.extend(np.std(contrast, axis=1))

        try:
            harmonic = librosa.effects.harmonic(y)
            tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
            features.extend(np.mean(tonnetz, axis=1))
            features.extend(np.std(tonnetz, axis=1))
        except Exception:
            features.extend([0.0] * 12)

        zcr = librosa.feature.zero_crossing_rate(y)
        features.extend([np.mean(zcr), np.std(zcr)])

        rms = librosa.feature.rms(y=y)
        features.extend([np.mean(rms), np.std(rms)])

        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.extend([np.mean(centroid), np.std(centroid)])

        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.extend([np.mean(bandwidth), np.std(bandwidth)])

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.extend([np.mean(rolloff), np.std(rolloff)])

        return np.array(features, dtype=np.float32)
    except Exception as e:
        print(f"  [WARN] {file_path}: {e}")
        return None


# ─── Dataset Loading ─────────────────────────────────────────────────────────
def load_ravdess(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse RAVDESS directory → (X, y)."""
    X, y = [], []
    wav_files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)

    if not wav_files:
        raise FileNotFoundError(
            f"No .wav files found in '{data_dir}'.\n"
            "Download RAVDESS from https://zenodo.org/record/1188976 "
            "and extract into the 'data/' folder."
        )

    print(f"Found {len(wav_files)} audio files. Extracting features…")
    for fp in tqdm(wav_files, desc="RAVDESS"):
        fname = os.path.basename(fp)
        parts = fname.replace(".wav", "").split("-")
        if len(parts) < 3:
            continue
        emotion_code = parts[2]
        if emotion_code not in EMOTION_MAP:
            continue
        feat = extract_features(fp, N_MFCC)
        if feat is not None:
            X.append(feat)
            y.append(EMOTION_MAP[emotion_code])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ─── Deep Learning Model ─────────────────────────────────────────────────────
def build_dl_model(input_dim: int, n_classes: int):
    """Build a CNN + Dense model suitable for tabular acoustic features."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.35),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.25),

        layers.Dense(n_classes, activation='softmax'),
    ], name="SER_DNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train_deep_learning(X_train, y_train, X_val, y_val):
    from tensorflow import keras
    import tensorflow as tf

    print("\n── Training Deep Neural Network ──")
    model = build_dl_model(X_train.shape[1], len(LABEL_NAMES))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint("ser_model.h5", save_best_only=True, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    val_acc = max(history.history['val_accuracy'])
    print(f"\n✓ Best val accuracy: {val_acc*100:.2f}%")
    return model, val_acc


# ─── Sklearn Model ────────────────────────────────────────────────────────────
def train_sklearn(X_train, y_train, X_val, y_val):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    print("\n── Training Ensemble (GB + RF + SVM) ──")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_v_s  = scaler.transform(X_val)

    gb  = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                     subsample=0.8, random_state=42)
    rf  = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    svm = SVC(probability=True, kernel='rbf', C=5, gamma='scale', random_state=42)

    ensemble = VotingClassifier(estimators=[('gb', gb), ('rf', rf), ('svm', svm)],
                                voting='soft', n_jobs=-1)
    ensemble.fit(X_tr_s, y_train)

    val_acc = accuracy_score(y_val, ensemble.predict(X_v_s))
    print(f"✓ Val accuracy: {val_acc*100:.2f}%")

    with open("ser_sklearn.pkl", "wb") as f:
        pickle.dump((ensemble, scaler), f)
    print("✓ Saved ser_sklearn.pkl")

    return ensemble, scaler, val_acc


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    print("=" * 60)
    print("  Speech Emotion Recognition — Model Training")
    print("=" * 60)

    X, y = load_ravdess(DATA_DIR)
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features, {len(set(y))} classes")
    print("Class distribution:", {LABEL_NAMES[i]: int((y == i).sum()) for i in range(len(LABEL_NAMES)) if (y == i).any()})

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}  |  Val: {len(X_val)}")

    if BACKEND in ("deep_learning", "both"):
        try:
            train_deep_learning(X_train, y_train, X_val, y_val)
        except ImportError:
            print("[WARN] TensorFlow not installed. Skipping deep learning.")

    if BACKEND in ("sklearn", "both"):
        train_sklearn(X_train, y_train, X_val, y_val)

    print("\n✅ Training complete. Run the app with:")
    print("   streamlit run app.py")
