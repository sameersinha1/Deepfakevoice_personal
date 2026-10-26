

import os
import sys
import numpy as np
import joblib
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from model import extract_features, MODEL_DIR, MODEL_PATH, SCALER_PATH


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
REAL_DIR = os.path.join(DATA_DIR, "real")
FAKE_DIR = os.path.join(DATA_DIR, "fake")


def augment_audio(y, sr=16000):
    """Apply realistic room reflection, ambient noise, and mic filtering."""
    reverb_delay = int(sr * np.random.uniform(0.015, 0.035))
    decay = np.random.uniform(0.12, 0.25)
    y_aug = y.copy()
    if len(y) > reverb_delay:
        y_aug[reverb_delay:] += decay * y[:-reverb_delay]

    y_aug += np.random.normal(0, np.random.uniform(0.001, 0.005), len(y))

    high_cut = np.random.uniform(6800, 7800) / (sr / 2)
    b, a = butter(2, high_cut, btype='low')
    y_aug = lfilter(b, a, y_aug)

    return librosa.util.normalize(y_aug.astype(np.float32))


def load_dataset():
    X = []
    y = []
    uploads_temp = os.path.join(BASE_DIR, "uploads")
    os.makedirs(uploads_temp, exist_ok=True)

    real_files = [f for f in os.listdir(REAL_DIR) if f.endswith(('.wav', '.flac', '.mp3', '.ogg'))]
    fake_files = [f for f in os.listdir(FAKE_DIR) if f.endswith(('.wav', '.flac', '.mp3', '.ogg'))]

    print(f"\n[DIR] Processing {len(real_files)} Real human audio files...")
    for i, fname in enumerate(real_files):
        fpath = os.path.join(REAL_DIR, fname)
        feat = extract_features(fpath)
        if feat is not None:
            X.append(feat)
            y.append(1)

        # Augmented version
        try:
            audio, sr = librosa.load(fpath, sr=16000, mono=True)
            audio_aug = augment_audio(audio, sr=sr)
            tpath = os.path.join(uploads_temp, f"_temp_real_{i}.wav")
            sf.write(tpath, audio_aug, sr)
            feat_aug = extract_features(tpath)
            if feat_aug is not None:
                X.append(feat_aug)
                y.append(1)
            if os.path.exists(tpath):
                os.remove(tpath)
        except Exception:
            pass

        if (i + 1) % 25 == 0:
            print(f"   Processed {i + 1}/{len(real_files)} real files", flush=True)

    print(f"\n[DIR] Processing {len(fake_files)} Neural AI Fake audio files...")
    for i, fname in enumerate(fake_files):
        fpath = os.path.join(FAKE_DIR, fname)
        feat = extract_features(fpath)
        if feat is not None:
            X.append(feat)
            y.append(0)

        # Augmented version (re-recorded / speaker acoustic simulation)
        try:
            audio, sr = librosa.load(fpath, sr=16000, mono=True)
            audio_aug = augment_audio(audio, sr=sr)
            tpath = os.path.join(uploads_temp, f"_temp_fake_{i}.wav")
            sf.write(tpath, audio_aug, sr)
            feat_aug = extract_features(tpath)
            if feat_aug is not None:
                X.append(feat_aug)
                y.append(0)
            if os.path.exists(tpath):
                os.remove(tpath)
        except Exception:
            pass

        if (i + 1) % 25 == 0:
            print(f"   Processed {i + 1}/{len(fake_files)} fake files", flush=True)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train():
    print("=" * 60)
    print("[*] DEEPFAKE VOICE DETECTION -- TRAINING WITH LFCC & NEURAL VOICES")
    print("=" * 60, flush=True)

    X, y = load_dataset()

    print(f"\n[#] Dataset Summary:")
    print(f"   Total samples: {len(y)}")
    print(f"   Real samples:  {np.sum(y == 1)}")
    print(f"   Fake (Neural AI) samples: {np.sum(y == 0)}")
    print(f"   Feature vector: {X.shape[1]} dimensions", flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[+] Training Calibrated Random Forest Ensemble...", flush=True)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[*] Test Accuracy: {acc * 100:.1f}%")
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]), flush=True)

    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='accuracy')
    print(f"[#] 5-Fold Cross-Validation Accuracy: {cv_scores.mean() * 100:.1f}% (+/-{cv_scores.std() * 100:.1f}%)", flush=True)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"[S] Model saved: {MODEL_PATH}")
    print(f"[S] Scaler saved: {SCALER_PATH}")
    print("\n[OK] TRAINING COMPLETE!", flush=True)


if __name__ == "__main__":
    train()
