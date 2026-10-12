"""
Deepfake Voice Detection Model
-------------------------------
State-of-the-art feature extraction for detecting modern Neural AI Voice Deepfakes:
- LFCCs + Delta + Delta-Delta (Linear Frequency Cepstral Coefficients - ASVspoof Standard)
- MFCCs + Delta + Delta-Delta (Mel-Frequency Cepstral Coefficients)
- Spectral Contrast (sub-band energy peak/valley ratio)
- Spectral Flatness & Rolloff (high-frequency neural vocoder energy distribution)
- Spectral Centroid & Bandwidth (vocal tract resonance dispersion)
- Zero Crossing Rate & RMS Energy Dynamics
- Harmonic Chroma Features

Classifies audio using a calibrated Random Forest Ensemble.
"""

import os
import numpy as np
import librosa
import joblib
from scipy.fftpack import dct
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# Path to saved model artifacts
def _find_model_paths():
    base_dirs = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend", "saved_model"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "api", "saved_model"),
        os.path.join(os.getcwd(), "backend", "saved_model"),
        os.path.join(os.getcwd(), "api", "saved_model"),
        os.path.join(os.getcwd(), "saved_model"),
    ]
    for d in base_dirs:
        m_p = os.path.join(d, "model.pkl")
        s_p = os.path.join(d, "scaler.pkl")
        if os.path.exists(m_p) and os.path.exists(s_p):
            return m_p, s_p
    # Default fallback
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model")
    return os.path.join(default_dir, "model.pkl"), os.path.join(default_dir, "scaler.pkl")

MODEL_PATH, SCALER_PATH = _find_model_paths()



def compute_lfcc(y, sr=16000, n_lfcc=20, n_fft=512, hop_length=256):
    """
    Compute Linear Frequency Cepstral Coefficients (LFCC) + Delta + Delta-Delta.
    LFCCs use linearly spaced filter banks to expose neural vocoder high-frequency artifacts.
    """
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    n_filters = 40
    linear_filters = np.zeros((n_filters, 1 + n_fft // 2), dtype=np.float32)
    bank_points = np.linspace(0, 1 + n_fft // 2, n_filters + 2, dtype=int)
    for i in range(n_filters):
        start, center, end = bank_points[i], bank_points[i + 1], bank_points[i + 2]
        if center > start:
            linear_filters[i, start:center] = np.linspace(0, 1, center - start)
        if end > center:
            linear_filters[i, center:end] = np.linspace(1, 0, end - center)

    fb_energies = np.dot(linear_filters, stft) + 1e-8
    log_fb = np.log(fb_energies)
    lfcc = dct(log_fb, type=2, axis=0, norm='ortho')[:n_lfcc]
    delta_lfcc = librosa.feature.delta(lfcc)
    delta2_lfcc = librosa.feature.delta(lfcc, order=2)

    feats = []
    for m in [lfcc, delta_lfcc, delta2_lfcc]:
        feats.extend(np.mean(m, axis=1).tolist())
        feats.extend(np.std(m, axis=1).tolist())
    return feats


def extract_features(file_path, sr=16000):
    """
    Extract comprehensive 230-dimensional deepfake & neural vocoder detection features.
    """
    try:
        y, sr = librosa.load(file_path, sr=sr, mono=True)

        if len(y) < sr:
            y = np.pad(y, (0, sr - len(y)), mode='constant')

        y = librosa.util.normalize(y)
        features = []

        # 1. LFCCs (Linear Frequency Cepstral Coefficients) = 120 features (ASVspoof standard)
        lfcc_feats = compute_lfcc(y, sr=sr, n_lfcc=20)
        features.extend(lfcc_feats)

        # 2. MFCCs + Delta + Delta-Delta = 60 features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        for m in [mfcc, delta_mfcc, delta2_mfcc]:
            features.extend(np.mean(m, axis=1).tolist())
            features.extend(np.std(m, axis=1).tolist())

        # 3. Spectral Features = 22 features
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        s_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
        s_flat = librosa.feature.spectral_flatness(y=y)
        features.extend([float(np.mean(sc)), float(np.std(sc))])
        features.extend([float(np.mean(sb)), float(np.std(sb))])
        features.extend([float(np.mean(s_roll)), float(np.std(s_roll))])
        features.extend([float(np.mean(s_flat)), float(np.std(s_flat))])

        s_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features.extend(np.mean(s_contrast, axis=1).tolist())
        features.extend(np.std(s_contrast, axis=1).tolist())

        # 4. Zero Crossing Rate & RMS Energy = 4 features
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        features.extend([float(np.mean(zcr)), float(np.std(zcr))])
        features.extend([float(np.mean(rms)), float(np.std(rms))])

        # 5. Chroma Harmonics = 24 features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1).tolist())
        features.extend(np.std(chroma, axis=1).tolist())

        return np.array(features, dtype=np.float32)

    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None


def get_feature_names():
    """Return human-readable names for each extracted feature group."""
    return [
        {"name": "LFCC (Vocoder Artifacts)", "description": "Linear Frequency Cepstral Coefficients revealing neural vocoder artifacts across full spectrum", "count": 120},
        {"name": "MFCCs (Vocal Dynamics)", "description": "Mel-Frequency Cepstral Coefficients with velocity & acceleration dynamics", "count": 60},
        {"name": "Spectral Centroid", "description": "Center of spectral mass and acoustic brightness distribution", "count": 2},
        {"name": "Spectral Bandwidth", "description": "Frequency spread and resonance dispersion", "count": 2},
        {"name": "Spectral Rolloff", "description": "High-frequency cutoff and harmonic energy decay profile", "count": 2},
        {"name": "Spectral Flatness", "description": "Noise-like vs tonal harmonic structure of the voice", "count": 2},
        {"name": "Spectral Contrast", "description": "Peak-to-valley ratio across 7 frequency sub-bands", "count": 14},
        {"name": "Zero Crossing Rate", "description": "Temporal sign-change rate separating voiced from unvoiced phones", "count": 2},
        {"name": "RMS Energy Dynamics", "description": "Syllabic stress and loudness dynamics over time", "count": 2},
        {"name": "Chroma Harmonics", "description": "Pitch class distribution and musical harmonic resonance", "count": 24},
    ]


class DeepfakeDetector:
    """
    Deepfake voice detection model using calibrated Random Forest classifier.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_loaded = False

    def load_model(self):
        """Load the pre-trained model and scaler from disk."""
        try:
            m_path, s_path = _find_model_paths()
            if os.path.exists(m_path) and os.path.exists(s_path):
                self.model = joblib.load(m_path)
                self.scaler = joblib.load(s_path)
                self.is_loaded = True
                print(f"[OK] Model loaded successfully from {m_path}!")
                return True
            else:
                print(f"[!] No saved model found at {m_path}. Please check saved_model directory.")
                return False
        except Exception as e:
            print(f"[X] Error loading model: {e}")
            return False

    def predict(self, file_path):
        """
        Predict whether an audio file contains real or deepfake voice.
        """
        if not self.is_loaded:
            if not self.load_model():
                return {
                    "error": "Model not loaded. Please train the model first.",
                    "label": "Unknown",
                    "confidence": 0
                }

        features = extract_features(file_path)
        if features is None:
            return {
                "error": "Could not extract features from audio file.",
                "label": "Unknown",
                "confidence": 0
            }

        features_scaled = self.scaler.transform(features.reshape(1, -1))
        proba = self.model.predict_proba(features_scaled)[0]
        prediction = self.model.predict(features_scaled)[0]

        label = "Real" if prediction == 1 else "Fake"
        confidence = float(max(proba) * 100)

        # Feature importances
        importances = self.model.feature_importances_
        feature_info = get_feature_names()
        feature_scores = []
        idx = 0
        for info in feature_info:
            group_importance = float(np.sum(importances[idx:idx + info["count"]]))
            feature_scores.append({
                "name": info["name"],
                "description": info["description"],
                "importance": round(group_importance * 100, 1)
            })
            idx += info["count"]

        feature_scores.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "label": label,
            "confidence": round(confidence, 1),
            "real_probability": round(float(proba[1]) * 100, 1) if len(proba) > 1 else 0,
            "fake_probability": round(float(proba[0]) * 100, 1) if len(proba) > 1 else 0,
            "features_analyzed": feature_scores[:6],
        }


# Singleton instance
detector = DeepfakeDetector()
