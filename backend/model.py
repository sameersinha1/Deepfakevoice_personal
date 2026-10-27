"""
Deepfake Voice Detection — Deep Learning Neural Network
--------------------------------------------------------
State-of-the-art Deep Neural Network (DNN) with Multi-Stage Forward Propagation
and 7 Deep Hidden Layers for detecting modern AI voice clones and deepfakes:

Acoustic Feature Vector (230 Dimensions):
- LFCCs + Delta + Delta-Delta (Linear Frequency Cepstral Coefficients - ASVspoof Standard)
- MFCCs + Delta + Delta-Delta (Mel-Frequency Cepstral Coefficients)
- Spectral Contrast (sub-band energy peak/valley ratio across 7 frequency bands)
- Spectral Flatness & Rolloff (high-frequency neural vocoder energy distribution)
- Spectral Centroid & Bandwidth (vocal tract resonance dispersion)
- Zero Crossing Rate & RMS Energy Dynamics
- Harmonic Chroma Features

Deep Neural Network Architecture:
- Input Layer: 230 acoustic dimensions
- Hidden Layer 1: 512 neurons (Dense + Batch Normalization + LeakyReLU + Dropout)
- Hidden Layer 2: 256 neurons (Dense + Batch Normalization + LeakyReLU + Dropout)
- Hidden Layer 3: 256 neurons (Dense + Batch Normalization + LeakyReLU + Dropout)
- Hidden Layer 4: 128 neurons (Dense + Multi-Scale Residual Skip Forward Fusion + BatchNorm + LeakyReLU)
- Hidden Layer 5: 64 neurons (Dense + Batch Normalization + LeakyReLU + Dropout)
- Hidden Layer 6: 32 neurons (Dense + LeakyReLU)
- Hidden Layer 7: 16 neurons (Dense + LeakyReLU Voice Bottleneck Embedding)
- Output Head: 2 neurons (Dense + Softmax Forward Normalization: [P(Fake), P(Real)])
"""

import os
import sys
import joblib
import numpy as np
import librosa
from scipy.fftpack import dct

# Ensure current module directory is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# Register cross-import module aliases for pickle serialization resilience
if "model" not in sys.modules and __name__ != "model":
    try:
        sys.modules["model"] = sys.modules.get(__name__)
    except Exception:
        pass
if "backend.model" not in sys.modules and __name__ != "backend.model":
    try:
        sys.modules["backend.model"] = sys.modules.get(__name__)
    except Exception:
        pass


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
            return d, m_p, s_p
    # Default fallback
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model")
    return default_dir, os.path.join(default_dir, "model.pkl"), os.path.join(default_dir, "scaler.pkl")


MODEL_DIR, MODEL_PATH, SCALER_PATH = _find_model_paths()
DEEP_MODEL_PATH = os.path.join(MODEL_DIR, "deep_model.pkl")


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

        # Voice Activity Detection (VAD) / Silence Gate
        peak_amp = float(np.max(np.abs(y)))
        rms_val = float(np.sqrt(np.mean(y ** 2)))
        if peak_amp < 0.005 or rms_val < 0.001:
            return None

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


class DeepVoiceNeuralNetwork:
    """
    Deep Neural Network with Multi-Stage Forward Propagation for Deepfake Voice Classification.

    Features:
    - 7 Deep Hidden Layers with non-linear LeakyReLU gating and Batch Normalization.
    - Multi-Scale Residual Skip Forward Fusion connecting intermediate representation to layer 4.
    - Vectorized sub-millisecond NumPy forward execution for 100% serverless/edge compatibility.
    - Full layer-wise forward activation tracking and forward propagation diagnostics.
    """

    def __init__(self, weights=None, architecture=None):
        self.weights = weights or {}
        self.architecture = architecture or {
            "input_dim": 230,
            "hidden_layers": [512, 256, 256, 128, 64, 32, 16],
            "output_dim": 2,
            "has_residual": True,
            "activation": "LeakyReLU (alpha=0.1)",
            "normalization": "BatchNormalization",
        }
        self.feature_importances_ = self._compute_base_feature_importances()

    def _compute_base_feature_importances(self):
        """Compute base input feature salience from layer 1 weights magnitude."""
        w1_key = "dense_l1_512_W"
        if w1_key in self.weights:
            w1 = self.weights[w1_key]
            salience = np.mean(np.abs(w1), axis=1)
            total = np.sum(salience)
            return (salience / (total + 1e-8)).astype(np.float32)
        return np.ones(230, dtype=np.float32) / 230.0

    @staticmethod
    def _leaky_relu(z, alpha=0.1):
        return np.where(z > 0, z, alpha * z)

    @staticmethod
    def _softmax(z):
        z_safe = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z_safe)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def _batch_norm(self, z, layer_prefix, epsilon=1e-3):
        gamma = self.weights.get(f"{layer_prefix}_gamma", 1.0)
        beta = self.weights.get(f"{layer_prefix}_beta", 0.0)
        mean = self.weights.get(f"{layer_prefix}_mean", 0.0)
        var = self.weights.get(f"{layer_prefix}_var", 1.0)
        inv_std = 1.0 / np.sqrt(var + epsilon)
        return gamma * (z - mean) * inv_std + beta

    def forward(self, x, return_activations=False):
        """
        Multi-Stage Deep Forward Propagation:
        Passes input acoustic features forward through 7 hidden layers with residual connection.

        Forward Propagation Stages:
          Stage 1: Input Feature Projection (230 -> 512 neurons) + BatchNorm + LeakyReLU
          Stage 2: Acoustic Representation (512 -> 256 neurons) + BatchNorm + LeakyReLU
          Stage 3: Spectral Pattern Abstraction (256 -> 256 neurons) + BatchNorm + LeakyReLU
          Stage 4: Residual Forward Fusion (256 -> 128 neurons + skip connection) + BatchNorm + LeakyReLU
          Stage 5: Latent Feature Compression (128 -> 64 neurons) + BatchNorm + LeakyReLU
          Stage 6: Subspace Refinement (64 -> 32 neurons) + LeakyReLU
          Stage 7: Voice Bottleneck Latent Embedding (32 -> 16 neurons) + LeakyReLU
          Stage 8: Softmax Classification Head (16 -> 2 logits -> Softmax Normalization)
        """
        x_in = np.asarray(x, dtype=np.float32)
        if x_in.ndim == 1:
            x_in = x_in.reshape(1, -1)

        activations = {}

        # Stage 1: Input -> Hidden L1 (512)
        z1 = np.dot(x_in, self.weights["dense_l1_512_W"]) + self.weights["dense_l1_512_b"]
        z1_bn = self._batch_norm(z1, "bn_l1")
        a1 = self._leaky_relu(z1_bn)
        activations["stage_1_dense_512"] = a1

        # Stage 2: Hidden L1 -> Hidden L2 (256)
        z2 = np.dot(a1, self.weights["dense_l2_256_W"]) + self.weights["dense_l2_256_b"]
        z2_bn = self._batch_norm(z2, "bn_l2")
        a2 = self._leaky_relu(z2_bn)
        activations["stage_2_dense_256"] = a2

        # Stage 3: Hidden L2 -> Hidden L3 (256)
        z3 = np.dot(a2, self.weights["dense_l3_256_W"]) + self.weights["dense_l3_256_b"]
        z3_bn = self._batch_norm(z3, "bn_l3")
        a3 = self._leaky_relu(z3_bn)
        activations["stage_3_dense_256"] = a3

        # Stage 4: Multi-Scale Residual Forward Fusion (256 -> 128 + residual skip projection from a2)
        skip = np.dot(a2, self.weights["residual_skip_proj_W"])
        z4 = np.dot(a3, self.weights["dense_l4_128_W"]) + self.weights["dense_l4_128_b"]
        z4_res = z4 + skip
        z4_bn = self._batch_norm(z4_res, "bn_l4")
        a4 = self._leaky_relu(z4_bn)
        activations["stage_4_dense_128_res"] = a4

        # Stage 5: Hidden L4 -> Hidden L5 (64)
        z5 = np.dot(a4, self.weights["dense_l5_64_W"]) + self.weights["dense_l5_64_b"]
        z5_bn = self._batch_norm(z5, "bn_l5")
        a5 = self._leaky_relu(z5_bn)
        activations["stage_5_dense_64"] = a5

        # Stage 6: Hidden L5 -> Hidden L6 (32)
        z6 = np.dot(a5, self.weights["dense_l6_32_W"]) + self.weights["dense_l6_32_b"]
        a6 = self._leaky_relu(z6)
        activations["stage_6_dense_32"] = a6

        # Stage 7: Hidden L6 -> Hidden L7 Voice Bottleneck (16)
        z7 = np.dot(a6, self.weights["dense_l7_16_W"]) + self.weights["dense_l7_16_b"]
        a7 = self._leaky_relu(z7)
        activations["stage_7_dense_16"] = a7

        # Stage 8: Output Head (16 -> 2 logits -> Softmax Forward Pass)
        logits = np.dot(a7, self.weights["forward_softmax_output_W"]) + self.weights["forward_softmax_output_b"]
        probabilities = self._softmax(logits)
        activations["stage_8_softmax"] = probabilities

        stages_meta = [
            {"stage": 1, "name": "Input Acoustic Projection", "neurons": 512, "activation": "LeakyReLU + BatchNorm", "flow": "230 -> 512"},
            {"stage": 2, "name": "Deep Harmonic Representation", "neurons": 256, "activation": "LeakyReLU + BatchNorm", "flow": "512 -> 256"},
            {"stage": 3, "name": "Spectral Pattern Abstraction", "neurons": 256, "activation": "LeakyReLU + BatchNorm", "flow": "256 -> 256"},
            {"stage": 4, "name": "Residual Skip Forward Fusion", "neurons": 128, "activation": "Residual Add + BatchNorm", "flow": "256 -> 128 (with Skip)"},
            {"stage": 5, "name": "Latent Feature Compression", "neurons": 64, "activation": "LeakyReLU + BatchNorm", "flow": "128 -> 64"},
            {"stage": 6, "name": "Discriminant Subspace Refinement", "neurons": 32, "activation": "LeakyReLU", "flow": "64 -> 32"},
            {"stage": 7, "name": "Voice Bottleneck Embedding", "neurons": 16, "activation": "LeakyReLU", "flow": "32 -> 16"},
            {"stage": 8, "name": "Softmax Forward Probability Head", "neurons": 2, "activation": "Softmax Normalization", "flow": "16 -> 2"},
        ]

        if return_activations:
            return probabilities, activations, stages_meta
        return probabilities, stages_meta

    def predict_proba(self, x):
        """Scikit-learn compatible predict_proba method."""
        probs, _ = self.forward(x)
        return probs

    def predict(self, x):
        """Scikit-learn compatible predict method."""
        probs, _ = self.forward(x)
        return np.argmax(probs, axis=-1)

    def compute_feature_attributions(self, x):
        """
        Compute input feature attribution using weight magnitude and input feature excitation.
        """
        x_flat = np.abs(np.asarray(x, dtype=np.float32).flatten())
        w1 = np.abs(self.weights.get("dense_l1_512_W", np.ones((230, 512))))
        salience = np.mean(w1, axis=1) * (x_flat + 1e-4)

        feature_info = get_feature_names()
        feature_scores = []
        idx = 0
        total_salience = np.sum(salience) + 1e-8

        for info in feature_info:
            group_salience = float(np.sum(salience[idx:idx + info["count"]]))
            group_pct = float((group_salience / total_salience) * 100.0)
            feature_scores.append({
                "name": str(info["name"]),
                "description": str(info["description"]),
                "importance": round(float(group_pct), 1)
            })
            idx += info["count"]

        feature_scores.sort(key=lambda item: item["importance"], reverse=True)
        return feature_scores


class DeepfakeDetector:
    """
    Deepfake voice detection detector using 7-Hidden-Layer Deep Neural Network (DNN)
    with Multi-Stage Forward Propagation.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_loaded = False

    def load_model(self):
        """Load the pre-trained Deep Neural Network model and scaler from disk."""
        try:
            d_path, m_path, s_path = _find_model_paths()
            deep_p = os.path.join(d_path, "deep_model.pkl")

            # Try loading deep_model.pkl first, or fallback to model.pkl
            target_model_file = deep_p if os.path.exists(deep_p) else m_path

            if os.path.exists(target_model_file) and os.path.exists(s_path):
                loaded_obj = joblib.load(target_model_file)
                self.scaler = joblib.load(s_path)

                if isinstance(loaded_obj, DeepVoiceNeuralNetwork):
                    self.model = loaded_obj
                elif isinstance(loaded_obj, dict) and "weights" in loaded_obj:
                    self.model = DeepVoiceNeuralNetwork(
                        weights=loaded_obj["weights"],
                        architecture=loaded_obj.get("architecture")
                    )
                else:
                    # Legacy fallback
                    self.model = loaded_obj

                self.is_loaded = True
                print(f"[OK] Deep Learning Model loaded successfully from {target_model_file}!")
                return True
            else:
                print(f"[!] No saved model found at {target_model_file}. Please run train_model.py.")
                return False
        except Exception as e:
            print(f"[X] Error loading model: {e}")
            return False

    def predict(self, file_path):
        """
        Predict whether an audio file contains real or deepfake voice
        using multi-stage forward propagation through 7 hidden layers.
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
                "error": "No audible voice activity detected. The audio is silent or too quiet. Please speak into the microphone.",
                "label": "Silence / No Speech",
                "confidence": 0,
                "real_probability": 0,
                "fake_probability": 0,
                "features_analyzed": [],
                "forward_propagation_stages": [],
                "deep_learning_model": {"architecture": "Deep Neural Network (DNN)"}
            }

        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Multi-stage Forward Propagation
        if isinstance(self.model, DeepVoiceNeuralNetwork):
            probs, stages = self.model.forward(features_scaled)
            proba = probs[0]
            feature_scores = self.model.compute_feature_attributions(features_scaled[0])
            model_info = {
                "architecture": "Deep Neural Network (DNN)",
                "hidden_layers": 7,
                "forward_propagation_stages": len(stages),
                "total_parameters": 396818
            }
        else:
            proba = self.model.predict_proba(features_scaled)[0]
            feature_scores = []
            stages = []
            model_info = {"architecture": "Ensemble Classifier"}

        fake_prob = round(float(proba[0]) * 100, 1) if len(proba) > 1 else 0
        real_prob = round(float(proba[1]) * 100, 1) if len(proba) > 1 else 0

        # Calibrated anti-spoofing decision threshold:
        # If Fake probability is >= 40.0%, classify as Fake (protects against subtle/compressed AI voices)
        is_fake = (fake_prob >= 40.0)
        label = "Fake" if is_fake else "Real"
        confidence = fake_prob if is_fake else real_prob

        return {
            "label": label,
            "confidence": round(confidence, 1),
            "real_probability": real_prob,
            "fake_probability": fake_prob,
            "features_analyzed": feature_scores[:6] if feature_scores else [],
            "forward_propagation_stages": stages,
            "deep_learning_model": model_info,
        }


# Singleton instance
detector = DeepfakeDetector()
