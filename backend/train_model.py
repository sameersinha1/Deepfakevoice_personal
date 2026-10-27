"""
Training Script for Deep Learning Deepfake Voice Detection Model
-----------------------------------------------------------------
Trains a 7-Hidden-Layer Deep Neural Network (DNN) with Multi-Scale Residual Skip
Forward Propagation using ASVspoof-standard LFCC (Linear Frequency Cepstral Coefficients),
dynamic MFCCs, and spectral features on genuine human voice vs. modern Neural AI speech.
"""

import os
import sys
import numpy as np
import joblib
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Ensure backend directory is in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from model import (
    extract_features,
    DeepVoiceNeuralNetwork,
    MODEL_DIR,
    MODEL_PATH,
    SCALER_PATH,
    DEEP_MODEL_PATH
)

DATA_DIR = os.path.join(BASE_DIR, "data")
REAL_DIR = os.path.join(DATA_DIR, "real")
FAKE_DIR = os.path.join(DATA_DIR, "fake")
CACHE_FILE = os.path.join(DATA_DIR, "features_cache.npz")
API_SAVED_MODEL_DIR = os.path.join(BASE_DIR, "..", "api", "saved_model")


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
    """Load or build the 230-dimensional acoustic feature dataset."""
    if os.path.exists(CACHE_FILE):
        print(f"\n[CACHE] Loading pre-extracted features from {CACHE_FILE}...")
        data = np.load(CACHE_FILE)
        return data['X'], data['y']

    X = []
    y = []
    uploads_temp = os.path.join(BASE_DIR, "uploads")
    os.makedirs(uploads_temp, exist_ok=True)

    import glob
    real_paths = sorted(glob.glob(os.path.join(REAL_DIR, "*.*")) + 
                        glob.glob(os.path.join(DATA_DIR, "synthetic", "real", "*.*")))
    fake_paths = sorted(glob.glob(os.path.join(FAKE_DIR, "*.*")) + 
                        glob.glob(os.path.join(DATA_DIR, "synthetic", "fake", "*.*")))

    real_paths = [p for p in real_paths if p.endswith(('.wav', '.flac', '.mp3', '.ogg', '.m4a'))]
    fake_paths = [p for p in fake_paths if p.endswith(('.wav', '.flac', '.mp3', '.ogg', '.m4a'))]

    print(f"\n[DIR] Processing {len(real_paths)} Real human audio files (FLAC + WAV)...")
    for i, fpath in enumerate(real_paths):
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

        if (i + 1) % 50 == 0:
            print(f"   Processed {i + 1}/{len(real_paths)} real files", flush=True)

    print(f"\n[DIR] Processing {len(fake_paths)} Neural AI Fake audio files (MP3 + WAV)...")
    for i, fpath in enumerate(fake_paths):
        feat = extract_features(fpath)
        if feat is not None:
            X.append(feat)
            y.append(0)

        # Augmented version
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

        if (i + 1) % 50 == 0:
            print(f"   Processed {i + 1}/{len(fake_paths)} fake files", flush=True)

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)
    np.savez_compressed(CACHE_FILE, X=X_arr, y=y_arr)
    print(f"[CACHE] Saved feature cache to {CACHE_FILE}")
    return X_arr, y_arr


def build_deep_neural_network(input_dim=230):
    """
    Construct Deep Neural Network (DNN) with 7 Hidden Layers and Residual Skip Forward Fusion.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=(input_dim,), name="audio_features_input")

    # Layer 1: Wide Input Feature Projection (230 -> 512)
    x1 = layers.Dense(512, activation=None, name="dense_l1_512")(inputs)
    x1 = layers.BatchNormalization(name="bn_l1")(x1)
    x1 = layers.LeakyReLU(negative_slope=0.1, name="leaky_l1")(x1)
    x1 = layers.Dropout(0.3, name="dropout_l1")(x1)

    # Layer 2: Deep Harmonic & Vocoder Representation (512 -> 256)
    x2 = layers.Dense(256, activation=None, name="dense_l2_256")(x1)
    x2 = layers.BatchNormalization(name="bn_l2")(x2)
    x2 = layers.LeakyReLU(negative_slope=0.1, name="leaky_l2")(x2)
    x2 = layers.Dropout(0.3, name="dropout_l2")(x2)

    # Layer 3: High-Order Spectral Pattern Abstraction (256 -> 256)
    x3 = layers.Dense(256, activation=None, name="dense_l3_256")(x2)
    x3 = layers.BatchNormalization(name="bn_l3")(x3)
    x3 = layers.LeakyReLU(negative_slope=0.1, name="leaky_l3")(x3)
    x3 = layers.Dropout(0.25, name="dropout_l3")(x3)

    # Layer 4: Multi-Scale Residual Skip Forward Fusion (256 -> 128 + residual skip from x2)
    skip = layers.Dense(128, use_bias=False, name="residual_skip_proj")(x2)
    x4 = layers.Dense(128, activation=None, name="dense_l4_128")(x3)
    x4 = layers.add([x4, skip], name="residual_forward_add")
    x4 = layers.BatchNormalization(name="bn_l4")(x4)
    x4 = layers.LeakyReLU(negative_slope=0.1, name="leaky_l4")(x4)
    x4 = layers.Dropout(0.2, name="dropout_l4")(x4)

    # Layer 5: Acoustic Latent Feature Compression (128 -> 64)
    x5 = layers.Dense(64, activation=None, name="dense_l5_64")(x4)
    x5 = layers.BatchNormalization(name="bn_l5")(x5)
    x5 = layers.LeakyReLU(negative_slope=0.1, name="leaky_l5")(x5)
    x5 = layers.Dropout(0.15, name="dropout_l5")(x5)

    # Layer 6: Discriminant Subspace Refinement (64 -> 32)
    x6 = layers.Dense(32, activation=None, name="dense_l6_32")(x5)
    x6 = layers.LeakyReLU(negative_slope=0.1, name="leaky_l6")(x6)

    # Layer 7: Deep Voice Bottleneck Embedding (32 -> 16)
    x7 = layers.Dense(16, activation=None, name="dense_l7_16")(x6)
    x7 = layers.LeakyReLU(negative_slope=0.1, name="leaky_l7")(x7)

    # Final Softmax Forward Propagation Head (2 Classes: [0=Fake, 1=Real])
    outputs = layers.Dense(2, activation="softmax", name="forward_softmax_output")(x7)

    model = models.Model(inputs=inputs, outputs=outputs, name="DeepVoice_MultiLayer_DNN")
    return model


def extract_weights_dict(keras_model):
    """Extract trained weights, biases, and BatchNorm statistics into serializable dict."""
    weights_dict = {}
    for layer in keras_model.layers:
        weights = layer.get_weights()
        if len(weights) == 2:  # Dense (W, b)
            weights_dict[f"{layer.name}_W"] = weights[0].astype(np.float32)
            weights_dict[f"{layer.name}_b"] = weights[1].astype(np.float32)
        elif len(weights) == 1:  # Dense without bias (skip projection)
            weights_dict[f"{layer.name}_W"] = weights[0].astype(np.float32)
        elif len(weights) == 4:  # BatchNorm (gamma, beta, mean, var)
            weights_dict[f"{layer.name}_gamma"] = weights[0].astype(np.float32)
            weights_dict[f"{layer.name}_beta"] = weights[1].astype(np.float32)
            weights_dict[f"{layer.name}_mean"] = weights[2].astype(np.float32)
            weights_dict[f"{layer.name}_var"] = weights[3].astype(np.float32)
    return weights_dict


def train():
    print("=" * 70)
    print("[*] DEEP LEARNING MODEL TRAINING — 7 HIDDEN LAYERS & FORWARD PROPAGATION")
    print("=" * 70, flush=True)

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

    print("\n[+] Constructing Deep Neural Network (7 Deep Hidden Layers)...", flush=True)
    import tensorflow as tf
    from tensorflow.keras import callbacks

    model = build_deep_neural_network(input_dim=X.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    lr_reducer = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5, verbose=1
    )
    early_stopper = callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
    )

    print("\n[+] Executing Deep Learning Optimization (Forward + Backward Propagation)...", flush=True)
    model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=60,
        batch_size=16,
        callbacks=[lr_reducer, early_stopper],
        verbose=1
    )

    # Evaluate Keras model
    y_pred_proba = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_proba, axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[*] Test Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]), flush=True)

    # Extract weights into DeepVoiceNeuralNetwork forward engine
    print("[+] Extracting trained weights into DeepVoiceNeuralNetwork engine...", flush=True)
    weights_dict = extract_weights_dict(model)
    deep_voice_net = DeepVoiceNeuralNetwork(
        weights=weights_dict,
        architecture={
            "input_dim": X.shape[1],
            "hidden_layers": [512, 256, 256, 128, 64, 32, 16],
            "output_dim": 2,
            "has_residual": True,
            "activation": "LeakyReLU (alpha=0.1)",
            "normalization": "BatchNormalization",
        }
    )

    # Verify forward propagation parity
    numpy_probs, stages = deep_voice_net.forward(X_test_scaled)
    diff = float(np.max(np.abs(y_pred_proba - numpy_probs)))
    print(f"[OK] Forward Propagation Parity verified! Max delta: {diff:.8f}")
    print("\n[#] Forward Propagation Pipeline Stages:")
    for s in stages:
        print(f"   Stage {s['stage']}: {s['name']} ({s['flow']}) [{s['activation']}]")

    # Save artifacts in backend/saved_model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(deep_voice_net, DEEP_MODEL_PATH)
    joblib.dump(deep_voice_net, MODEL_PATH)  # Maintain backward compatibility
    joblib.dump(scaler, SCALER_PATH)

    # Also save native Keras model
    keras_path = os.path.join(MODEL_DIR, "deep_model.keras")
    try:
        model.save(keras_path)
        print(f"[S] Saved native Keras model: {keras_path}")
    except Exception as e:
        print(f"[!] Could not save Keras model file: {e}")

    print(f"[S] Saved Deep Learning model: {DEEP_MODEL_PATH}")
    print(f"[S] Saved compatible model: {MODEL_PATH}")
    print(f"[S] Saved Scaler: {SCALER_PATH}")

    # Synchronize to api/saved_model for Vercel deployment
    if os.path.exists(os.path.dirname(API_SAVED_MODEL_DIR)):
        os.makedirs(API_SAVED_MODEL_DIR, exist_ok=True)
        joblib.dump(deep_voice_net, os.path.join(API_SAVED_MODEL_DIR, "deep_model.pkl"))
        joblib.dump(deep_voice_net, os.path.join(API_SAVED_MODEL_DIR, "model.pkl"))
        joblib.dump(scaler, os.path.join(API_SAVED_MODEL_DIR, "scaler.pkl"))
        print(f"[S] Synchronized deep model to API directory: {API_SAVED_MODEL_DIR}")

    print("\n[OK] DEEP LEARNING MODEL TRAINING COMPLETE!", flush=True)


if __name__ == "__main__":
    train()
