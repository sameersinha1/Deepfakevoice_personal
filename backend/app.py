"""
Deepfake Voice Detection — Flask API
--------------------------------------
REST API that accepts audio recordings and returns
real/fake predictions with confidence scores.
"""

import os
import uuid
import tempfile
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from model import detector

# Pre-load model on startup/import if present
try:
    detector.load_model()
except Exception as _e:
    print(f"Initial model load warning: {_e}")


app = Flask(__name__, static_folder=None)
CORS(app)

# Upload config - use temp directory so it works on read-only environments (Vercel Serverless / AWS Lambda)
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "deepvoice_uploads")
try:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
except Exception:
    pass

# Serve frontend
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
PUBLIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "public")


@app.route("/")
def serve_frontend():
    """Serve the frontend index.html."""
    if os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
        return send_from_directory(FRONTEND_DIR, "index.html")
    if os.path.exists(os.path.join(PUBLIC_DIR, "index.html")):
        return send_from_directory(PUBLIC_DIR, "index.html")
    return jsonify({"status": "DeepVoice API is running", "model_loaded": detector.is_loaded})


@app.route("/<path:path>")
def serve_static(path):
    """Serve frontend static files."""
    if os.path.exists(os.path.join(FRONTEND_DIR, path)):
        return send_from_directory(FRONTEND_DIR, path)
    if os.path.exists(os.path.join(PUBLIC_DIR, path)):
        return send_from_directory(PUBLIC_DIR, path)
    return jsonify({"error": "File not found"}), 404


@app.route("/api/health", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    if not detector.is_loaded:
        detector.load_model()
    return jsonify({
        "status": "ok",
        "model_loaded": detector.is_loaded,
        "model_architecture": "Deep Neural Network (DNN)",
        "hidden_layers": 7,
        "forward_propagation_stages": 8,
        "total_parameters": 396818,
        "message": "Deepfake Voice Detection Deep Learning API is running"
    })


@app.route("/api/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict whether uploaded audio is real or deepfake.
    
    Expects: multipart/form-data with 'audio' file field
    Returns: JSON with prediction label, confidence, and feature analysis
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]

    if audio_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save to temp file with unique name
    ext = os.path.splitext(audio_file.filename)[1] or ".webm"
    temp_filename = f"{uuid.uuid4().hex}{ext}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    try:
        audio_file.save(temp_path)

        # If the file is webm, convert to wav using pydub
        wav_path = temp_path
        if ext.lower() in [".webm", ".ogg", ".mp4", ".m4a"]:
            try:
                from pydub import AudioSegment
                audio_segment = AudioSegment.from_file(temp_path)
                wav_path = temp_path.replace(ext, ".wav")
                audio_segment.export(wav_path, format="wav")
            except Exception as conv_err:
                print(f"Warning: Could not convert {ext} to wav: {conv_err}")
                # Try to use the file as-is (librosa may handle it)
                wav_path = temp_path

        # Run prediction
        result = detector.predict(wav_path)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    finally:
        # Clean up temp files
        for path in [temp_path, temp_path.replace(ext, ".wav")]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


if __name__ == "__main__":
    print("\n[*] Starting Deepfake Voice Detection Server...")
    print("=" * 50)

    # Load the model on startup
    if detector.load_model():
        print("[OK] Model ready for predictions")
    else:
        print("[!] Model not found -- run train_model.py first!")
        print("   python train_model.py")

    print("=" * 60)
    print("[>] OPEN THIS URL IN YOUR BROWSER:")
    print("    👉 http://localhost:5000")
    print("    (Note: Browsers block microphone recording on IP addresses like 192.168.x.x)")
    print("    (Always use http://localhost:5000 to allow microphone permissions)")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=False)
