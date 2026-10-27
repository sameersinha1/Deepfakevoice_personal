"""
DeepVoice Shield — Root Application Launcher
Allows running 'python app.py' directly from the project root folder.
"""

import os
import sys

# Add backend directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Change current working directory to backend so relative paths work seamlessly
os.chdir(BACKEND_DIR)

# Import and run the Flask app from backend
import app

if __name__ == "__main__":
    print("\n[*] Starting Deepfake Voice Detection Server from root...")
    print("=" * 60)
    if app.detector.load_model():
        print("[OK] Model ready for predictions")
    else:
        print("[!] Model not found -- run train_model.py first!")

    print("=" * 60)
    print("[>] OPEN THIS URL IN YOUR BROWSER:")
    print("    👉 http://localhost:5000")
    print("=" * 60)

    app.app.run(host="0.0.0.0", port=5000, debug=False)
