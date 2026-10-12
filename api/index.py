"""
Vercel Serverless Function Entry Point for DeepVoice Shield.
Exposes Flask WSGI application callable named 'app'.
"""

import os
import sys

# Configure python path to find backend modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")

for path in [BACKEND_DIR, CURRENT_DIR, ROOT_DIR]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

# Import the configured Flask app from backend
from app import app

# Vercel WSGI entry point
# 'app' is the standard WSGI callable expected by @vercel/python
