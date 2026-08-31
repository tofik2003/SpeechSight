#!/data/data/com.termux/files/usr/bin/bash
# ==============================================================================
# SpeechSight AI - On-Device Android Server Setup Script via Termux
# Enables running the FastAPI server and local AVSR inference directly on Android.
# ==============================================================================

set -e

echo "=== Setting up SpeechSight on Android (Termux) ==="

# 1. Update Termux packages
echo "[1/4] Updating package repository..."
pkg update -y && pkg upgrade -y

# 2. Install Python, clang, ffmpeg, libjpeg, libpng, git
echo "[2/4] Installing core dependencies (Python, Clang, OpenCV deps, FFmpeg)..."
pkg install -y python python-pip clang libjpeg-turbo libpng ffmpeg git

# 3. Install Python requirements
echo "[3/4] Installing Python machine learning and server packages..."
pip install --upgrade pip setuptools wheel
pip install fastapi uvicorn pydantic numpy scipy opencv-python-headless pillow python-multipart httpx

# 4. Launch SpeechSight FastAPI Server on localhost:8000
echo "[4/4] Starting SpeechSight Local AVSR Server on 0.0.0.0:8000..."
echo "Open the SpeechSight Android App and connect to http://127.0.0.1:8000"

python -m speechsight.server.app --host 0.0.0.0 --port 8000
