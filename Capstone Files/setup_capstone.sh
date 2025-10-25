#!/bin/bash

echo "[1/7] Updating & upgrading system..."
sudo apt update && sudo apt full-upgrade -y

echo "[2/7] Installing system dependencies..."
sudo apt install -y \
  git \
  libv4l-dev \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev \
  libxvidcore-dev \
  libx264-dev \
  v4l-utils \
  python3.9 \
  python3.9-venv \
  python3.9-distutils \
  libpython3.9-dev \
  wget \
  curl

echo "[3/7] Cloning GitHub repo (Capstone-Stuff)..."
if [ ! -d "Capstone-Stuff" ]; then
  git clone https://github.com/Gabriel-M02/Capstone-Stuff.git
else
  echo "Repo already exists. Skipping clone."
fi

echo "[4/7] Creating Python 3.9 virtual environment..."
/usr/bin/python3.9 -m venv capstone-venv

echo "[5/7] Activating virtual environment..."
source capstone-venv/bin/activate

echo "[6/7] Upgrading pip and installing Python libraries..."
python -m ensurepip
pip install --upgrade pip

# Required libraries for ThreatDetectionMK2.py
pip install opencv-python numpy psutil mediapipe
pip install ultralytics==8.0.20
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "[7/7] All installations complete!"

echo ""
echo "To activate the environment and run your HUD:"
echo "source capstone-venv/bin/activate"
echo "python3 Capstone-Stuff/ThreatDetectionMK3.py"
