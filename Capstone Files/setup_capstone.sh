#!/bin/bash

echo "Updating & upgrading system..."
sudo apt update && sudo apt full-upgrade -y

echo "Installing system dependencies..."
sudo apt install -y \
  git \
  python3 \
  python3-pip \
  python3-venv \
  libqtgui4 \
  libqt4-test \
  libv4l-dev \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev \
  libxvidcore-dev \
  libx264-dev \
  v4l-utils


echo "Cloning GitHub repo (Capstone-Stuff)..."
if [ ! -d "Capstone-Stuff" ]; then
  git clone https://github.com/Gabriel-M02/Capstone-Stuff.git
else
  echo "Repo already exists. Skipping clone."
fi

echo "Creating virtual environment..."
python3 -m venv capstone-venv

echo "Activating virtual environment..."
source capstone-venv/bin/activate

echo "Installing Python libraries individually..."

pip install --upgrade pip

# Core libraries
pip install opencv-python
pip install numpy
pip install psutil

# MediaPipe
pip install mediapipe

# Ultralytics + YOLOv8
pip install ultralytics==8.0.20

# PyTorch (CPU for Pi)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "All installations complete!"

echo ""
echo "To activate the environment and run your HUD:"
echo "source capstone-venv/bin/activate"
echo "python3 Capstone-Stuff/ThreatDetectionMK2.py"
