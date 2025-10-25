#!/bin/bash

set -e
echo "🚀 Starting EVST HUD Clean Setup Script"

# -------------------------------
# 1. System Update & Essentials
# -------------------------------
echo "🔧 Updating system packages..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y git wget build-essential zlib1g-dev libncurses5-dev \
  libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev curl \
  libsqlite3-dev libbz2-dev liblzma-dev

# -------------------------------
# 2. Install Python 3.9 from Source
# -------------------------------
echo "🐍 Installing Python 3.9.19 from source..."
cd /tmp
wget https://www.python.org/ftp/python/3.9.19/Python-3.9.19.tgz
tar -xf Python-3.9.19.tgz
cd Python-3.9.19

./configure --enable-optimizations
make -j$(nproc)
sudo make altinstall

# Confirm Python 3.9 is available
python3.9 --version

# -------------------------------
# 3. Create Virtual Environment
# -------------------------------
echo "🧪 Creating EVST virtual environment..."
cd ~
python3.9 -m venv evst-venv

# Activate environment
source ~/evst-venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# -------------------------------
# 4. Install Dependencies
# -------------------------------
echo "📦 Installing EVST Python dependencies..."
pip install numpy opencv-python psutil ultralytics

# ✅ MediaPipe install (stable with 3.9)
echo "🎯 Installing MediaPipe for Python 3.9..."
pip install mediapipe==0.10.9

# -------------------------------
# 5. Done
# -------------------------------
echo "✅ EVST environment setup complete!"
echo "To activate it later, run: source ~/evst-venv/bin/activate"
