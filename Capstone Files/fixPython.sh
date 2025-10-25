#!/bin/bash

echo "=============================="
echo "EVST CLEAN PYTHON 3.9.13 ENV SETUP"
echo "=============================="

# ---- 0. Cleanup (optional) ----
echo "Removing old virtual environment (if it exists)..."
rm -rf capstone-venv

# ---- 1. Install build dependencies ----
echo "Installing system dependencies for building Python..."
sudo apt update && sudo apt install -y \
  wget build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
  libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev \
  libbz2-dev liblzma-dev uuid-dev libtk8.6 libtk8.6-dev tk-dev

# ---- 2. Download & Compile Python 3.9.13 ----
echo "Downloading Python 3.9.13..."
cd /tmp
wget https://www.python.org/ftp/python/3.9.13/Python-3.9.13.tgz
tar -xf Python-3.9.13.tgz
cd Python-3.9.13

echo "Configuring build..."
./configure --enable-optimizations --with-ensurepip=install

echo "Building Python (this may take a while)..."
make -j$(nproc)

echo "Installing Python 3.9.13..."
sudo make altinstall  # Use altinstall to avoid replacing system python

# ---- 3. Verify Python 3.9 ----
PYTHON_PATH=$(which python3.9)
echo "Python 3.9 installed at: $PYTHON_PATH"
$PYTHON_PATH --version

# ---- 4. Create and activate virtual environment ----
echo "Creating Python 3.9 virtual environment..."
cd ~
$PYTHON_PATH -m venv capstone-venv

echo "Activating virtual environment..."
source ~/capstone-venv/bin/activate

# ---- 5. Upgrade pip and install required libraries ----
echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing required Python libraries..."
pip install numpy
pip install opencv-python
pip install psutil
pip install mediapipe
pip install ultralytics==8.0.20
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# ---- 6. Final Info ----
echo "=============================="
echo "All installations complete!"
echo "To activate later, run:"
echo "source ~/capstone-venv/bin/activate"
echo "Then:"
echo "python3 Capstone-Stuff/'Capstone Files'/ThreatDetectionMK3.py"
echo "=============================="
