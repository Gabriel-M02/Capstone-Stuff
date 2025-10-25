#!/bin/bash

echo "🟡 Updating and installing system dependencies..."
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y \
  build-essential \
  libssl-dev \
  zlib1g-dev \
  libbz2-dev \
  libreadline-dev \
  libsqlite3-dev \
  wget \
  curl \
  llvm \
  libncursesw5-dev \
  xz-utils \
  tk-dev \
  libxml2-dev \
  libxmlsec1-dev \
  libffi-dev \
  liblzma-dev \
  git \
  libqtgui4 \
  libqt4-test \
  libv4l-dev \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev \
  libxvidcore-dev \
  libx264-dev \
  v4l-utils

echo "🟢 Downloading Python 3.9.13 source..."
cd ~
wget https://www.python.org/ftp/python/3.9.13/Python-3.9.13.tgz
tar -xf Python-3.9.13.tgz
cd Python-3.9.13

echo "🔧 Building Python 3.9.13 (this will take 5-10 minutes)..."
./configure --enable-optimizations
make -j $(nproc)
sudo make altinstall

echo "✅ Python 3.9.13 installed at: /usr/local/bin/python3.9"
cd ~

echo "📁 Cloning EVST repo..."
if [ ! -d "Capstone-Stuff" ]; then
  git clone https://github.com/Gabriel-M02/Capstone-Stuff.git
else
  echo "Repo already exists. Skipping clone."
fi

echo "🛠️ Creating virtual environment with Python 3.9.13..."
/usr/local/bin/python3.9 -m venv capstone-venv

echo "⚙️ Activating environment and installing Python libraries..."
source capstone-venv/bin/activate

pip install --upgrade pip
pip install opencv-python
pip install numpy
pip install psutil
pip install mediapipe
pip install ultralytics==8.0.20
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "✅ All libraries installed successfully!"

echo ""
echo "To run your HUD, use:"
echo "source capstone-venv/bin/activate"
echo "python3 Capstone-Stuff/Capstone\ Files/ThreatDetectionMK3.py"
