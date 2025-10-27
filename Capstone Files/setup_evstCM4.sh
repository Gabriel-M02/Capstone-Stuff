#!/bin/bash
set -e

echo "=== EVST Tactical Solutions CM4 Setup ==="

# Step 1: Update repositories
echo "[1/6] Updating system packages..."
sudo apt update -y && sudo apt upgrade -y

# Step 2: Expand swap to 2 GB (for heavy builds)
echo "[2/6] Configuring swap..."
sudo apt install -y dphys-swapfile
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup && sudo systemctl restart dphys-swapfile

# Step 3: Install core build dependencies
echo "[3/6] Installing core development tools..."
sudo apt install -y python3-pip python3-opencv python3-numpy libatlas-base-dev libhdf5-dev libjpeg-dev libpng-dev libtiff5-dev pkg-config git

# Step 4: Upgrade pip and essential Python utilities
echo "[4/6] Upgrading pip and setuptools..."
python3 -m pip install --upgrade pip setuptools wheel

# Step 5: Install required Python libraries
echo "[5/6] Installing Python libraries..."
python3 -m pip install psutil mediapipe==0.10.0 ultralytics==8.0.196

# Step 6: Verify OpenCV and MediaPipe
echo "[6/6] Verifying OpenCV and MediaPipe installation..."
python3 - <<'EOF'
import cv2, mediapipe as mp, psutil
print("✅ OpenCV:", cv2.__version__)
print("✅ MediaPipe:", mp.__version__)
print("✅ psutil:", psutil.__version__)
EOF

echo "=== Setup complete. System ready for EVST HUD deployment. ==="
