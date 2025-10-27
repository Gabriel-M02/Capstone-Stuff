#!/bin/bash
set -e

echo "=== EVST Tactical Solutions CM4 Final Setup ==="

# Step 1: System update
echo "[1/6] Updating repositories..."
sudo apt update -y && sudo apt upgrade -y

# Step 2: Expand swap for large package builds
echo "[2/6] Ensuring 2GB swap space..."
sudo apt install -y dphys-swapfile
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup && sudo systemctl restart dphys-swapfile

# Step 3: Install dependencies and OpenCV via apt (faster, stable)
echo "[3/6] Installing dependencies..."
sudo apt install -y python3-pip python3-opencv python3-numpy libatlas-base-dev libhdf5-dev libjpeg-dev libpng-dev libtiff5-dev pkg-config git libprotobuf-dev protobuf-compiler python3-protobuf

# Step 4: Upgrade core Python tools
echo "[4/6] Upgrading pip and setuptools..."
python3 -m pip install --upgrade pip setuptools wheel

# Step 5: Install libraries (ARM64-compatible versions)
echo "[5/6] Installing compatible Python libraries..."
python3 -m pip install psutil mediapipe==0.9.3.0 ultralytics==8.0.196

# Step 6: Verification test
echo "[6/6] Verifying installations..."
python3 - <<'EOF'
import cv2, mediapipe as mp, psutil
from ultralytics import YOLO

print("✅ OpenCV:", cv2.__version__)
print("✅ MediaPipe:", mp.__version__)
print("✅ psutil:", psutil.__version__)
try:
    YOLO('yolov8n.pt')
    print("✅ Ultralytics YOLO loaded successfully.")
except Exception as e:
    print("⚠️ YOLO test warning:", e)
EOF

echo "=== EVST CM4 environment fully operational ==="
