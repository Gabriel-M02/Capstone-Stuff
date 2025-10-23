#!/bin/bash

echo "📦 Updating system packages..."
sudo apt update && sudo apt full-upgrade -y

echo "🔧 Installing required system packages..."
sudo apt install -y git python3 python3-venv python3-pip libatlas-base-dev libjasper-dev libqtgui4 libqt4-test libilmbase-dev libopenexr-dev libgstreamer1.0-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev

echo "📁 Cloning Capstone GitHub repository if not already present..."
if [ ! -d "Capstone-Stuff" ]; then
    git clone https://github.com/Gabriel-M02/Capstone-Stuff.git
else
    echo "✅ Repository already exists. Skipping clone."
fi

echo "🧪 Creating Python virtual environment..."
python3 -m venv capstone-venv

echo "📂 Activating virtual environment and installing Python requirements..."
source capstone-venv/bin/activate

if [ -f "Capstone-Stuff/requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r Capstone-Stuff/requirements.txt
else
    echo "❌ Error: requirements.txt not found in Capstone-Stuff."
    exit 1
fi

echo "✅ Setup complete!"

echo "📌 To activate the virtual environment and run the program later:"
echo "    source capstone-venv/bin/activate"
echo "    python3 Capstone-Stuff/ThreatDetectionMK2.py"
