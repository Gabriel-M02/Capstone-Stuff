#!/bin/bash
# EVST CM4 Adafruit 1.44" TFT Setup Script (ST7735 Green Tab)
# This installs and builds the fbcp-ili9341 driver for SPI display mirroring

set -e  # Exit if any command fails

echo "Updating system and installing dependencies..."
sudo apt update && sudo apt install -y git cmake build-essential

echo "Cloning fbcp-ili9341 repository..."
cd ~
git clone https://github.com/juj/fbcp-ili9341.git || true
cd fbcp-ili9341
mkdir -p build && cd build

echo "Configuring build for Adafruit 1.44 TFT (ST7735, Green Tab)..."
cmake -DARMV8A=ON \
  -DST7735=ON \
  -DGPIO_TFT_DATA_CONTROL=25 \
  -DGPIO_TFT_RESET_PIN=24 \
  -DGPIO_TFT_CS=8 \
  -DSPI_BUS_CLOCK_DIVISOR=6 \
  ..

echo "Building driver..."
make -j4

echo "Installing fbcp-ili9341 binary and service..."
sudo cp fbcp-ili9341 /usr/local/bin/
sudo tee /etc/systemd/system/fbcp.service >/dev/null <<'EOF'
[Unit]
Description=fbcp-ili9341 display driver
After=multi-user.target

[Service]
ExecStart=/usr/local/bin/fbcp-ili9341
Restart=always
User=root

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now fbcp.service

echo "Setup complete! The screen will now start automatically on boot."
