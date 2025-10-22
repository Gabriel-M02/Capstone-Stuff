#!/bin/bash

# Ensure the system is updated
sudo apt update && sudo apt upgrade -y

# Install dependencies for building Python
sudo apt install -y build-essential libssl-dev zlib1g-dev \
  libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
  libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev \
  libffi-dev git wget

# Download and install Python 3.12.3
cd /tmp
wget https://www.python.org/ftp/python/3.12.3/Python-3.12.3.tgz
tar -xf Python-3.12.3.tgz
cd Python-3.12.3
./configure --enable-optimizations
make -j$(nproc)
sudo make altinstall

# Make python3.12 the default (optional)
sudo update-alternatives --install /usr/bin/python python /usr/local/bin/python3.12 1

# Install pip for Python 3.12
wget https://bootstrap.pypa.io/get-pip.py
sudo /usr/local/bin/python3.12 get-pip.py

# Install virtualenv (optional but recommended)
sudo /usr/local/bin/pip3.12 install virtualenv

# Clone the repo (use your GitHub HTTPS or SSH URL)
cd ~
git clone https://github.com/Gabriel-M02/Capstone-Stuff.git

# Create and activate virtual environment
cd Capstone-Stuff
virtualenv -p /usr/local/bin/python3.12 .venv
source .venv/bin/activate

# Install required libraries
pip install opencv-python numpy psutil ultralytics mediapipe
