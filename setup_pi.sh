#!/bin/bash

echo "🍓 Raspberry Pi Environment Setup for Drone-openCV"
echo "================================================"

# 1. Update system packages
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# 2. Install system dependencies for OpenCV and system tools
echo "🔧 Installing system dependencies..."
sudo apt-get install -y python3-opencv python3-pip python3-venv libatlas-base-dev cmake

# 3. Create Virtual Environment
echo "🐍 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv --system-site-packages # Use system site-packages to inherit apt-installed opencv
    echo "✅ Virtual environment created."
else
    echo "⚠️  Virtual environment already exists."
fi

# 4. Activate Venv and Install Python Libraries
echo "📥 Installing Python libraries..."
source venv/bin/activate

# Install non-heavy libraries first
pip install numpy matplotlib pandas

# Ultralytics for YOLO (automatically installs torch, check for arm version if needed)
echo "🧠 Installing Ultralytics (YOLO)..."
pip install ultralytics

# 5. Fix for potential externally managed environment issues (PEP 668)
# If pip install fails, we might need to rely more on apt, but venv usually solves this.

echo "================================================"
echo "✅ Setup Complete!"
echo ""
echo "To activate the environment, run:"
echo "source venv/bin/activate"
echo ""
echo "To test OpenCV:"
echo "python3 -c 'import cv2; print(cv2.__version__)'"
echo ""
echo "To test YOLO:"
echo "yolo version"
echo "================================================"
