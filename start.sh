#!/bin/bash

# Quick Start Script for Hardware Validation Suite

echo "🚁 Hardware Validation Suite - Quick Start"
echo "==========================================="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! pip show opencv-contrib-python &> /dev/null; then
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

echo ""
echo "Environment ready! 🎉"
echo ""
echo "Available commands:"
echo "  python validation_suite.py              # Run with capture card (camera index 1)"
echo "  python validation_suite.py --camera 0   # Run with webcam"
echo "  python validation_suite.py --debug      # Debug mode (uses webcam)"
echo "  python validation_suite.py --help       # Show all options"
echo ""
echo "To deactivate virtual environment when done: deactivate"
echo ""
