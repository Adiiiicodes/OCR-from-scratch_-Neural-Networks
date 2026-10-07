#!/bin/bash
# run.sh - Full setup + run for OCR-from-scratch project

set -e

# 1. Create/activate venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# 2. Install deps
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# 3. Setup MNIST
./setup_mnist.sh

# 4. Run the main script (adjust name if different)
if [ -f "main.py" ]; then
    python3 main.py
elif [ -f "train.py" ]; then
    python3 train.py
else
    echo "No main.py or train.py found. Files in project:"
    ls -la *.py 2>/dev/null || echo "  (no .py files found)"
fi
