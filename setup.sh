#!/bin/bash
# Setup script for SAM2 GUI: C. Elegans Tracker

set -e

# 1. Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

# 1. Clone and install SAM2 if not already present
if [ ! -d "segment_anything_2" ]; then
    echo "Cloning SAM2..."
    git clone https://github.com/facebookresearch/segment-anything-2 segment_anything_2
fi

echo "Installing SAM2..."
uv pip install -q -e segment_anything_2/.

# 3. Install project dependencies
echo "Installing project dependencies..."
uv pip install -q -r requirements.txt

# 3. Download model checkpoints
echo "Downloading model checkpoints..."
mkdir -p checkpoints
cd checkpoints
bash ../segment_anything_2/checkpoints/download_ckpts.sh
cd ..

echo "Setup complete. Activate your environment with 'source .venv/bin/activate' then run 'python Main.py' to start."
