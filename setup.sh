#!/usr/bin/env bash
# setup.sh — Install dependencies and prepare the datagen workspace.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Checking Python version..."
python3 --version

echo "==> Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install rembg with GPU support if CUDA is available
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "==> CUDA detected — reinstalling onnxruntime-gpu..."
  pip install --force-reinstall onnxruntime-gpu
else
  echo "==> No CUDA detected — using CPU onnxruntime."
fi

echo "==> Creating required directories..."
mkdir -p assets/backgrounds
mkdir -p models
mkdir -p workspace/input
mkdir -p workspace/frames
mkdir -p workspace/data/phase1
mkdir -p workspace/data/phase4

echo "==> Initialising config.yaml (fills defaults for missing keys)..."
python3 -c "
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('app')))
from server import ensure_config_defaults
ensure_config_defaults()
print('config.yaml ready.')
"

echo ""
echo "Setup complete. Start the server with:"
echo "  python3 app/server.py"
