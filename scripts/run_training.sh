#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$ROOT_DIR/evaluation"

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$ROOT_DIR/evaluation/train_with_tuning_${timestamp}.log"

python3 "$SCRIPT_DIR/train_with_tuning.py" \
  --batch 16 \
  --epochs 150 \
  --patience 10 \
  --tune-epochs 10 \
  --tune-iterations 30 \
  --device 0 \
  2>&1 | tee "$log_file"
