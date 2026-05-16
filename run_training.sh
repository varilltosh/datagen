#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p evaluation

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="evaluation/train_with_tuning_${timestamp}.log"

python3 train_with_tuning.py \
  --model yolov8s.pt \
  --data /home/skuba/skuba_ws/src/skuba_vision_project/datagen/website/data/sythesized_data.yaml \
  --project /home/skuba/skuba_ws/src/skuba_vision_project/datagen/evaluation \
  --batch 16 \
  --epochs 150 \
  --patience 10 \
  --tune-epochs 10 \
  --tune-iterations 30 \
  --device 0 \
  2>&1 | tee "$log_file"
