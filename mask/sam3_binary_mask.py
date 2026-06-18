#!/usr/bin/env python3
"""Create binary masks with a SAM3-compatible package when installed."""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import torch

from mask_common import Timer, choose_object_mask, image_files, read_rgb, relative_mask_path, write_binary_mask


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_DIR / "website" / "dataset_image" / "input_video"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "website" / "dataset_mask_sam3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate binary masks with SAM3 if a compatible package is installed.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, required=True, help="SAM3 checkpoint path.")
    parser.add_argument("--model-type", default="default")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--min-area-ratio", type=float, default=0.005)
    parser.add_argument("--max-area-ratio", type=float, default=0.75)
    parser.add_argument("--limit", type=int, help="Process only the first N images.")
    return parser.parse_args()


def build_generator(args: argparse.Namespace):
    if importlib.util.find_spec("sam3") is not None:
        raise RuntimeError(
            "A 'sam3' package is installed, but this project does not know its API yet. "
            "Update build_generator() in mask/sam3_binary_mask.py to match that package."
        )

    if importlib.util.find_spec("segment_anything_3") is not None:
        raise RuntimeError(
            "A 'segment_anything_3' package is installed, but this project does not know its API yet. "
            "Update build_generator() in mask/sam3_binary_mask.py to match that package."
        )

    raise RuntimeError(
        "SAM3 is not installed in this environment. Install your SAM3 package and provide --checkpoint. "
        "Current environment has SAM2 and segment_anything, but no sam3/segment_anything_3 module."
    )


def main() -> int:
    args = parse_args()
    input_dir = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for SAM3, but torch.cuda.is_available() is False.")

    images = image_files(input_dir)
    if args.limit:
        images = images[: max(1, args.limit)]
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")

    generator = build_generator(args)
    timer = Timer()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"SAM3 device: {args.device}. Images: {len(images)}. Output: {output_dir}", flush=True)

    for index, image_path in enumerate(images, start=1):
        started_at = time.perf_counter()
        rgb = read_rgb(image_path)
        masks = generator.generate(rgb)
        if isinstance(masks, np.ndarray):
            masks = [{"segmentation": masks}]
        object_mask = choose_object_mask(masks, rgb.shape[:2], args.min_area_ratio, args.max_area_ratio)
        mask_path = relative_mask_path(image_path, input_dir, output_dir)
        write_binary_mask(object_mask, mask_path)
        elapsed = time.perf_counter() - started_at
        timer.add(image_path.relative_to(input_dir), elapsed, int(object_mask.sum()))
        print(f"{index}/{len(images)} {image_path.relative_to(input_dir)}: {elapsed:.3f}s", flush=True)

    timer.write(output_dir / "timing_report.json", "sam3")
    print(f"SAM3 mask creation complete. Timing report: {output_dir / 'timing_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
