#!/usr/bin/env python3
"""Create black-object/white-background binary masks with SAM2 on GPU."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from mask_common import Timer, choose_object_mask, image_files, read_rgb, relative_mask_path, write_binary_mask

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [path for path in sys.path if Path(path or ".").resolve() != SCRIPT_DIR]

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_DIR / "website" / "dataset_image" / "input_video"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "website" / "dataset_mask_sam2"
DEFAULT_CHECKPOINT = Path("/home/skuba/skuba_ws/src/try_vision_project/datagen_sam/checkpoints/sam2_hiera_base_plus.pt")
DEFAULT_CONFIG = "configs/sam2/sam2_hiera_b+.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate binary masks with SAM2 automatic mask generation.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.8)
    parser.add_argument("--stability-score-thresh", type=float, default=0.9)
    parser.add_argument("--min-area-ratio", type=float, default=0.005)
    parser.add_argument("--max-area-ratio", type=float, default=0.75)
    parser.add_argument("--limit", type=int, help="Process only the first N images.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for SAM2, but torch.cuda.is_available() is False.")

    images = image_files(input_dir)
    if args.limit:
        images = images[: max(1, args.limit)]
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")

    model = build_sam2(args.config, str(checkpoint), device=args.device, mode="eval")
    generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        output_mode="binary_mask",
    )

    timer = Timer()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"SAM2 device: {args.device}. Images: {len(images)}. Output: {output_dir}", flush=True)
    for index, image_path in enumerate(images, start=1):
        started_at = time.perf_counter()
        rgb = read_rgb(image_path)
        masks = generator.generate(rgb)
        object_mask = choose_object_mask(masks, rgb.shape[:2], args.min_area_ratio, args.max_area_ratio)
        mask_path = relative_mask_path(image_path, input_dir, output_dir)
        write_binary_mask(object_mask, mask_path)
        elapsed = time.perf_counter() - started_at
        timer.add(image_path.relative_to(input_dir), elapsed, int(object_mask.sum()))
        print(f"{index}/{len(images)} {image_path.relative_to(input_dir)}: {elapsed:.3f}s", flush=True)

    timer.write(output_dir / "timing_report.json", "sam2")
    print(f"SAM2 mask creation complete. Timing report: {output_dir / 'timing_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
