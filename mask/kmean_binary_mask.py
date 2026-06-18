#!/usr/bin/env python3
"""Create black-object/white-background binary masks with color K-means."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from mask_common import Timer, clean_mask, image_files, read_rgb, relative_mask_path, write_binary_mask


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_DIR / "website" / "dataset_image" / "input_video"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "website" / "dataset_mask_kmean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate binary masks using K-means color clustering.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--border-pixels", type=int, default=12)
    parser.add_argument("--min-area-ratio", type=float, default=0.003)
    parser.add_argument("--max-area-ratio", type=float, default=0.75)
    parser.add_argument("--limit", type=int, help="Process only the first N images.")
    return parser.parse_args()


def border_mask(height: int, width: int, border_pixels: int) -> np.ndarray:
    border_pixels = max(1, min(border_pixels, height // 2, width // 2))
    mask = np.zeros((height, width), dtype=bool)
    mask[:border_pixels, :] = True
    mask[-border_pixels:, :] = True
    mask[:, :border_pixels] = True
    mask[:, -border_pixels:] = True
    return mask


def kmeans_object_mask(rgb: np.ndarray, clusters: int, border_pixels: int, min_area_ratio: float, max_area_ratio: float) -> np.ndarray:
    height, width = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    samples = lab.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, _ = cv2.kmeans(samples, max(2, clusters), None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape((height, width))
    borders = border_mask(height, width, border_pixels)
    image_area = height * width

    cluster_count = max(2, clusters)
    border_counts = np.array([float(((labels == cluster_id) & borders).sum()) for cluster_id in range(cluster_count)])
    background_cluster = int(border_counts.argmax())
    object_mask = labels != background_cluster
    area_ratio = float(object_mask.sum()) / image_area

    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        center_y1, center_y2 = height // 4, height - height // 4
        center_x1, center_x2 = width // 4, width - width // 4
        best_cluster = None
        best_score = -1.0
        for cluster_id in range(cluster_count):
            if cluster_id == background_cluster:
                continue
            cluster_mask = labels == cluster_id
            cluster_area_ratio = float(cluster_mask.sum()) / image_area
            if cluster_area_ratio < min_area_ratio or cluster_area_ratio > max_area_ratio:
                continue
            center_count = float(cluster_mask[center_y1:center_y2, center_x1:center_x2].sum())
            border_count = float(cluster_mask[borders].sum())
            score = center_count / max(1.0, border_count + 1.0)
            if score > best_score:
                best_score = score
                best_cluster = cluster_id
        object_mask = labels == best_cluster if best_cluster is not None else labels != background_cluster

    return clean_mask(object_mask)


def main() -> int:
    args = parse_args()
    input_dir = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images = image_files(input_dir)
    if args.limit:
        images = images[: max(1, args.limit)]
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")

    timer = Timer()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"K-means images: {len(images)}. Output: {output_dir}", flush=True)
    for index, image_path in enumerate(images, start=1):
        started_at = time.perf_counter()
        rgb = read_rgb(image_path)
        object_mask = kmeans_object_mask(
            rgb,
            args.clusters,
            args.border_pixels,
            args.min_area_ratio,
            args.max_area_ratio,
        )
        mask_path = relative_mask_path(image_path, input_dir, output_dir)
        write_binary_mask(object_mask, mask_path)
        elapsed = time.perf_counter() - started_at
        timer.add(image_path.relative_to(input_dir), elapsed, int(object_mask.sum()))
        print(f"{index}/{len(images)} {image_path.relative_to(input_dir)}: {elapsed:.3f}s", flush=True)

    timer.write(output_dir / "timing_report.json", "kmean")
    print(f"K-means mask creation complete. Timing report: {output_dir / 'timing_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
