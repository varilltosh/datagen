#!/usr/bin/env python3
"""Build hard-negative training images from distractor objects.

Pipeline:
  1. Run rembg on every image in distractor/ -> RGBA cutouts + masks.
     Masks are saved in mask_distractor/ (white bg, black object).
  2. Paste random distractor cutouts onto random backgrounds at random size,
     position and HSV (hue/saturation/brightness) jitter, then save each
     composite into the YOLO train split with an EMPTY label file.

An empty .txt label tells YOLO the image contains NO target object, so the
model learns these distractors (e.g. nearby green things) are background and
stops firing high-confidence false positives on them.
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
REMBG_SCRIPT = PROJECT_DIR / "tools" / "rembg.py"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def image_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def run_rembg(distractor_dir: Path, cutout_dir: Path, mask_dir: Path, device: str) -> None:
    print(f"[rembg] {distractor_dir} -> cutouts={cutout_dir}  masks={mask_dir}", flush=True)
    cmd = [
        sys.executable, str(REMBG_SCRIPT),
        "--input", str(distractor_dir),
        "--output", str(cutout_dir),
        "--mask-output", str(mask_dir),
        "--device", device,
    ]
    subprocess.run(cmd, cwd=PROJECT_DIR, check=True)


def load_cutout(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (bgr, alpha) cropped to the object's bounding box, or None."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 2:
        return None
    if raw.shape[2] == 4:
        bgr = raw[:, :, :3]
        alpha = raw[:, :, 3]
    else:
        bgr = raw[:, :, :3]
        alpha = np.full(bgr.shape[:2], 255, dtype=np.uint8)
    ys, xs = np.where(alpha > 0)
    if ys.size == 0:
        return None
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    return bgr[y1:y2, x1:x2].copy(), alpha[y1:y2, x1:x2].copy()


def jitter_hsv(bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Random hue shift + saturation + brightness/value scaling."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_shift = rng.uniform(-15, 15)            # degrees on the 0-179 OpenCV hue ring
    sat_scale = rng.uniform(0.5, 1.5)
    val_scale = rng.uniform(0.5, 1.4)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def paste(canvas: np.ndarray, bgr: np.ndarray, alpha: np.ndarray,
          x: int, y: int) -> None:
    ch, cw = canvas.shape[:2]
    oh, ow = bgr.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(cw, x + ow), min(ch, y + oh)
    if x2 <= x1 or y2 <= y1:
        return
    sx, sy = x1 - x, y1 - y
    obj = bgr[sy:sy + (y2 - y1), sx:sx + (x2 - x1)]
    a = alpha[sy:sy + (y2 - y1), sx:sx + (x2 - x1)]
    m = a > 0
    roi = canvas[y1:y2, x1:x2]
    roi[m] = obj[m]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distractor-dir", type=Path, default=PROJECT_DIR / "distractor")
    ap.add_argument("--mask-dir", type=Path, default=PROJECT_DIR / "mask_distractor")
    ap.add_argument("--cutout-dir", type=Path, default=PROJECT_DIR / "workspace" / "cutouts_distractor")
    ap.add_argument("--bg-dir", type=Path, default=PROJECT_DIR / "assets" / "backgrounds")
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_DIR / "workspace" / "data" / "sythesized_data" / "train")
    ap.add_argument("--count", type=int, default=1500, help="Number of hard-negative images.")
    ap.add_argument("--min-objs", type=int, default=1)
    ap.add_argument("--max-objs", type=int, default=4)
    ap.add_argument("--min-scale", type=float, default=0.08, help="Min object size as fraction of canvas width.")
    ap.add_argument("--max-scale", type=float, default=0.45, help="Max object size as fraction of canvas width.")
    ap.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda", "rocm"])
    ap.add_argument("--skip-rembg", action="store_true", help="Reuse existing cutouts (skip rembg step).")
    ap.add_argument("--prefix", default="hardneg", help="Output filename prefix.")
    args = ap.parse_args()

    rng = random.Random()

    # Step 1: rembg -> cutouts + mask_distractor/
    if not args.skip_rembg:
        run_rembg(args.distractor_dir, args.cutout_dir, args.mask_dir, args.device)
    else:
        print("[rembg] skipped (--skip-rembg)", flush=True)

    cutouts = image_files(args.cutout_dir)
    if not cutouts:
        print(f"ERROR: no cutouts found in {args.cutout_dir}", file=sys.stderr)
        return 1
    backgrounds = image_files(args.bg_dir)
    if not backgrounds:
        print(f"ERROR: no backgrounds found in {args.bg_dir}", file=sys.stderr)
        return 1

    # Pre-load cutouts once.
    loaded = [c for c in (load_cutout(p) for p in cutouts) if c is not None]
    print(f"Loaded {len(loaded)} distractor cutout(s); {len(backgrounds)} background(s).", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: composite hard negatives with empty labels.
    saved = 0
    attempts = 0
    max_attempts = args.count * 10
    while saved < args.count and attempts < max_attempts:
        attempts += 1
        bg = cv2.imread(str(rng.choice(backgrounds)), cv2.IMREAD_COLOR)
        if bg is None:
            continue
        canvas = bg.copy()
        ch, cw = canvas.shape[:2]

        n_objs = rng.randint(args.min_objs, args.max_objs)
        for _ in range(n_objs):
            bgr, alpha = rng.choice(loaded)
            bgr = jitter_hsv(bgr, rng)
            oh, ow = bgr.shape[:2]
            target_w = rng.uniform(args.min_scale, args.max_scale) * cw
            scale = target_w / max(ow, oh)
            new_w = max(8, int(ow * scale))
            new_h = max(8, int(oh * scale))
            bgr_r = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            alpha_r = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            # allow up to ~30% out of frame
            x = rng.randint(int(-0.3 * new_w), int(cw - 0.7 * new_w))
            y = rng.randint(int(-0.3 * new_h), int(ch - 0.7 * new_h))
            paste(canvas, bgr_r, alpha_r, x, y)

        stem = f"{args.prefix}_{saved:06d}"
        cv2.imwrite(str(args.out_dir / f"{stem}.jpg"), canvas)
        (args.out_dir / f"{stem}.txt").write_text("", encoding="utf-8")  # empty = background/negative
        saved += 1
        if saved % 100 == 0:
            print(f"  {saved}/{args.count}", flush=True)

    print(f"Done: {saved} hard-negative image(s) -> {args.out_dir}", flush=True)
    print(f"Masks saved in: {args.mask_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
