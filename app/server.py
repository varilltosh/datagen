from __future__ import annotations

import base64
import csv
import json
import re
import random
import shutil
import statistics
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import yaml
from flask import Flask, Response, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_functional


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
VIDEO_DIR = PROJECT_DIR / "workspace" / "input"
IMAGE_DIR = PROJECT_DIR / "workspace" / "frames"
MASK_DIR = PROJECT_DIR / "workspace" / "masks"
REMBG_DIR = PROJECT_DIR / "workspace" / "cutouts"
DISTRACTOR_CUTOUT_DIR = PROJECT_DIR / "workspace" / "cutouts_distractor"
BG_DIR = PROJECT_DIR / "assets" / "backgrounds"
DATA_DIR = PROJECT_DIR / "workspace" / "data"
PHASE1_DIR = DATA_DIR / "phase1"
PHASE4_DIR = DATA_DIR / "phase4"
SYNTH_IMAGE_DIR = DATA_DIR / "sythesized_data"
YOLO_IMAGE_ALIAS_DIR = DATA_DIR / "images"
# Labels live *inside* each image split folder (sythesized_data/<split>/*.txt),
# not in a separate labels/ tree. YOLO resolves the images→sythesized_data
# symlink, so the resolved image path loses the "/images/" segment and the
# images→labels path swap can't fire — it then looks for each label next to
# its image. Keeping labels alongside images is what makes them discoverable.
SYNTH_LABEL_DIR = SYNTH_IMAGE_DIR
YOLO_DATASET_YAML = DATA_DIR / "sythesized_data.yaml"
OBJECT_PROFILE_JSON = DATA_DIR / "object_profiles.json"
FRAME_STRIDE_SCRIPT = PROJECT_DIR / "scripts" / "frame_stride.py"
REMBG_SCRIPT = PROJECT_DIR / "tools" / "rembg.py"
CONFIG_PATH = PROJECT_DIR / "config.yaml"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".move", ".avi", ".mkv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024 * 1024

job_lock = threading.Lock()
job: dict = {
    "id": None,
    "running": False,
    "stage": "Idle",
    "percent": 0,
    "message": "Upload a folder of videos to begin.",
    "log": [],
    "error": None,
    "frame_count_warning": None,
}

_config_lock = threading.Lock()
_config: dict = {}

_class_status_lock = threading.Lock()
_class_status: dict[str, str] = {}

phase3_lock = threading.Lock()
_phase3_job: dict = {
    "running": False,
    "stage": "Idle",
    "epoch": 0,
    "total_epochs": 0,
    "current_cls_loss": None,
    "cls_loss": None,
    "quality": None,
    "best_pt": None,
    "error": None,
}
_phase3_process: subprocess.Popen | None = None

phase4_lock = threading.Lock()
_phase4_job: dict = {
    "running": False,
    "stage": "Idle",
    "frame": 0,
    "total_frames": 0,
    "results_count": 0,
    "csv_path": None,
    "error": None,
    "summary": None,
}
_phase4_stop = threading.Event()

ROS_MODEL_PATH = PROJECT_DIR / "best(1).pt"
ROS_IMAGE_TOPIC = "/camera/camera/color/image_raw"


# ── Config ───────────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict = {
    "phase0": {
        "frame_count_tolerance_pct": 20,
        "images_per_video": 108,
        "ssim_dedup_threshold": 0.97,
        "target_frame_count": 108,
    },
    "phase1": {
        "expected_images_per_class": 108,
        "min_backgrounds_per_category": 15,
    },
    "phase2": {
        "blur_max_pct": 0,
        "brightness_variation_pct": 10,
        "contrast_variation_pct": 20,
        "grid_cols": 3,
        "grid_rows": 3,
        "max_objects_per_image": 15,
        "out_of_frame_pct": 10,
        "output_resolution": ["640", "640"],
        "overlap_threshold_pct": 20,
        "placement_mode": "random",
        "start_idx": 0,
        "test_pct": 0,
        "train_pct": 60,
        "val_pct": 40,
        # Systematic distance/scale sweep: one image per 1px step of a
        # `sweep_reference_px`-wide reference object, from full size (close)
        # down to size*min_scale (far). The whole pool is synthesized with full
        # position × scale coverage, then split randomly into train/val by the
        # *_pct ratios above (test_pct is unused — no test set).
        "sweep_reference_px": 200,
        # Placement grid: objects are pinned to grid cells so every class visits
        # every cell at every scale. The grid is shared (no per-split cells);
        # train/val independence comes from the random split.
        "position_grid_rows": 6,
        "position_grid_cols": 6,
    },
    "phase3": {
        "batch": 4,
        "cls_loss_acceptable": 1,
        "cls_loss_excellent": 0.01,
        "device": "0",
        "epochs": 100,
        "imgsz": 640,
        "model": "yolov8s.pt",
        "name": "train",
        "patience": 20,
        "project": "runs/detect",
        "tune_epochs": 0,
        "tune_iterations": 0,
    },
    "phase4": {
        "accepted_formats": ["mp4", "avi", "mov"],
        "confidence_threshold": 0.75,
        "sample_fps": 1,
    },
}


def _deep_merge_defaults(base: dict, defaults: dict) -> dict:
    """Return a copy of base with any missing keys filled from defaults (one level deep)."""
    merged = dict(base)
    for section, section_defaults in defaults.items():
        if section not in merged or not isinstance(merged[section], dict):
            merged[section] = dict(section_defaults)
        else:
            section_copy = dict(merged[section])
            for key, val in section_defaults.items():
                if key not in section_copy:
                    section_copy[key] = val
            merged[section] = section_copy
    return merged


def ensure_config_defaults() -> None:
    """Write any missing config keys to disk using DEFAULT_CONFIG values."""
    with _config_lock:
        if not CONFIG_PATH.exists():
            merged = dict(DEFAULT_CONFIG)
        else:
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                    on_disk = yaml.safe_load(fh) or {}
            except Exception:
                on_disk = {}
            merged = _deep_merge_defaults(on_disk, DEFAULT_CONFIG)
        _config.update(merged)
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        yaml.dump(dict(_config), fh, default_flow_style=False, allow_unicode=True, sort_keys=False)


def load_config() -> dict:
    global _config
    with _config_lock:
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                    loaded = yaml.safe_load(fh) or {}
                _config = loaded
            except Exception:
                pass
        return dict(_config)


def get_cfg(*keys, default=None):
    with _config_lock:
        node = _config
        for key in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(key)
            if node is None:
                return default
        return node


def save_config(data: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
    load_config()


# ── Job state ─────────────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    for path in (
        VIDEO_DIR, IMAGE_DIR, MASK_DIR, REMBG_DIR,
        DATA_DIR, PHASE1_DIR, SYNTH_IMAGE_DIR, SYNTH_LABEL_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def safe_relative_path(raw_name: str) -> Path:
    parts = []
    for part in Path(raw_name.replace("\\", "/")).parts:
        if part in {"", ".", ".."}:
            continue
        clean = secure_filename(part)
        if clean:
            parts.append(clean)
    if not parts:
        parts = [f"upload_{uuid.uuid4().hex}"]
    return Path(*parts)


def safe_uploaded_video_path(raw_name: str) -> Path:
    """Store uploads as <class>/<filename>, where class is the containing folder."""
    safe_path = safe_relative_path(raw_name)
    parts = safe_path.parts
    if len(parts) >= 2:
        return Path(parts[-2]) / parts[-1]
    return safe_path


def set_job(**updates) -> None:
    with job_lock:
        job.update(updates)


def append_log(line: str) -> None:
    with job_lock:
        job["log"].append(line)
        job["log"] = job["log"][-200:]


def snapshot_job() -> dict:
    with job_lock:
        return dict(job)


# ── Class status ──────────────────────────────────────────────────────────────

def set_class_status(class_name: str, status: str) -> None:
    with _class_status_lock:
        _class_status[class_name] = status


def set_classes_status(names: list[str], status: str) -> None:
    with _class_status_lock:
        for name in names:
            _class_status[name] = status


def get_class_statuses() -> dict[str, str]:
    with _class_status_lock:
        return dict(_class_status)


def find_class_dir(base_dir: Path, class_name: str) -> Path | None:
    """Find the directory for class_name inside base_dir.

    Checks both base_dir/input_video/<class> and base_dir/<class> so the
    code works regardless of whether the user uploaded inside an 'input_video'
    parent folder or directly named the class folder.
    """
    for candidate in (
        base_dir / "input_video" / class_name,
        base_dir / class_name,
    ):
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _dirs_with_images(base_dir: Path) -> list[str]:
    """Return names of immediate subdirs of base_dir that contain image files.

    Also looks one level deeper under an 'input_video' subdirectory.
    """
    names: set[str] = set()
    if not base_dir.exists():
        return []
    for sub in base_dir.iterdir():
        if not sub.is_dir():
            continue
        if sub.name == "input_video":
            for class_sub in sub.iterdir():
                if class_sub.is_dir() and any(
                    p.suffix.lower() in IMAGE_EXTENSIONS
                    for p in class_sub.iterdir() if p.is_file()
                ):
                    names.add(class_sub.name)
        else:
            if any(p.suffix.lower() in IMAGE_EXTENSIONS for p in sub.iterdir() if p.is_file()):
                names.add(sub.name)
    return list(names)


def _dirs_with_files(base_dir: Path, extensions: set[str]) -> list[str]:
    names: set[str] = set()
    if not base_dir.exists():
        return []
    for path in base_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            try:
                rel = path.relative_to(base_dir)
            except ValueError:
                continue
            if len(rel.parts) >= 2:
                names.add(rel.parts[-2])
    return list(names)


def uploaded_class_names() -> list[str]:
    return sorted(_dirs_with_files(VIDEO_DIR, VIDEO_EXTENSIONS))


def all_known_classes() -> list[str]:
    """Return class names found across VIDEO_DIR, IMAGE_DIR, REMBG_DIR, and profiles."""
    classes: set[str] = set()
    classes.update(uploaded_class_names())
    for base in (IMAGE_DIR, REMBG_DIR):
        classes.update(_dirs_with_images(base))
    classes.update(load_object_profiles().keys())
    return sorted(classes)


def infer_class_status(class_name: str) -> str:
    """Infer status from filesystem when no explicit status is set."""
    phase1_img_dir = PHASE1_DIR / class_name / "images"
    if phase1_img_dir.exists() and any(
        p for p in phase1_img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ):
        return "Done"
    rembg_dir = find_class_dir(REMBG_DIR, class_name)
    if rembg_dir is not None and any(
        p for p in rembg_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ):
        return "In Review"
    return "Pending"


# ── Object profiles ───────────────────────────────────────────────────────────

def load_object_profiles() -> dict:
    if not OBJECT_PROFILE_JSON.exists():
        return {}
    try:
        return json.loads(OBJECT_PROFILE_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_object_profiles(profiles: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OBJECT_PROFILE_JSON.write_text(
        json.dumps(profiles, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def distance_sample(data: dict, key: str) -> dict | None:
    sample = data.get(key)
    if not isinstance(sample, dict):
        return None
    try:
        distance_cm = float(sample.get("distance_cm", 0))
        width = int(round(float(sample.get("width", 0))))
        height = int(round(float(sample.get("height", 0))))
        brightness = float(sample.get("brightness", 0))
        source_width = int(round(float(sample.get("source_width", 0))))
        source_height = int(round(float(sample.get("source_height", 0))))
        bbox_x = int(round(float(sample.get("bbox_x", 0))))
        bbox_y = int(round(float(sample.get("bbox_y", 0))))
    except (TypeError, ValueError):
        return None
    if distance_cm <= 0 or width <= 1 or height <= 1:
        return None
    return {
        "distance_cm": round(distance_cm, 3),
        "width": width,
        "height": height,
        "brightness": round(max(0.0, min(1.0, brightness)), 6),
        "source_width": source_width,
        "source_height": source_height,
        "bbox_x": bbox_x,
        "bbox_y": bbox_y,
    }


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining_seconds:.1f}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(remaining_minutes)}m {remaining_seconds:.1f}s"


def run_step(name: str, percent: int, command: list[str]) -> None:
    started_at = time.perf_counter()
    set_job(stage=name, percent=percent, message=f"{name} started")
    append_log(f"$ {' '.join(command)}")
    append_log(f"{name} timer started.")

    process = subprocess.Popen(
        command,
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line:
            append_log(line)
            set_job(message=line)

    return_code = process.wait()
    if return_code != 0:
        elapsed = format_duration(time.perf_counter() - started_at)
        raise RuntimeError(f"{name} failed with exit code {return_code} after {elapsed}")
    elapsed = format_duration(time.perf_counter() - started_at)
    append_log(f"{name} completed in {elapsed}.")
    set_job(message=f"{name} completed in {elapsed}")


# ── Phase 0: file renaming ────────────────────────────────────────────────────

def rename_to_convention(class_name: str) -> dict:
    """Rename rembg outputs and masks to <classname>_<tilt>_<index>.png.

    Tilt groups (by frame order): 00 → first third, 30 → middle third, 45 → last third.
    If fewer than 108 frames exist, distributes proportionally across 3 groups.
    """
    tilts = ["00", "30", "45"]
    result = {"images": 0, "masks": 0, "skipped": 0}

    rembg_class_dir = find_class_dir(REMBG_DIR, class_name)
    if rembg_class_dir is None:
        return result

    files = sorted(
        p for p in rembg_class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    n = len(files)
    if n == 0:
        return result

    # Divide into 3 groups using integer boundaries
    b0 = n // 3
    b1 = 2 * n // 3

    def group_and_within(i: int) -> tuple[int, int]:
        if i < b0:
            return 0, i
        if i < b1:
            return 1, i - b0
        return 2, i - b1

    # Build plan first so we don't corrupt filenames mid-loop
    plan: list[tuple[Path, str]] = []
    for i, fp in enumerate(files):
        grp, within = group_and_within(i)
        new_name = f"{class_name}_{tilts[grp]}_{within + 1:03d}.png"
        plan.append((fp, new_name))

    for old_path, new_name in plan:
        if old_path.name == new_name:
            result["skipped"] += 1
            continue
        if not old_path.exists():
            continue

        # Locate the corresponding mask before renaming the rembg file
        mask_class_dir = find_class_dir(MASK_DIR, class_name)
        mask_old: Path | None = None
        if mask_class_dir is not None:
            candidate = mask_class_dir / old_path.name
            if candidate.exists():
                mask_old = candidate
            else:
                cands = list(mask_class_dir.glob(f"{old_path.stem}.*"))
                mask_old = cands[0] if cands else None

        new_path = old_path.parent / new_name
        old_path.rename(new_path)
        result["images"] += 1

        if mask_old and mask_old.exists():
            mask_new = mask_old.parent / new_name
            if mask_old.name != new_name:
                mask_old.rename(mask_new)
                result["masks"] += 1

    return result


# ── Phase 0: composite helper ─────────────────────────────────────────────────

def make_review_composite(img_path: Path, mask_path: Path | None) -> bytes | None:
    """Return JPEG bytes of image with semi-transparent green mask overlay."""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return None

    composite = img.copy()
    if mask_path and mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            # Mask convention: 0 = object, 255 = background
            object_pixels = mask < 128
            green = np.zeros_like(img)
            green[object_pixels] = [0, 200, 0]
            composite = cv2.addWeighted(img, 0.65, green, 0.35, 0)

    ok, buf = cv2.imencode(".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return buf.tobytes() if ok else None


# ── Dataset processing ────────────────────────────────────────────────────────

def process_dataset(images_per_video: int) -> None:
    total_started_at = time.perf_counter()
    try:
        set_job(
            running=True,
            stage="Preparing",
            percent=5,
            message="Preparing output folders",
            error=None,
            log=[],
            frame_count_warning=None,
        )
        ensure_dirs()
        shutil.rmtree(IMAGE_DIR, ignore_errors=True)
        shutil.rmtree(MASK_DIR, ignore_errors=True)
        shutil.rmtree(REMBG_DIR, ignore_errors=True)
        ensure_dirs()

        # Mark all known classes as Processing
        known = all_known_classes()
        if known:
            set_classes_status(known, "Processing")

        run_step(
            "Frame stride",
            20,
            [
                sys.executable,
                str(FRAME_STRIDE_SCRIPT),
                "--input", str(VIDEO_DIR),
                "--output", str(IMAGE_DIR),
                "--images-per-video", str(images_per_video),
                "--device", "cuda",
            ],
        )

        # Frame count warning
        target = get_cfg("phase0", "target_frame_count", default=108)
        tolerance_pct = get_cfg("phase0", "frame_count_tolerance_pct", default=20)
        all_frames = [
            p for p in IMAGE_DIR.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        frame_count = len(all_frames)
        warning_msg = None
        if target and target > 0:
            deviation_pct = abs(frame_count - target) / target * 100
            if deviation_pct > tolerance_pct:
                warning_msg = (
                    f"WARNING: Extracted {frame_count} frame(s) but target is {target} "
                    f"(deviation {deviation_pct:.1f}% > tolerance {tolerance_pct}%)."
                )
                append_log(warning_msg)
                set_job(frame_count_warning=warning_msg)

        set_job(stage="Frame stride", percent=55, message="Frames extracted")

        run_step(
            "Background remove",
            60,
            [
                sys.executable,
                str(REMBG_SCRIPT),
                "--input", str(IMAGE_DIR),
                "--output", str(REMBG_DIR),
                "--mask-output", str(MASK_DIR),
                "--device", "cuda",
            ],
        )

        # Rename all class outputs to naming convention and update statuses
        processed_classes = class_names()
        for cls in processed_classes:
            rename_info = rename_to_convention(cls)
            append_log(
                f"Renamed class '{cls}': "
                f"{rename_info['images']} image(s), {rename_info['masks']} mask(s)."
            )
            set_class_status(cls, "In Review")

        total_elapsed = format_duration(time.perf_counter() - total_started_at)
        append_log(f"Total processing time: {total_elapsed}.")
        set_job(
            stage="Complete",
            percent=100,
            message=f"Processing complete in {total_elapsed}",
        )
    except Exception as exc:
        append_log(str(exc))
        set_job(stage="Error", error=str(exc), message=str(exc))
    finally:
        set_job(running=False)


def image_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def class_names() -> list[str]:
    return sorted(set(_dirs_with_images(IMAGE_DIR)) | set(uploaded_class_names()))


def source_split_dir(class_name: str, split: str, root: Path) -> Path:
    class_dir = find_class_dir(root, class_name)
    if class_dir is None:
        class_dir = root / class_name
    split_dir = class_dir / split
    return split_dir if split_dir.exists() else class_dir


def match_mask(image_path: Path) -> Path | None:
    try:
        relative = image_path.relative_to(IMAGE_DIR)
    except ValueError:
        return None
    mask_path = MASK_DIR / relative
    if mask_path.exists():
        return mask_path
    matches = list((MASK_DIR / relative.parent).glob(f"{image_path.stem}.*"))
    return matches[0] if matches else None


def brightness_target_range(profile: dict) -> tuple[float, float] | None:
    try:
        value = float(profile.get("brightness", profile.get("hsv_v")))
    except (KeyError, TypeError, ValueError):
        return None
    if value <= 0:
        return None
    variation = get_cfg("phase2", "brightness_variation_pct", default=10) / 100.0
    return max(0.0, value * (1.0 - variation)), min(1.0, value * (1.0 + variation))


def adjust_object_brightness(
    image: np.ndarray,
    mask: np.ndarray,
    rng: random.Random,
    profile: dict | None,
) -> np.ndarray:
    if not profile:
        return image
    brightness_range = brightness_target_range(profile)
    if brightness_range is None:
        return image
    object_pixels = mask > 0
    if not np.any(object_pixels):
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    masked_hsv = hsv[object_pixels]
    current_v = max(float(masked_hsv[:, 2].mean()) / 255.0, 1e-6)
    target_v = rng.uniform(*brightness_range)
    hsv[..., 2] = np.clip(hsv[..., 2] * (target_v / current_v), 0.0, 255.0)
    adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return np.where(object_pixels[..., None], adjusted, image)


def apply_contrast(image: np.ndarray, mask: np.ndarray, rng: random.Random) -> np.ndarray:
    variation = get_cfg("phase2", "contrast_variation_pct", default=20) / 100.0
    if variation <= 0:
        return image
    factor = max(0.5, min(2.0, 1.0 + rng.uniform(-variation, variation)))
    object_pixels = mask > 0
    if not np.any(object_pixels):
        return image
    adjusted = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return np.where(object_pixels[..., None], adjusted, image)


def apply_blur(image: np.ndarray, mask: np.ndarray, rng: random.Random) -> np.ndarray:
    blur_max_pct = get_cfg("phase2", "blur_max_pct", default=0) / 100.0
    if blur_max_pct <= 0:
        return image
    max_k = min(3, max(1, int(min(image.shape[:2]) * blur_max_pct)))
    k = rng.randint(0, max_k)
    if k == 0:
        return image
    ksize = 2 * k + 1
    object_pixels = mask > 0
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return np.where(object_pixels[..., None], blurred, image)


def apply_geometric_augmentation(
    image: np.ndarray, mask: np.ndarray, rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    """Color-safe geometric augmentation: random horizontal flip + small
    rotation, applied identically to the object image and its mask. Adds pose /
    orientation variety (which photometric augmentation cannot) without touching
    hue or saturation, so colour class cues (e.g. a green cup) are preserved.
    Used both during synthesis and to upsample under-represented classes with
    fresh, non-duplicated images."""
    hflip_pct      = get_cfg("phase2", "hflip_pct", default=50) / 100.0
    rotate_max_deg = float(get_cfg("phase2", "rotate_max_deg", default=15))

    if hflip_pct > 0 and rng.random() < hflip_pct:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    if rotate_max_deg > 0:
        angle = rng.uniform(-rotate_max_deg, rotate_max_deg)
        if abs(angle) > 0.1:
            h, w = image.shape[:2]
            cx, cy = w / 2.0, h / 2.0
            m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            cos, sin = abs(m[0, 0]), abs(m[0, 1])
            # Expand the output canvas so corners are never clipped.
            new_w = int(round(h * sin + w * cos))
            new_h = int(round(h * cos + w * sin))
            m[0, 2] += new_w / 2.0 - cx
            m[1, 2] += new_h / 2.0 - cy
            image = cv2.warpAffine(image, m, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)
            mask = cv2.warpAffine(mask, m, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)
    return image, mask


def normalize_object_mask(
    mask: np.ndarray,
    alpha: np.ndarray | None,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Return a uint8 mask where object pixels are 255 and background is 0."""
    if alpha is not None:
        if alpha.shape[:2] != target_shape:
            alpha = cv2.resize(alpha, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        # Step 4: Filter out the transparent alpha pixels. Only the remaining
        # visible object pixels are eligible to be pasted onto a background.
        _, alpha_binary = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
        if cv2.findNonZero(alpha_binary) is not None:
            return alpha_binary

    if mask.shape[:2] != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

    top = mask[0, :]
    bottom = mask[-1, :]
    left = mask[:, 0]
    right = mask[:, -1]
    border_mean = float(np.concatenate((top, bottom, left, right)).mean())

    if border_mean >= 127.0:
        object_pixels = mask < 250
    else:
        object_pixels = mask > 5

    binary = np.zeros(mask.shape, dtype=np.uint8)
    binary[object_pixels] = 255
    return binary


def load_object(
    image_path: Path,
    mask_path: Path,
    device: torch.device,
    rng: random.Random,
    profile: dict | None,
):
    raw_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if raw_image is None or mask is None:
        return None
    alpha = None
    if raw_image.ndim == 2:
        image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
    elif raw_image.shape[2] == 4:
        image = raw_image[:, :, :3]
        alpha = raw_image[:, :, 3]
        image[alpha == 0] = 0
    else:
        image = raw_image[:, :, :3]
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = normalize_object_mask(mask, alpha, image.shape[:2])
    image = adjust_object_brightness(image, mask, rng, profile)
    image = apply_contrast(image, mask, rng)
    image = apply_blur(image, mask, rng)
    image, mask = apply_geometric_augmentation(image, mask, rng)
    if device.type == "cuda":
        image_tensor = torch.from_numpy(image).to(device=device, dtype=torch.float32).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).to(device=device)
        object_mask = mask_tensor > 0
        points = torch.nonzero(object_mask, as_tuple=False)
        if points.numel() == 0:
            return None
        y_min = int(points[:, 0].min().item())
        y_max = int(points[:, 0].max().item()) + 1
        x_min = int(points[:, 1].min().item())
        x_max = int(points[:, 1].max().item()) + 1
        cropped_image = image_tensor[:, y_min:y_max, x_min:x_max]
        cropped_mask = object_mask[y_min:y_max, x_min:x_max].to(dtype=torch.float32).unsqueeze(0)
        if cropped_image.numel() == 0 or cropped_mask.numel() == 0:
            return None
        return cropped_image, cropped_mask
    binary = mask
    points = cv2.findNonZero(binary)
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    cropped_image = image[y : y + h, x : x + w]
    cropped_mask = binary[y : y + h, x : x + w]
    if cropped_image.size == 0 or cropped_mask.size == 0:
        return None
    return cropped_image, cropped_mask


def canvas_from_background(background: np.ndarray, device: torch.device):
    if device.type == "cuda":
        return torch.from_numpy(background).to(device=device, dtype=torch.float32).permute(2, 0, 1)
    return background


def canvas_shape(canvas) -> tuple[int, int]:
    if isinstance(canvas, torch.Tensor):
        return int(canvas.shape[1]), int(canvas.shape[2])
    return canvas.shape[:2]


def object_shape(object_image) -> tuple[int, int]:
    if isinstance(object_image, torch.Tensor):
        return int(object_image.shape[1]), int(object_image.shape[2])
    return object_image.shape[:2]


def distance_scale_bounds(size_profile: dict | None) -> tuple[float, float] | None:
    """Return (min_scale, max_scale) for a profile's distance sweep, or None.

    The scale factor moves the object between its smallest (far) and largest
    (close) on-canvas size. max_scale is always 1.0; min_scale is the
    camera/wanted distance ratio (e.g. 35cm/210cm = 0.167)."""
    if not size_profile:
        return None
    try:
        d = float(size_profile.get("camera_object_distance_cm", 0) or 0)
        w = float(size_profile.get("wanted_distance_cm", 0) or 0)
        distance_scale = d / w
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if distance_scale <= 0:
        return None
    return tuple(sorted((1.0, distance_scale)))  # (min_scale, max_scale)


def distance_scaled_object_size(
    obj_w: int,
    obj_h: int,
    video_width: int,
    video_height: int,
    size_profile: dict | None,
    rng: random.Random,
    forced_scale: float | None = None,
) -> tuple[int, int] | None:
    if not size_profile:
        return None

    # Step 1: Calculate the minimum pixel boundaries from the current
    # video/background resolution and the original object dimensions.
    if video_width <= 0 or video_height <= 0 or obj_w <= 0 or obj_h <= 0:
        return None
    smallest_w = (1280 / video_width) * obj_w
    smallest_h = (720 / video_height) * obj_h

    # Step 2: Calculate maximum boundaries using the requested distance scale.
    try:
        distance_from_camera_to_object_cm = float(size_profile.get("camera_object_distance_cm", 0) or 0)
        wanted_distance_cm = float(size_profile.get("wanted_distance_cm", 0) or 0)
        distance_scale = distance_from_camera_to_object_cm / wanted_distance_cm
    except (TypeError, ValueError, ZeroDivisionError):
        distance_scale = 0.0
    if distance_scale > 0:
        # Step 3: Pick one scale factor so width and height move together
        # from the smallest object size to the largest object size. During a
        # systematic sweep `forced_scale` pins this to an exact step; otherwise
        # it is sampled at random within the range.
        min_scale, max_scale = sorted((1.0, distance_scale))
        if forced_scale is not None:
            sampled_scale = max(min_scale, min(max_scale, forced_scale))
        else:
            sampled_scale = rng.uniform(min_scale, max_scale)
        new_w = int(round(smallest_w * sampled_scale))
        new_h = int(round(smallest_h * sampled_scale))
        fit_scale = min(1.0, video_width / max(1, new_w), video_height / max(1, new_h))
        if fit_scale < 1.0:
            new_w = int(round(new_w * fit_scale))
            new_h = int(round(new_h * fit_scale))
        return max(2, new_w), max(2, new_h)

    try:
        max_range_cm = float(size_profile.get("object_max_range_cm", 0) or 0)
    except (TypeError, ValueError):
        return None
    if max_range_cm <= 0:
        return None

    try:
        min_distance_cm = float(size_profile.get("min_distance_cm", 0) or 0)
    except (TypeError, ValueError):
        min_distance_cm = 0.0
    if min_distance_cm <= 0:
        close_sample = size_profile.get("close_sample") or {}
        try:
            min_distance_cm = float(close_sample.get("distance_cm", max_range_cm))
        except (TypeError, ValueError):
            min_distance_cm = max_range_cm

    min_distance_cm = max(1.0, min(min_distance_cm, max_range_cm))
    sampled_distance_cm = rng.uniform(min_distance_cm, max_range_cm)
    scale = max_range_cm / sampled_distance_cm
    return max(2, round(obj_w * scale)), max(2, round(obj_h * scale))


def canvas_to_image(canvas) -> np.ndarray:
    if isinstance(canvas, torch.Tensor):
        return canvas.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return canvas


def paste_object(
    canvas,
    object_image,
    object_mask,
    rng: random.Random,
    occupied_boxes: list[tuple[int, int, int, int]],
    device: torch.device,
    size_profile: dict | None,
    grid_cell: tuple[int, int, int, int] | None = None,
    out_of_frame_pct: float = 0.0,
    overlap_threshold_pct: float = 0,
    max_attempts: int = 100,
    forced_scale: float | None = None,
    position_cells: list[tuple[int, int, int, int]] | None = None,
) -> tuple[any, tuple[int, int, int, int]] | None:
    canvas_h, canvas_w = canvas_shape(canvas)
    obj_h, obj_w = object_shape(object_image)
    min_side = max(24, int(canvas_w * 0.12))
    max_side = max(min_side, int(canvas_w * 0.24))

    placed = None
    for _ in range(max_attempts):
        distance_scaled_size = distance_scaled_object_size(
            obj_w, obj_h, canvas_w, canvas_h, size_profile, rng, forced_scale,
        )
        if distance_scaled_size is not None:
            new_w, new_h = distance_scaled_size
            new_w = min(canvas_w, new_w)
            new_h = min(canvas_h, new_h)
        elif size_profile:
            min_w = max(2, int(size_profile.get("min_width", size_profile.get("width", obj_w))))
            max_w = max(min_w, int(size_profile.get("max_width", size_profile.get("width", obj_w))))
            min_h = max(2, int(size_profile.get("min_height", size_profile.get("height", obj_h))))
            max_h = max(min_h, int(size_profile.get("max_height", size_profile.get("height", obj_h))))
            new_w = min(canvas_w, rng.randint(min_w, max_w))
            new_h = min(canvas_h, rng.randint(min_h, max_h))
        else:
            target_side = rng.randint(min_side, max_side)
            scale = target_side / max(obj_w, obj_h)
            new_w = max(2, min(canvas_w, int(obj_w * scale)))
            new_h = max(2, min(canvas_h, int(obj_h * scale)))

        if position_cells:
            # Aim the object's CENTRE at a point inside one of the allowed cells
            # (the cell is a target location, not a box the object must fit in —
            # objects are routinely larger than a cell). The centre is then
            # clamped so the whole object stays on-canvas. This lets the object
            # centroid walk every grid cell while always being placeable; only
            # an object larger than the entire canvas is rejected (retry smaller).
            pcx, pcy, pcw, pch = rng.choice(position_cells)
            cx_min, cx_max = new_w // 2, canvas_w - (new_w - new_w // 2)
            cy_min, cy_max = new_h // 2, canvas_h - (new_h - new_h // 2)
            if cx_min > cx_max or cy_min > cy_max:
                continue  # object bigger than the canvas — retry at a smaller scale
            des_x = rng.randint(pcx, pcx + pcw - 1)
            des_y = rng.randint(pcy, pcy + pch - 1)
            center_x = min(max(des_x, cx_min), cx_max)
            center_y = min(max(des_y, cy_min), cy_max)
            x = center_x - new_w // 2
            y = center_y - new_h // 2
            vis_box = (x, y, new_w, new_h)
            if not any(boxes_overlap(vis_box, box, overlap_threshold_pct) for box in occupied_boxes):
                placed = (x, y, new_w, new_h)
                break
        elif grid_cell is not None:
            cx, cy, cw, ch = grid_cell
            x = rng.randint(cx, max(cx, cx + cw - new_w))
            y = rng.randint(cy, max(cy, cy + ch - new_h))
            placed = (x, y, new_w, new_h)
            break
        else:
            out_x = int(new_w * out_of_frame_pct)
            out_y = int(new_h * out_of_frame_pct)
            x = rng.randint(-out_x, max(0, canvas_w - new_w + out_x))
            y = rng.randint(-out_y, max(0, canvas_h - new_h + out_y))
            vis_box = (
                max(0, x), max(0, y),
                min(canvas_w, x + new_w) - max(0, x),
                min(canvas_h, y + new_h) - max(0, y),
            )
            if not any(boxes_overlap(vis_box, box, overlap_threshold_pct) for box in occupied_boxes):
                placed = (x, y, new_w, new_h)
                break

    if placed is None:
        return None

    x, y, new_w, new_h = placed
    vis_x1 = max(0, x)
    vis_y1 = max(0, y)
    vis_x2 = min(canvas_w, x + new_w)
    vis_y2 = min(canvas_h, y + new_h)
    if vis_x2 <= vis_x1 or vis_y2 <= vis_y1:
        return None
    vis_box = (vis_x1, vis_y1, vis_x2 - vis_x1, vis_y2 - vis_y1)

    if device.type == "cuda":
        paste_object_cuda(
            canvas, object_image, object_mask, x, y, new_w, new_h, device,
            copy_pixels=distance_scaled_size is not None,
        )
    else:
        src_x, src_y = max(0, -x), max(0, -y)
        paste_w, paste_h = vis_x2 - vis_x1, vis_y2 - vis_y1
        interpolation = cv2.INTER_NEAREST if distance_scaled_size is not None else cv2.INTER_AREA
        obj_r = cv2.resize(object_image, (new_w, new_h), interpolation=interpolation)
        msk_r = cv2.resize(object_mask,  (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        obj_crop = obj_r[src_y:src_y + paste_h, src_x:src_x + paste_w]
        msk_crop = msk_r[src_y:src_y + paste_h, src_x:src_x + paste_w]
        # Step 4: Paste only alpha-surviving object pixels. Transparent
        # pixels from the cutout never overwrite the random background.
        object_pixels = msk_crop > 0
        roi = canvas[vis_y1:vis_y2, vis_x1:vis_x2]
        roi[object_pixels] = obj_crop[object_pixels]
    return canvas, vis_box


def paste_object_cuda(
    canvas: torch.Tensor,
    object_image: torch.Tensor,
    object_mask: torch.Tensor,
    x: int,
    y: int,
    width: int,
    height: int,
    device: torch.device,
    copy_pixels: bool = False,
) -> None:
    ch_t = int(canvas.shape[1])
    cw_t = int(canvas.shape[2])
    src_x, src_y = max(0, -x), max(0, -y)
    dst_x, dst_y = max(0, x), max(0, y)
    paste_w = min(width - src_x, cw_t - dst_x)
    paste_h = min(height - src_y, ch_t - dst_y)
    if paste_w <= 0 or paste_h <= 0:
        return
    object_image = object_image.to(device=device, dtype=torch.float32)
    object_mask  = object_mask.to(device=device, dtype=torch.float32)
    if copy_pixels:
        resized_image = torch_functional.interpolate(
            object_image.unsqueeze(0), size=(height, width), mode="nearest",
        ).squeeze(0)
    else:
        resized_image = torch_functional.interpolate(
            object_image.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False,
        ).squeeze(0)
    resized_mask = torch_functional.interpolate(
        object_mask.unsqueeze(0), size=(height, width), mode="nearest",
    ).squeeze(0)
    img_crop  = resized_image[:, src_y:src_y + paste_h, src_x:src_x + paste_w]
    mask_crop = resized_mask[:, src_y:src_y + paste_h, src_x:src_x + paste_w]
    roi = canvas[:, dst_y:dst_y + paste_h, dst_x:dst_x + paste_w]
    # Step 4: Paste only alpha-surviving object pixels on the CUDA path too.
    canvas[:, dst_y:dst_y + paste_h, dst_x:dst_x + paste_w] = torch.where(mask_crop > 0, img_crop, roi)


def boxes_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
    threshold_pct: float = 0,
) -> bool:
    fx, fy, fw, fh = first
    sx, sy, sw, sh = second
    if fx + fw <= sx or sx + sw <= fx or fy + fh <= sy or sy + sh <= fy:
        return False
    if threshold_pct <= 0:
        return True
    ix = min(fx + fw, sx + sw) - max(fx, sx)
    iy = min(fy + fh, sy + sh) - max(fy, sy)
    inter = max(0, ix) * max(0, iy)
    smaller = min(fw * fh, sw * sh)
    return smaller > 0 and (inter / smaller) > (threshold_pct / 100.0)


def write_yaml(names: list[str]) -> None:
    ensure_yolo_image_alias()
    lines = [
        f"path: {DATA_DIR}",
        "train: images/train",
        "val: images/validate",
        f"nc: {len(names)}",
        "names:",
    ]
    lines.extend(f"  {i}: {name}" for i, name in enumerate(names))
    YOLO_DATASET_YAML.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_yolo_image_alias() -> None:
    if YOLO_IMAGE_ALIAS_DIR.is_symlink() or YOLO_IMAGE_ALIAS_DIR.exists():
        if YOLO_IMAGE_ALIAS_DIR.resolve() == SYNTH_IMAGE_DIR.resolve():
            return
        if YOLO_IMAGE_ALIAS_DIR.is_dir() and not YOLO_IMAGE_ALIAS_DIR.is_symlink():
            shutil.rmtree(YOLO_IMAGE_ALIAS_DIR)
        else:
            YOLO_IMAGE_ALIAS_DIR.unlink()
    YOLO_IMAGE_ALIAS_DIR.symlink_to(SYNTH_IMAGE_DIR.name, target_is_directory=True)


def datagen_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _list_bg_categories() -> list[str]:
    """Return names of immediate subdirs of BG_DIR (each is a background category)."""
    if not BG_DIR.exists():
        return []
    return sorted(d.name for d in BG_DIR.iterdir() if d.is_dir() and not d.name.startswith("."))


def _bg_pool_for_class(
    class_name: str,
    all_backgrounds: list[Path],
    object_profiles: dict,
) -> list[Path]:
    """Backgrounds allowed for class_name based on its profile's background_categories.

    Falls back to all_backgrounds when the field is absent, empty, or all named
    category dirs are empty.
    """
    profile = object_profiles.get(class_name, {})
    cats = profile.get("background_categories") or []
    if not cats:
        return all_backgrounds
    pool: list[Path] = []
    for cat in cats:
        pool.extend(image_files(BG_DIR / cat))
    return pool if pool else all_backgrounds


def _build_bg_pool_per_class(
    names: list[str],
    all_backgrounds: list[Path],
    object_profiles: dict,
) -> dict[str, list[Path]]:
    return {name: _bg_pool_for_class(name, all_backgrounds, object_profiles) for name in names}


def measure_class_reference_px(
    names: list[str],
    sources: dict[str, list[tuple[Path, Path]]],
    device: torch.device,
    rng: random.Random,
    object_profiles: dict,
    sample: int = 40,
) -> dict[str, int]:
    """Measure each class's on-screen reference pixel size (the larger of the
    object's width/height) from its Phase-1 cutouts. This is the "close" size
    the distance sweep starts from; multiplied by the shared scale range it
    defines that class's own pixel range. Bigger objects therefore get more
    1px steps and stay in more images; smaller objects exhaust their range
    sooner and drop out. Returns {class_name: median_max_dim_px}."""
    ref_px: dict[str, int] = {}
    for name in names:
        pairs = sources[name]
        picks = pairs if len(pairs) <= sample else rng.sample(pairs, sample)
        dims: list[int] = []
        for img_path, mask_path in picks:
            loaded = load_object(img_path, mask_path, device, rng, object_profiles.get(name))
            if loaded is None:
                continue
            oh, ow = object_shape(loaded[0])
            dims.append(max(oh, ow))
        ref_px[name] = int(statistics.median(dims)) if dims else 2
    return ref_px


def _build_sources(names: list[str]) -> dict[str, list[tuple[Path, Path]]]:
    sources: dict[str, list[tuple[Path, Path]]] = {}
    for name in names:
        candidates: list[tuple[Path, Path]] = []
        phase1_img_dir  = PHASE1_DIR / name / "images"
        phase1_mask_dir = PHASE1_DIR / name / "masks"
        for image_path in image_files(phase1_img_dir):
            mask_path = phase1_mask_dir / image_path.name
            if not mask_path.exists():
                cands = list(phase1_mask_dir.glob(f"{image_path.stem}.*"))
                mask_path = cands[0] if cands else None
            if mask_path and mask_path.exists():
                candidates.append((image_path, mask_path))
        if not candidates:
            raise RuntimeError(f"No image/mask pairs for class '{name}'. Accept it in Phase 0 first.")
        sources[name] = candidates
    return sources


def partition_position_cells(
    rows: int, cols: int, train_pct: int, val_pct: int, test_pct: int,
    seed: int = 20240601,
) -> dict[str, list[tuple[int, int]]]:
    """Split a rows×cols grid of logical (row, col) cells into disjoint
    train / validate / test pools. Deterministic (fixed seed) so the same
    cells are always reserved for each split — that is what guarantees a
    val/test object centre never lands in a cell training could have used."""
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    random.Random(seed).shuffle(cells)
    total = max(1, train_pct + val_pct + test_pct)
    n = len(cells)
    n_val = max(1, round(n * val_pct / total)) if val_pct > 0 else 0
    n_test = max(1, round(n * test_pct / total)) if test_pct > 0 else 0
    n_train = max(1, n - n_val - n_test)
    return {
        "train": cells[:n_train],
        "validate": cells[n_train:n_train + n_val],
        "test": cells[n_train + n_val:n_train + n_val + n_test],
    }


def cells_to_pixels(
    logical_cells: list[tuple[int, int]], rows: int, cols: int,
    canvas_w: int, canvas_h: int,
) -> list[tuple[int, int, int, int]]:
    """Map logical (row, col) cells to pixel rectangles on the current canvas."""
    cw = max(1, canvas_w // cols)
    ch = max(1, canvas_h // rows)
    return [(c * cw, r * ch, cw, ch) for (r, c) in logical_cells]


def jitter_background_hsv(background: np.ndarray, rng: random.Random) -> np.ndarray:
    """Wide random hue/saturation + clamped brightness jitter on a BACKGROUND
    image only. Applied before any object is pasted, so target objects keep
    their original colour cues (class colour is preserved) while the scene
    lighting/colour varies — closing the sim-to-real gap and stopping the model
    keying on background colour. Brightness is clamped to ×0.4–1.6 to avoid
    degenerate pure-black/white frames."""
    hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_shift = rng.uniform(0, 180)
    sat_scale = rng.uniform(0.0, 2.0)
    val_scale = rng.uniform(0.4, 1.6)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def paste_distractors_preview(image: np.ndarray, rng: random.Random,
                              min_objs: int = 1, max_objs: int = 3) -> int:
    """Paste a few hard-negative distractor cutouts (HSV-jittered + rotated ±15°)
    onto `image` in place, mirroring scripts/make_hard_negatives.py, so the
    Phase-2 preview shows what hard negatives look like. Returns how many were
    pasted (0 if no distractor cutouts exist yet)."""
    cutouts = [
        p for p in DISTRACTOR_CUTOUT_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ] if DISTRACTOR_CUTOUT_DIR.exists() else []
    if not cutouts:
        return 0
    ch, cw = image.shape[:2]
    pasted = 0
    for _ in range(rng.randint(min_objs, max_objs)):
        raw = cv2.imread(str(rng.choice(cutouts)), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.ndim != 3 or raw.shape[2] != 4:
            continue
        bgr, alpha = raw[:, :, :3], raw[:, :, 3]
        ys, xs = np.where(alpha > 0)
        if ys.size == 0:
            continue
        bgr = bgr[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()
        alpha = alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()
        bgr = jitter_background_hsv(bgr, rng)
        # rotate ±15° (expand canvas, keep alpha in sync)
        angle = rng.uniform(-15.0, 15.0)
        h, w = bgr.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        nw, nh = int(round(h * sin + w * cos)), int(round(h * cos + w * sin))
        m[0, 2] += nw / 2.0 - w / 2.0
        m[1, 2] += nh / 2.0 - h / 2.0
        bgr = cv2.warpAffine(bgr, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
        alpha = cv2.warpAffine(alpha, m, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)
        # random size 8–45% of canvas width
        target_w = rng.uniform(0.08, 0.45) * cw
        scale = target_w / max(nw, nh)
        rw, rh = max(8, int(nw * scale)), max(8, int(nh * scale))
        bgr = cv2.resize(bgr, (rw, rh), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, (rw, rh), interpolation=cv2.INTER_NEAREST)
        x = rng.randint(int(-0.3 * rw), int(cw - 0.7 * rw))
        y = rng.randint(int(-0.3 * rh), int(ch - 0.7 * rh))
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(cw, x + rw), min(ch, y + rh)
        if x2 <= x1 or y2 <= y1:
            continue
        sx, sy = x1 - x, y1 - y
        obj = bgr[sy:sy + (y2 - y1), sx:sx + (x2 - x1)]
        a = alpha[sy:sy + (y2 - y1), sx:sx + (x2 - x1)] > 0
        image[y1:y2, x1:x2][a] = obj[a]
        pasted += 1
    return pasted


def sweep_scale(index: int, count: int, min_scale: float) -> float:
    """Scale for image `index` of `count`, stepping evenly from 1.0 (close /
    largest) down to `min_scale` (far / smallest)."""
    if count <= 1:
        return 1.0
    return 1.0 - (index / (count - 1)) * (1.0 - min_scale)


def synthesize_split(
    split: str,
    count: int,
    names: list[str],
    bg_pool_per_class: dict[str, list[Path]],
    rng: random.Random,
    device: torch.device,
    object_profiles: dict,
    progress_start: int,
    progress_end: int,
    start_idx: int = 0,
    sources: dict | None = None,
    min_scale: float | None = None,
    position_cells_logical: list[tuple[int, int]] | None = None,
    pos_rows: int = 6,
    pos_cols: int = 6,
    scale_steps: int | None = None,
    class_steps: dict[str, int] | None = None,
    scale_min: float = 0.166667,
    scale_max: float = 1.0,
) -> int:
    if sources is None:
        sources = _build_sources(names)

    # Shared background pool: intersection of every class's allowed pool.
    # A class with no restriction contributes the full set, so the result is
    # always the most restrictive non-empty subset across restricted classes.
    # Example: class A = all bgs, class B = indoor only → intersection = indoor only.
    pool_sets = [set(bg_pool_per_class[n]) for n in names]
    shared = set.intersection(*pool_sets) if pool_sets else set()
    if not shared:  # conflicting restrictions → fall back to union
        shared = set.union(*pool_sets) if pool_sets else set()
    bg_shared_pool = list(shared)
    if not bg_shared_pool:
        raise RuntimeError("No background images available. Check the bg/ folder.")

    placement_mode    = get_cfg("phase2", "placement_mode", default="random")
    out_of_frame      = float(get_cfg("phase2", "out_of_frame_pct", default=10)) / 100.0
    grid_rows         = max(1, get_cfg("phase2", "grid_rows", default=3))
    grid_cols         = max(1, get_cfg("phase2", "grid_cols", default=3))

    image_output = SYNTH_IMAGE_DIR / split
    label_output = SYNTH_LABEL_DIR / split
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    saved = start_idx
    new_count = count - start_idx
    attempts = 0
    max_attempts = max(new_count * 20, 20)

    while saved < count and attempts < max_attempts:
        attempts += 1
        stem = f"{split}_{saved:06d}"
        # Resume: skip already-generated files
        if (image_output / f"{stem}.jpg").exists():
            saved += 1
            continue

        background_path = rng.choice(bg_shared_pool)
        background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
        if background is None:
            continue
        if get_cfg("phase2", "bg_hsv_jitter", default=True):
            background = jitter_background_hsv(background, rng)
        canvas = canvas_from_background(background, device)
        canvas_h, canvas_w = canvas_shape(canvas)

        # Build grid cells if needed
        grid_cells: list[tuple[int, int, int, int]] | None = None
        if placement_mode == "grid":
            cell_w = canvas_w // grid_cols
            cell_h = canvas_h // grid_rows
            cells = [
                (col * cell_w, row * cell_h, cell_w, cell_h)
                for row in range(grid_rows)
                for col in range(grid_cols)
            ]
            rng.shuffle(cells)
            grid_cells = cells

        # Disjoint placement region for this split (train cells never overlap
        # val/test cells), mapped to the current canvas size.
        position_cells = (
            cells_to_pixels(position_cells_logical, pos_rows, pos_cols, canvas_w, canvas_h)
            if position_cells_logical else None
        )
        num_cells = len(position_cells) if position_cells else 0

        # Systematic position × scale grid. The dataset is the Cartesian product
        # of every scale step (close→far) and every placement cell: image index
        # `saved` decodes to one (scale_idx, pos_idx) pair, so over the whole
        # split each class visits every cell at every scale. When there are at
        # least as many cells as classes, every class gets its own distinct
        # cell this image (offset so no two classes collide and, across pos_idx,
        # each class sweeps all cells). Otherwise we fall back to free choice
        # within the pool for that (small) split.
        systematic = bool(scale_steps and num_cells)
        strict_cells = systematic and num_cells >= len(names)
        if systematic:
            scale_idx = saved // num_cells
            pos_idx = saved % num_cells
        else:
            scale_idx = saved
            pos_idx = 0

        # Per-class forced scale for this image. Each class walks its OWN pixel
        # range (close→far) over its own number of steps; once scale_idx passes
        # a class's last step it has finished sweeping and simply isn't placed in
        # this (or any later) image. forced=None → class not present this image.
        def _forced_scale_for(name: str) -> float | None:
            steps = (class_steps or {}).get(name, scale_steps or 1)
            if scale_idx >= steps:
                return None  # class exhausted its range — drop out
            if steps <= 1:
                return scale_max
            return scale_max - (scale_idx / (steps - 1)) * (scale_max - scale_min)

        present_classes = [n for n in names if _forced_scale_for(n) is not None]

        labels: list[str] = []
        occupied_boxes: list[tuple[int, int, int, int]] = []

        for obj_idx, class_name in enumerate(names):
            forced_scale = _forced_scale_for(class_name)
            if forced_scale is None:
                continue  # this class has finished its sweep — skip it
            class_id     = names.index(class_name)
            object_pair  = rng.choice(sources[class_name])
            object_profile = object_profiles.get(class_name)
            loaded = load_object(*object_pair, device, rng, object_profile)
            if loaded is None:
                continue
            grid_cell = grid_cells[obj_idx % len(grid_cells)] if grid_cells else None
            # Pin this class to its own cell so it walks every position across
            # the split; small splits (fewer cells than classes) keep the pool.
            obj_position_cells = (
                [position_cells[(pos_idx + obj_idx) % num_cells]]
                if strict_cells else position_cells
            )
            pasted = paste_object(
                canvas, loaded[0], loaded[1], rng, occupied_boxes, device,
                object_profile, grid_cell, out_of_frame, 0,
                forced_scale=forced_scale, position_cells=obj_position_cells,
            )
            if pasted is None:
                continue
            canvas, (vx, vy, vw, vh) = pasted
            occupied_boxes.append((vx, vy, vw, vh))
            ch, cw = canvas_shape(canvas)
            labels.append(f"{class_id} {(vx + vw/2)/cw:.6f} {(vy + vh/2)/ch:.6f} {vw/cw:.6f} {vh/ch:.6f}")

        # Keep only fully-populated images: every class that should appear at this
        # scale step must have placed successfully (retry otherwise).
        if not labels or len(labels) != len(present_classes):
            continue

        cv2.imwrite(str(image_output / f"{stem}.jpg"), canvas_to_image(canvas))
        (label_output / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
        saved += 1
        if new_count > 0:
            pct = progress_start + round((progress_end - progress_start) * (saved - start_idx) / new_count)
            set_job(
                stage=f"Synthesizing {split}",
                percent=min(progress_end, pct),
                message=f"({saved}/{count}) → {image_output}/{stem}.jpg",
            )

    new_saved = saved - start_idx
    if new_saved < new_count:
        raise RuntimeError(f"Only generated {new_saved}/{new_count} new {split} image(s). Try smaller object sizes or less restrictive placement settings.")
    return saved


def synthesize_extra(
    split: str,
    class_name: str,
    class_id: int,
    n_needed: int,
    start_idx: int,
    bg_pool: list[Path],
    rng: random.Random,
    device: torch.device,
    object_profile: dict | None,
    sources: dict,
    scale_min: float,
    scale_max: float,
    progress_start: int,
    progress_end: int,
) -> int:
    """Upsample one under-represented class with EXTRA, distinct images (never
    duplicated files). Each image is a fresh solo composite: random background,
    random free position, a random scale within the class's range, plus the
    object augmentation pipeline (brightness/contrast/blur + flip/rotation). The
    pixel content differs every time, so balancing class counts this way adds no
    memorisation/overfitting that copying files would. Returns the next free
    image index."""
    image_output = SYNTH_IMAGE_DIR / split
    label_output = SYNTH_LABEL_DIR / split
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    out_of_frame = float(get_cfg("phase2", "out_of_frame_pct", default=10)) / 100.0

    idx = start_idx
    saved = 0
    attempts = 0
    max_attempts = max(n_needed * 30, 30)
    while saved < n_needed and attempts < max_attempts:
        attempts += 1
        stem = f"{split}_{idx:06d}"
        if (image_output / f"{stem}.jpg").exists():
            idx += 1
            continue
        background = cv2.imread(str(rng.choice(bg_pool)), cv2.IMREAD_COLOR)
        if background is None:
            continue
        if get_cfg("phase2", "bg_hsv_jitter", default=True):
            background = jitter_background_hsv(background, rng)
        canvas = canvas_from_background(background, device)
        object_pair = rng.choice(sources[class_name])
        loaded = load_object(*object_pair, device, rng, object_profile)
        if loaded is None:
            continue
        forced_scale = rng.uniform(scale_min, scale_max)
        pasted = paste_object(
            canvas, loaded[0], loaded[1], rng, [], device,
            object_profile, None, out_of_frame, 0, forced_scale=forced_scale,
        )
        if pasted is None:
            continue
        canvas, (vx, vy, vw, vh) = pasted
        ch, cw = canvas_shape(canvas)
        label = f"{class_id} {(vx + vw/2)/cw:.6f} {(vy + vh/2)/ch:.6f} {vw/cw:.6f} {vh/ch:.6f}"
        cv2.imwrite(str(image_output / f"{stem}.jpg"), canvas_to_image(canvas))
        (label_output / f"{stem}.txt").write_text(label + "\n", encoding="utf-8")
        saved += 1
        idx += 1
        if n_needed > 0:
            pct = progress_start + round((progress_end - progress_start) * saved / n_needed)
            set_job(
                stage=f"Balancing {class_name}",
                percent=min(progress_end, pct),
                message=f"+{class_name} ({saved}/{n_needed}) → {image_output}/{stem}.jpg",
            )
    return idx


def random_split_train_val(val_fraction: float, rng: random.Random) -> tuple[int, int]:
    """Randomly move `val_fraction` of the generated train image/label pairs
    into the validate split. The whole pool is synthesized into train/ first;
    this carves out a random hold-out so train and val are independent samples
    of the same distribution. Returns (train_count, val_count) after the move."""
    train_img, train_lbl = SYNTH_IMAGE_DIR / "train", SYNTH_LABEL_DIR / "train"
    val_img,   val_lbl   = SYNTH_IMAGE_DIR / "validate", SYNTH_LABEL_DIR / "validate"
    for d in (val_img, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    stems = [p.stem for p in train_img.glob("*.jpg")]
    rng.shuffle(stems)
    n_val = round(len(stems) * val_fraction)

    for i, stem in enumerate(stems[:n_val]):
        dst = f"validate_{i:06d}"
        (train_img / f"{stem}.jpg").replace(val_img / f"{dst}.jpg")
        (train_lbl / f"{stem}.txt").replace(val_lbl / f"{dst}.txt")
    return len(stems) - n_val, n_val


def synthesize_dataset(images_per_class: int) -> None:
    total_started_at = time.perf_counter()
    try:
        set_job(running=True, stage="Synthesizing", percent=5,
                message="Preparing synthesized dataset", error=None, log=[])
        ensure_dirs()

        names = sorted(
            d.name for d in PHASE1_DIR.iterdir()
            if d.is_dir() and any(
                p.suffix.lower() in IMAGE_EXTENSIONS
                for p in (d / "images").iterdir() if p.is_file()
            )
        ) if PHASE1_DIR.exists() else []
        if not names:
            raise RuntimeError("No accepted classes found. Complete Phase 0 and accept at least one class first.")
        backgrounds = image_files(BG_DIR)
        if not backgrounds:
            raise RuntimeError(f"No background images found in {BG_DIR}")

        start_idx    = max(0, get_cfg("phase2", "start_idx", default=0))
        train_pct    = get_cfg("phase2", "train_pct",  default=60)
        val_pct      = get_cfg("phase2", "val_pct",    default=40)
        images_per_class = max(1, images_per_class)

        rng = random.Random()

        if start_idx == 0:
            shutil.rmtree(SYNTH_IMAGE_DIR, ignore_errors=True)
            shutil.rmtree(SYNTH_LABEL_DIR, ignore_errors=True)
            SYNTH_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
            SYNTH_LABEL_DIR.mkdir(parents=True, exist_ok=True)

        device = datagen_device()
        object_profiles = load_object_profiles()
        sources = _build_sources(names)
        bg_pool_per_class = _build_bg_pool_per_class(names, backgrounds, object_profiles)
        placement_mode = get_cfg("phase2", "placement_mode", default="random")

        # Per-class distance sweep. Every class is measured from its own Phase-1
        # cutouts (its "close" reference pixel size), then walked down a SHARED
        # scale range [scale_min, scale_max]. Each class therefore has its own
        # pixel range and its own number of 1px steps: a big object (e.g. a tall
        # bottle) gets many steps and appears in many images, a small object
        # (e.g. an inhaler) gets fewer steps and drops out of later images once
        # it has swept its whole range. The pool length is driven by the class
        # with the most steps.
        scale_min = float(get_cfg("phase2", "scale_min", default=0.166667))
        scale_max = float(get_cfg("phase2", "scale_max", default=1.0))
        if scale_max <= scale_min:
            scale_max = scale_min + 1e-3
        class_ref_px = measure_class_reference_px(names, sources, device, rng, object_profiles)
        class_steps = {
            n: max(1, round(class_ref_px[n] * (scale_max - scale_min))) for n in names
        }
        min_scale = scale_min  # kept for downstream (forced-scale clamp / sweep)

        # One shared placement grid for the whole pool — no per-split disjoint
        # cells. Every image may use any cell; train/val independence comes from
        # the random split below, not from where objects are placed.
        pos_rows = max(1, get_cfg("phase2", "position_grid_rows", default=6))
        pos_cols = max(1, get_cfg("phase2", "position_grid_cols", default=6))
        all_cells_logical = [(r, c) for r in range(pos_rows) for c in range(pos_cols)]

        # Synthesize one flat pool (full position × scale coverage), then split
        # it randomly into train/val by the configured ratios (no test set).
        val_fraction = val_pct / max(1, train_pct + val_pct)

        # Full systematic coverage: (longest class's scale steps) × every cell.
        scale_steps = max(class_steps.values())
        pool_count = scale_steps * len(all_cells_logical)
        total_images = pool_count

        append_log(f"Classes: {', '.join(names)}")
        append_log(f"Images to synthesize: {total_images} | up to {len(names)} objects per image")
        append_log(f"Saving images to: {SYNTH_IMAGE_DIR}")
        append_log(f"Saving labels to: {SYNTH_LABEL_DIR}")
        append_log(
            f"Shared scale range: ×{scale_max:.3f} (close) → ×{scale_min:.3f} (far) | "
            f"position grid {pos_rows}×{pos_cols} = {len(all_cells_logical)} cells | "
            f"pool = {scale_steps} scales × {len(all_cells_logical)} cells = {pool_count} image(s)"
        )
        for n in names:
            steps = class_steps[n]
            append_log(
                f"  {n}: ref {class_ref_px[n]}px → range {round(class_ref_px[n]*scale_max)}px"
                f"→{round(class_ref_px[n]*scale_min)}px in {steps} steps "
                f"(appears in {min(steps, scale_steps) * len(all_cells_logical)} image(s), "
                f"then drops out)"
            )
        append_log(f"Device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
        append_log(f"Placement: {placement_mode} | each class sweeps its own pixel range")
        append_log(f"Random split: train {train_pct} / val {val_pct} (no test)" +
                   (f" (resuming from idx {start_idx})" if start_idx > 0 else ""))
        for n in names:
            cats = (object_profiles.get(n) or {}).get("background_categories") or []
            pool_size = len(bg_pool_per_class[n])
            if cats:
                append_log(f"  {n}: bg restricted to {cats} ({pool_size} image(s))")

        kw = dict(names=names, bg_pool_per_class=bg_pool_per_class, rng=rng, device=device,
                  object_profiles=object_profiles, start_idx=start_idx, sources=sources,
                  min_scale=min_scale, pos_rows=pos_rows, pos_cols=pos_cols,
                  scale_steps=scale_steps, class_steps=class_steps,
                  scale_min=scale_min, scale_max=scale_max)

        t0 = time.perf_counter()
        pool_saved = synthesize_split("train", pool_count, progress_start=10, progress_end=70,
                                      position_cells_logical=all_cells_logical, **kw)
        append_log(f"Pool: {pool_saved} image(s) in {format_duration(time.perf_counter() - t0)}.")

        # Balance class counts: the per-class sweep leaves smaller objects under-
        # represented (they drop out earlier). Generate EXTRA distinct images for
        # each minority class — never duplicating files — until every class
        # appears in the same number of images as the majority class.
        class_counts = {
            n: min(class_steps[n], scale_steps) * len(all_cells_logical) for n in names
        }
        balanced_counts = dict(class_counts)
        if get_cfg("phase2", "balance_classes", default=True) and start_idx == 0:
            target = max(class_counts.values())
            deficits = {n: target - class_counts[n] for n in names if target - class_counts[n] > 0}
            total_deficit = sum(deficits.values())
            if total_deficit > 0:
                append_log(
                    f"Balancing minority classes up to {target} image(s) each "
                    f"(+{total_deficit} extra unique image(s), no duplication): "
                    + ", ".join(f"{n}+{d}" for n, d in deficits.items())
                )
                next_idx = pool_count
                done = 0
                for n, deficit in deficits.items():
                    p0 = 70 + round(20 * done / total_deficit)
                    p1 = 70 + round(20 * (done + deficit) / total_deficit)
                    next_idx = synthesize_extra(
                        "train", n, names.index(n), deficit, next_idx,
                        bg_pool_per_class[n], rng, device, object_profiles.get(n),
                        sources, scale_min, scale_max, progress_start=p0, progress_end=p1,
                    )
                    balanced_counts[n] = target
                    done += deficit
                append_log("Balanced per-class image counts: " +
                           ", ".join(f"{n}={c}" for n, c in balanced_counts.items()))

        # Randomly move val_fraction of the generated pool into the validate split.
        set_job(stage="Splitting train/val", percent=92, message="Random train/val split")
        train_saved, val_saved = random_split_train_val(val_fraction, rng)
        test_saved = 0
        append_log(f"Split: {train_saved} train / {val_saved} val (random, no test).")

        write_yaml(names)
        total_elapsed = format_duration(time.perf_counter() - total_started_at)

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "classes": names,
            "images_per_class": images_per_class,
            "total_images": total_images,
            "train_count": train_saved,
            "validate_count": val_saved,
            "test_count": test_saved,
            "placement_mode": placement_mode,
            "max_objects_per_image": len(names),
            "class_reference_px": class_ref_px,
            "class_scale_steps": class_steps,
            "class_image_counts": class_counts,
            "balanced_image_counts": balanced_counts,
            "scale_range": [scale_min, scale_max],
            "augmentation": {
                "hflip_pct": get_cfg("phase2", "hflip_pct", default=50),
                "rotate_max_deg": get_cfg("phase2", "rotate_max_deg", default=15),
                "brightness_variation_pct": get_cfg("phase2", "brightness_variation_pct", default=10),
                "contrast_variation_pct": get_cfg("phase2", "contrast_variation_pct", default=20),
                "saturation": "off (colour cues preserved)",
            },
            "elapsed": total_elapsed,
        }
        (DATA_DIR / "generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        append_log(f"Total: {total_elapsed}.")
        append_log(f"Images saved in: {SYNTH_IMAGE_DIR} (train/ + validate/)")
        set_job(stage="Complete", percent=100,
                message=f"Done in {total_elapsed}: {train_saved} train / {val_saved} val → {SYNTH_IMAGE_DIR}")
    except Exception as exc:
        append_log(str(exc))
        set_job(stage="Error", error=str(exc), message=str(exc))
    finally:
        set_job(running=False)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    ensure_dirs()
    return render_template("index.html")


@app.get("/config")
def config_get():
    return jsonify(load_config())


@app.post("/config")
def config_post():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Expected a JSON object.", "detail": "Body must be a JSON dict."}), 400
    try:
        save_config(data)
        return jsonify({"ok": True, "config": load_config()})
    except Exception as exc:
        return jsonify({"error": "Failed to save configuration.", "detail": str(exc)}), 500


@app.post("/upload")
def upload():
    ensure_dirs()
    files = request.files.getlist("videos")
    if not files:
        return jsonify({"error": "No files were uploaded."}), 400
    saved = skipped = 0
    for fs in files:
        raw_name = fs.filename or ""
        if Path(raw_name).suffix.lower() not in VIDEO_EXTENSIONS:
            skipped += 1
            continue
        out = VIDEO_DIR / safe_uploaded_video_path(raw_name)
        out.parent.mkdir(parents=True, exist_ok=True)
        fs.save(out)
        saved += 1
    if saved == 0:
        return jsonify({"error": "No supported video files were uploaded."}), 400
    set_job(
        id=None, running=False, stage="Uploaded", percent=0,
        message=f"Saved {saved} video file(s).", error=None,
        log=[f"Saved {saved} video file(s), skipped {skipped}."],
    )
    return jsonify({"saved": saved, "skipped": skipped})


@app.post("/start")
def start():
    if snapshot_job()["running"]:
        return jsonify({"error": "A job is already running."}), 409
    default_ipv = get_cfg("phase0", "images_per_video", default=108)
    try:
        raw = request.form.get("images_per_video") or (
            request.json.get("images_per_video") if request.is_json else None
        )
        images_per_video = int(raw) if raw is not None else default_ipv
    except Exception:
        images_per_video = default_ipv
    images_per_video = max(1, images_per_video)
    videos = [p for p in VIDEO_DIR.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    if not videos:
        return jsonify({"error": "Upload videos before starting."}), 400
    job_id = uuid.uuid4().hex
    set_job(id=job_id, running=True, stage="Queued", percent=1, message="Job queued", error=None, log=[])
    threading.Thread(target=process_dataset, args=(images_per_video,), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.post("/synthesize")
def synthesize():
    if snapshot_job()["running"]:
        return jsonify({"error": "A job is already running."}), 409
    try:
        images_per_class = int(get_cfg("phase2", "expected_images_per_class", default=108))
    except Exception:
        images_per_class = 108
    images_per_class = max(1, images_per_class)
    job_id = uuid.uuid4().hex
    set_job(id=job_id, running=True, stage="Queued", percent=1, message="Synthesis queued", error=None, log=[])
    threading.Thread(target=synthesize_dataset, args=(images_per_class,), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.post("/phase2/preview")
def phase2_preview():
    names = sorted(
        d.name for d in PHASE1_DIR.iterdir()
        if d.is_dir() and any(
            p.suffix.lower() in IMAGE_EXTENSIONS
            for p in (d / "images").iterdir() if p.is_file()
        )
    ) if PHASE1_DIR.exists() else []
    if not names:
        return jsonify({"error": "No accepted classes. Complete Phase 0 first."}), 400
    backgrounds = image_files(BG_DIR)
    if not backgrounds:
        return jsonify({"error": f"No background images found in {BG_DIR}"}), 400
    try:
        rng = random.Random()
        device = datagen_device()
        object_profiles = load_object_profiles()
        sources = _build_sources(names)
        bg_pool_per_class = _build_bg_pool_per_class(names, backgrounds, object_profiles)

        placement_mode = get_cfg("phase2", "placement_mode", default="random")
        out_of_frame   = float(get_cfg("phase2", "out_of_frame_pct", default=10)) / 100.0
        grid_rows      = max(1, get_cfg("phase2", "grid_rows", default=3))
        grid_cols      = max(1, get_cfg("phase2", "grid_cols", default=3))

        preview_pool_sets = [set(bg_pool_per_class[n]) for n in names]
        preview_shared = set.intersection(*preview_pool_sets) if preview_pool_sets else set()
        if not preview_shared:
            preview_shared = set.union(*preview_pool_sets) if preview_pool_sets else set()
        preview_bg_pool = list(preview_shared) or backgrounds
        background = cv2.imread(str(rng.choice(preview_bg_pool)), cv2.IMREAD_COLOR)
        if background is None:
            return jsonify({"error": "Could not read background image."}), 500
        if get_cfg("phase2", "bg_hsv_jitter", default=True):
            background = jitter_background_hsv(background, rng)
        canvas = canvas_from_background(background, device)
        canvas_h, canvas_w = canvas_shape(canvas)

        grid_cells = None
        if placement_mode == "grid":
            cell_w = canvas_w // grid_cols
            cell_h = canvas_h // grid_rows
            cells = [
                (col * cell_w, row * cell_h, cell_w, cell_h)
                for row in range(grid_rows) for col in range(grid_cols)
            ]
            rng.shuffle(cells)
            grid_cells = cells

        occupied_boxes: list[tuple[int, int, int, int]] = []
        for obj_idx, class_name in enumerate(names):
            object_pair = rng.choice(sources[class_name])
            loaded = load_object(*object_pair, device, rng, object_profiles.get(class_name))
            if loaded is None:
                continue
            grid_cell = grid_cells[obj_idx % len(grid_cells)] if grid_cells else None
            pasted = paste_object(
                canvas, loaded[0], loaded[1], rng, occupied_boxes, device,
                object_profiles.get(class_name), grid_cell, out_of_frame, 0,
            )
            if pasted is None:
                continue
            canvas, vis_box = pasted
            occupied_boxes.append(vis_box)

        if len(occupied_boxes) != len(names):
            return jsonify({"error": "Could not place one object for every class without overlap. Try smaller object sizes or grid placement."}), 500
        preview_img = canvas_to_image(canvas)
        paste_distractors_preview(preview_img, rng)
        ok, buf = cv2.imencode(".jpg", preview_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return jsonify({"error": "Failed to encode preview image."}), 500
        return jsonify({"image_b64": base64.b64encode(buf.tobytes()).decode("utf-8"),
                        "objects_placed": len(occupied_boxes), "classes": names})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/classes")
def classes():
    names = class_names()
    if not names:
        names = sorted(load_object_profiles().keys())
    return jsonify({"classes": names})


@app.get("/bg/categories")
def bg_categories():
    return jsonify({"categories": _list_bg_categories()})


@app.get("/object-profiles")
def object_profiles_route():
    return jsonify(load_object_profiles())


@app.post("/object-profile/bg-categories")
def object_profile_bg_categories():
    """Patch only the background_categories field of an existing (or new) profile."""
    data = request.get_json(silent=True) or {}
    class_name = secure_filename(str(data.get("class_name", ""))).strip()
    if not class_name:
        return jsonify({"error": "class_name is required."}), 400
    raw_cats = data.get("background_categories")
    background_categories = (
        [str(c) for c in raw_cats if c]
        if isinstance(raw_cats, list)
        else []
    )
    profiles = load_object_profiles()
    profile = profiles.get(class_name, {})
    profile["background_categories"] = background_categories
    profile.setdefault("updated_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    profile["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    profiles[class_name] = profile
    save_object_profiles(profiles)
    return jsonify({"ok": True, "class_name": class_name, "background_categories": background_categories})


@app.post("/object-profiles/distance")
def object_profiles_distance():
    data = request.get_json(silent=True) or {}
    names = uploaded_class_names() or class_names()
    if not names:
        return jsonify({"error": "Upload a class folder before saving the distance profile."}), 400
    try:
        camera_object_distance_cm = float(data.get("camera_object_distance_cm", 0))
        wanted_distance_cm = float(data.get("wanted_distance_cm", 0))
    except (TypeError, ValueError):
        camera_object_distance_cm = wanted_distance_cm = 0.0
    if camera_object_distance_cm <= 0 or wanted_distance_cm <= 0:
        return jsonify({"error": "Enter distance from camera to object and wanted distance in cm."}), 400
    distance_scale = camera_object_distance_cm / wanted_distance_cm
    try:
        width = int(round(float(data.get("width", 640))))
        height = int(round(float(data.get("height", 640))))
    except (TypeError, ValueError):
        width = height = 640
    width = max(2, width)
    height = max(2, height)

    profiles = load_object_profiles()
    saved: list[str] = []
    for class_name in names:
        existing = profiles.get(class_name, {})
        profile = {
            "width": width, "height": height,
            "min_width": width, "max_width": width,
            "min_height": height, "max_height": height,
            "bbox_x": int(existing.get("bbox_x", 0) or 0),
            "bbox_y": int(existing.get("bbox_y", 0) or 0),
            "source_width": int(existing.get("source_width", 0) or 0),
            "source_height": int(existing.get("source_height", 0) or 0),
            "target_width": int(existing.get("target_width", width) or width),
            "target_height": int(existing.get("target_height", height) or height),
            "close_sample": existing.get("close_sample"),
            "far_sample": existing.get("far_sample"),
            "camera_object_distance_cm": round(camera_object_distance_cm, 3),
            "wanted_distance_cm": round(wanted_distance_cm, 3),
            "distance_scale": round(distance_scale, 6),
            "scaled_wanted_distance_cm": round(camera_object_distance_cm / distance_scale, 3),
            "min_distance_cm": existing.get("min_distance_cm"),
            "max_distance_cm": existing.get("max_distance_cm"),
            "object_max_range_cm": existing.get("object_max_range_cm"),
            "brightness": existing.get("brightness", 0.0),
            "background_categories": existing.get("background_categories", []),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        profiles[class_name] = profile
        saved.append(class_name)
    save_object_profiles(profiles)
    return jsonify({"classes": saved, "distance_scale": round(distance_scale, 6), "profiles": profiles})


@app.post("/object-profile")
def object_profile():
    data = request.get_json(silent=True) or {}
    class_name = secure_filename(str(data.get("class_name", ""))).strip()
    try:
        width = int(round(float(data.get("width", 0))))
        height = int(round(float(data.get("height", 0))))
    except Exception:
        width = height = 0
    try:
        source_width  = int(round(float(data.get("source_width", 0))))
        source_height = int(round(float(data.get("source_height", 0))))
        target_width  = int(round(float(data.get("target_width", 640))))
        target_height = int(round(float(data.get("target_height", 640))))
        bbox_x = int(round(float(data.get("bbox_x", 0))))
        bbox_y = int(round(float(data.get("bbox_y", 0))))
    except Exception:
        source_width = source_height = 0
        target_width = target_height = 640
        bbox_x = bbox_y = 0
    if not class_name:
        return jsonify({"error": "Select a class before saving the object profile."}), 400
    if width <= 1 or height <= 1:
        return jsonify({"error": "Enter a valid object profile size before saving."}), 400
    try:
        brightness = float(data.get("brightness", 0))
    except (TypeError, ValueError):
        brightness = 0.0
    raw_cats = data.get("background_categories")
    background_categories = (
        [str(c) for c in raw_cats if c]
        if isinstance(raw_cats, list)
        else []
    )
    close_sample = distance_sample(data, "close_sample")
    far_sample   = distance_sample(data, "far_sample")
    try:
        camera_object_distance_cm = float(data.get("camera_object_distance_cm", 0))
        wanted_distance_cm = float(data.get("wanted_distance_cm", 0))
        min_distance_cm    = float(data.get("min_distance_cm", 0))
        max_distance_cm    = float(data.get("max_distance_cm", 0))
        object_max_range_cm = float(data.get("object_max_range_cm", max_distance_cm))
        min_width  = int(round(float(data.get("min_width",  width))))
        max_width  = int(round(float(data.get("max_width",  width))))
        min_height = int(round(float(data.get("min_height", height))))
        max_height = int(round(float(data.get("max_height", height))))
    except (TypeError, ValueError):
        camera_object_distance_cm = wanted_distance_cm = min_distance_cm = max_distance_cm = object_max_range_cm = 0.0
        min_width = max_width = width
        min_height = max_height = height
    if camera_object_distance_cm <= 0 or wanted_distance_cm <= 0:
        return jsonify({"error": "Enter distance from camera to object and wanted distance in cm."}), 400
    distance_scale = camera_object_distance_cm / wanted_distance_cm
    profile = {
        "width": width, "height": height,
        "min_width": max(2, min_width), "max_width": max(2, max_width),
        "min_height": max(2, min_height), "max_height": max(2, max_height),
        "bbox_x": bbox_x, "bbox_y": bbox_y,
        "source_width": source_width, "source_height": source_height,
        "target_width": target_width, "target_height": target_height,
        "close_sample": close_sample, "far_sample": far_sample,
        "camera_object_distance_cm": round(camera_object_distance_cm, 3),
        "wanted_distance_cm": round(wanted_distance_cm, 3),
        "distance_scale": round(distance_scale, 6),
        "min_distance_cm": round(min_distance_cm, 3) if min_distance_cm > 0 else None,
        "max_distance_cm": round(max_distance_cm, 3) if max_distance_cm > 0 else None,
        "object_max_range_cm": round(object_max_range_cm, 3) if object_max_range_cm > 0 else None,
        "brightness": round(max(0.0, min(1.0, brightness)), 6),
        "background_categories": background_categories,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    profiles = load_object_profiles()
    profiles[class_name] = profile
    save_object_profiles(profiles)
    return jsonify({"class_name": class_name, "profile": profile, "profiles": profiles})


@app.get("/progress")
def progress():
    return jsonify(snapshot_job())


@app.get("/events")
def events():
    def stream():
        last_payload = None
        while True:
            payload = json.dumps(snapshot_job())
            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload
            time.sleep(0.25 if snapshot_job()["running"] else 0.5)

    resp = Response(stream(), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


@app.post("/clear")
def clear():
    if snapshot_job()["running"]:
        return jsonify({"error": "Cannot clear while a job is running."}), 409
    if YOLO_IMAGE_ALIAS_DIR.is_symlink():
        YOLO_IMAGE_ALIAS_DIR.unlink()
    for path in (VIDEO_DIR, IMAGE_DIR, MASK_DIR, REMBG_DIR, SYNTH_IMAGE_DIR, SYNTH_LABEL_DIR, YOLO_IMAGE_ALIAS_DIR):
        shutil.rmtree(path, ignore_errors=True)
    if YOLO_DATASET_YAML.exists():
        YOLO_DATASET_YAML.unlink()
    with _class_status_lock:
        _class_status.clear()
    ensure_dirs()
    set_job(
        id=None, running=False, stage="Idle", percent=0,
        message="Cleared dataset folders.", error=None,
        log=["Cleared dataset folders."], frame_count_warning=None,
    )
    return jsonify({"ok": True})


# ── Phase 0 routes ────────────────────────────────────────────────────────────

@app.get("/phase0/status")
def phase0_status():
    explicit = get_class_statuses()
    result = {}
    for cls in all_known_classes():
        if cls in explicit:
            result[cls] = explicit[cls]
        else:
            result[cls] = infer_class_status(cls)
    return jsonify(result)


@app.get("/phase0/review/<class_name>")
def phase0_review(class_name: str):
    class_name = secure_filename(class_name)
    rembg_class_dir = find_class_dir(REMBG_DIR, class_name)
    if rembg_class_dir is None:
        return jsonify({
            "error": f"No processed images for '{class_name}'. Run Process Videos first.",
        }), 404

    files = sorted(
        p for p in rembg_class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        return jsonify({"error": f"No images found for class '{class_name}'."}), 404

    n = len(files)
    count = min(9, n)
    if n <= 9:
        selected = files
    else:
        selected = [files[round(i * (n - 1) / (count - 1))] for i in range(count)]

    pairs = []
    mask_class_dir = find_class_dir(MASK_DIR, class_name)
    for img_path in selected:
        # After rename_to_convention, mask has same name in the mask class dir
        mask_path: Path | None = None
        if mask_class_dir is not None:
            candidate = mask_class_dir / img_path.name
            if candidate.exists():
                mask_path = candidate
            else:
                cands = list(mask_class_dir.glob(f"{img_path.stem}.*"))
                mask_path = cands[0] if cands else None

        data = make_review_composite(img_path, mask_path)
        if data is None:
            continue
        pairs.append({
            "filename": img_path.name,
            "image_b64": base64.b64encode(data).decode("utf-8"),
        })

    return jsonify({"class_name": class_name, "total_files": n, "pairs": pairs})


@app.post("/phase0/accept/<class_name>")
def phase0_accept(class_name: str):
    class_name = secure_filename(class_name)
    rembg_class_dir = find_class_dir(REMBG_DIR, class_name)
    mask_class_dir  = find_class_dir(MASK_DIR, class_name)

    if rembg_class_dir is None:
        return jsonify({"error": f"No processed files for '{class_name}'. Run Process Videos first."}), 404

    img_files = sorted(
        p for p in rembg_class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not img_files:
        return jsonify({"error": f"No images to accept for '{class_name}'."}), 404

    dst_images = PHASE1_DIR / class_name / "images"
    dst_masks  = PHASE1_DIR / class_name / "masks"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_masks.mkdir(parents=True, exist_ok=True)

    for src in img_files:
        shutil.copy2(src, dst_images / src.name)

    mask_files: list[Path] = []
    if mask_class_dir.exists():
        mask_files = sorted(
            p for p in mask_class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        for src in mask_files:
            shutil.copy2(src, dst_masks / src.name)

    # Count original extracted frames for the report
    img_src_dir = find_class_dir(IMAGE_DIR, class_name)
    extracted_count = (
        len([p for p in img_src_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
        if img_src_dir is not None else 0
    )

    set_class_status(class_name, "Done")

    report = {
        "class_name": class_name,
        "frames_extracted": extracted_count,
        "frames_retained": len(img_files),
        "masks_generated": len(mask_files),
        "frame_count_warning": snapshot_job().get("frame_count_warning"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (PHASE1_DIR / class_name / "phase0_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    return jsonify({
        "ok": True,
        "class_name": class_name,
        "copied_images": len(img_files),
        "copied_masks": len(mask_files),
        "report": report,
    })


# ── Phase 1 routes ────────────────────────────────────────────────────────────

_CONVENTION_RE = re.compile(r'^(.+)_(00|30|45)_(\d+)\.png$', re.IGNORECASE)


def _count_backgrounds() -> int:
    if not BG_DIR.exists():
        return 0
    return sum(
        1 for p in BG_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _validate_class(class_name: str) -> dict:
    img_dir = PHASE1_DIR / class_name / "images"
    mask_dir = PHASE1_DIR / class_name / "masks"
    report_path = PHASE1_DIR / class_name / "phase0_report.json"
    target = get_cfg("phase1", "expected_images_per_class", default=108)

    images = sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ) if img_dir.exists() else []
    total_images = len(images)

    tilt_counts = {"00": 0, "30": 0, "45": 0}
    bad_names: list[str] = []
    for p in images:
        m = _CONVENTION_RE.match(p.name)
        if m and m.group(1) == class_name:
            tilt = m.group(2)
            if tilt in tilt_counts:
                tilt_counts[tilt] += 1
        else:
            bad_names.append(p.name)
    convention_ok = len(bad_names) == 0

    masks = sorted(
        p for p in mask_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ) if mask_dir.exists() else []
    total_masks = len(masks)

    mask_bad: list[str] = []
    for mp in masks[:20]:
        try:
            img = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if img is None or not np.any(img < 128):
                mask_bad.append(mp.name)
        except Exception:
            mask_bad.append(mp.name)

    report_data: dict = {}
    if report_path.exists():
        try:
            report_data = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    if total_images == 0 or total_masks == 0:
        overall = "error"
    elif not convention_ok or total_images < target or mask_bad:
        overall = "warn"
    else:
        overall = "ok"

    return {
        "class_name": class_name,
        "total_images": total_images,
        "target_images": target,
        "tilt_counts": tilt_counts,
        "convention_ok": convention_ok,
        "bad_name_count": len(bad_names),
        "total_masks": total_masks,
        "masks_ok": total_masks > 0,
        "mask_integrity_ok": len(mask_bad) == 0,
        "bad_mask_samples": mask_bad[:3],
        "overall": overall,
        "report": report_data,
    }


@app.get("/phase1/validate")
def phase1_validate():
    if not PHASE1_DIR.exists():
        return jsonify({"classes": [], "total_classes": 0, "backgrounds": 0, "min_backgrounds": 15, "backgrounds_ok": False})
    class_dirs = sorted(d for d in PHASE1_DIR.iterdir() if d.is_dir())
    results = [_validate_class(d.name) for d in class_dirs]
    bg_count = _count_backgrounds()
    min_bg = get_cfg("phase1", "min_backgrounds_per_category", default=15)
    return jsonify({
        "classes": results,
        "total_classes": len(results),
        "backgrounds": bg_count,
        "min_backgrounds": min_bg,
        "backgrounds_ok": bg_count >= min_bg,
    })


@app.post("/phase1/rename/<class_name>")
def phase1_rename(class_name: str):
    class_name = secure_filename(class_name)
    img_dir = PHASE1_DIR / class_name / "images"
    mask_dir = PHASE1_DIR / class_name / "masks"
    if not img_dir.exists():
        return jsonify({"error": f"No phase1 images for '{class_name}'."}), 404
    images = sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    n = len(images)
    if n == 0:
        return jsonify({"error": "No images to rename."}), 404
    tilts = ["00", "30", "45"]
    b0, b1 = n // 3, 2 * n // 3
    renamed_images = renamed_masks = 0
    for i, img_path in enumerate(images):
        grp = 0 if i < b0 else (1 if i < b1 else 2)
        within = i if i < b0 else (i - b0 if i < b1 else i - b1)
        new_name = f"{class_name}_{tilts[grp]}_{within + 1:03d}.png"
        if img_path.name != new_name:
            mask_src = mask_dir / img_path.name
            if mask_src.exists():
                mask_src.rename(mask_dir / new_name)
                renamed_masks += 1
            img_path.rename(img_dir / new_name)
            renamed_images += 1
    return jsonify({"ok": True, "class_name": class_name, "renamed_images": renamed_images, "renamed_masks": renamed_masks})


# ── Phase 3: training ─────────────────────────────────────────────────────────

def snapshot_phase3() -> dict:
    with phase3_lock:
        return dict(_phase3_job)


def _parse_training_results(results_csv: Path) -> tuple[float | None, dict]:
    try:
        rows = list(csv.DictReader(results_csv.read_text(encoding="utf-8").splitlines()))
        if not rows:
            return None, {}
        last = {k.strip(): v.strip() for k, v in rows[-1].items() if k}
        cls_val = last.get("val/cls_loss")
        return (float(cls_val) if cls_val else None), last
    except Exception:
        return None, {}


def train_model() -> None:
    global _phase3_process
    started_at = time.perf_counter()
    try:
        with phase3_lock:
            _phase3_job.update(
                running=True, stage="Starting", epoch=0, total_epochs=0,
                current_cls_loss=None, cls_loss=None, quality=None, best_pt=None, error=None,
            )
        set_job(running=True, stage="Training", percent=5,
                message="Starting YOLO training", error=None, log=[])

        if not YOLO_DATASET_YAML.exists():
            raise RuntimeError("Dataset YAML not found. Complete Phase 2 (Synthesize) first.")

        epochs   = int(get_cfg("phase3", "epochs",  default=100))
        batch    = int(get_cfg("phase3", "batch",   default=16))
        imgsz    = int(get_cfg("phase3", "imgsz",   default=640))
        model    = str(get_cfg("phase3", "model",   default="yolo11n.pt"))
        device   = str(get_cfg("phase3", "device",  default="0"))
        proj_rel = str(get_cfg("phase3", "project", default="runs/detect"))
        name     = str(get_cfg("phase3", "name",    default="train"))
        proj_abs = str(PROJECT_DIR / proj_rel)
        # rect=True keeps the native 1280×720 aspect (padded to a 32-multiple,
        # ~1280×736) instead of forcing a square imgsz×imgsz, so far/small
        # objects keep their pixels without wasting compute on padding.
        rect = bool(get_cfg("phase3", "rect", default=False))

        with phase3_lock:
            _phase3_job["total_epochs"] = epochs

        yolo_exe = shutil.which("yolo") or "yolo"
        cmd = [
            yolo_exe, "train",
            f"data={YOLO_DATASET_YAML}",
            f"model={model}",
            f"epochs={epochs}",
            f"batch={batch}",
            f"imgsz={imgsz}",
            f"device={device}",
            f"project={proj_abs}",
            f"name={name}",
            f"rect={rect}",
            "exist_ok=True",
        ]
        append_log(f"$ {' '.join(cmd)}")
        set_job(stage="Training", percent=10, message="YOLO training started")

        epoch_re = re.compile(r'^\s*(\d+)/(\d+)\s+[\d.]+\S*\s+([\d.]+)\s+([\d.]+)')

        process = subprocess.Popen(
            cmd, cwd=PROJECT_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        _phase3_process = process

        for line in process.stdout:
            line = line.rstrip()
            if not line:
                continue
            append_log(line)
            m = epoch_re.match(line)
            if m:
                ep = int(m.group(1))
                tot = int(m.group(2))
                cls_live = float(m.group(4))
                pct = 10 + round(85 * ep / max(tot, 1))
                with phase3_lock:
                    _phase3_job["epoch"] = ep
                    _phase3_job["total_epochs"] = tot
                    _phase3_job["current_cls_loss"] = cls_live
                set_job(
                    stage=f"Training {ep}/{tot}",
                    percent=min(95, pct),
                    message=f"Epoch {ep}/{tot}  cls_loss={cls_live:.4f}",
                )

        rc = process.wait()
        _phase3_process = None

        if rc not in (0, -9, -15):
            raise RuntimeError(f"YOLO training exited with code {rc}")

        # Find best.pt
        best_pt: Path | None = PROJECT_DIR / proj_rel / name / "weights" / "best.pt"
        if not best_pt.exists():
            candidates = sorted(
                (PROJECT_DIR / proj_rel).glob(f"{name}*/weights/best.pt"),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            best_pt = candidates[0] if candidates else None

        # Parse results.csv
        results_csv_path = PROJECT_DIR / proj_rel / name / "results.csv"
        if not results_csv_path.exists() and best_pt:
            results_csv_path = best_pt.parent.parent / "results.csv"
        final_cls_loss, _ = (
            _parse_training_results(results_csv_path) if results_csv_path.exists() else (None, {})
        )

        exc_thr = float(get_cfg("phase3", "cls_loss_excellent",  default=0.5))
        acc_thr = float(get_cfg("phase3", "cls_loss_acceptable", default=1.0))
        quality: str | None = None
        if final_cls_loss is not None:
            if final_cls_loss < exc_thr:
                quality = "Excellent"
            elif final_cls_loss < acc_thr:
                quality = "Acceptable"
            else:
                quality = "Poor"

        elapsed = format_duration(time.perf_counter() - started_at)
        with phase3_lock:
            _phase3_job.update(
                running=False, stage="Complete",
                best_pt=str(best_pt) if best_pt else None,
                cls_loss=final_cls_loss,
                quality=quality,
            )

        msg = f"Training complete in {elapsed}"
        if final_cls_loss is not None:
            msg += f"  val/cls_loss={final_cls_loss:.4f}  [{quality}]"
        append_log(msg)
        set_job(stage="Complete", percent=100, message=msg)

    except Exception as exc:
        err_msg = str(exc)
        append_log(err_msg)
        set_job(stage="Error", error=err_msg, message=err_msg)
        with phase3_lock:
            _phase3_job.update(running=False, stage="Error", error=err_msg)
    finally:
        set_job(running=False)
        with phase3_lock:
            _phase3_job["running"] = False
        _phase3_process = None


@app.post("/phase3/train")
def phase3_train():
    if snapshot_job()["running"] or snapshot_phase3()["running"]:
        return jsonify({"error": "A job is already running."}), 409
    threading.Thread(target=train_model, daemon=True).start()
    return jsonify({"ok": True})


@app.get("/phase3/status")
def phase3_status():
    return jsonify(snapshot_phase3())


@app.post("/phase3/stop")
def phase3_stop():
    global _phase3_process
    proc = _phase3_process
    if proc is not None:
        try:
            proc.terminate()
        except Exception:
            pass
    return jsonify({"ok": True})


@app.get("/phase3/best_pt")
def phase3_best_pt():
    best = snapshot_phase3().get("best_pt")
    if not best:
        return jsonify({"error": "No trained model available. Run training first."}), 404
    path = Path(best)
    if not path.exists():
        return jsonify({"error": "best.pt not found on disk."}), 404
    return send_file(str(path), as_attachment=True, download_name="best.pt")


# ── Phase 4: video evaluation ─────────────────────────────────────────────────

def snapshot_phase4() -> dict:
    with phase4_lock:
        return dict(_phase4_job)


def evaluate_video(video_path: Path, model_path: Path) -> None:
    started_at = time.perf_counter()
    _phase4_stop.clear()
    cap = None
    try:
        with phase4_lock:
            _phase4_job.update(
                running=True, stage="Loading model", frame=0, total_frames=0,
                results_count=0, csv_path=None, error=None, summary=None,
            )
        set_job(running=True, stage="Evaluating", percent=5,
                message="Loading model…", error=None, log=[])

        try:
            from ultralytics import YOLO as _YOLO
            model = _YOLO(str(model_path))
        except Exception as exc:
            raise RuntimeError(f"Failed to load model: {exc}") from exc

        append_log(f"Model loaded: {model_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path.name}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_fps = max(1, int(get_cfg("phase4", "sample_fps", default=1)))
        frame_interval = max(1, round(fps / sample_fps))
        sample_count = max(1, total_frames // frame_interval)

        append_log(
            f"Video: {video_path.name}  FPS={fps:.1f}  frames={total_frames}  samples≈{sample_count}"
        )

        with phase4_lock:
            _phase4_job.update(stage="Evaluating", total_frames=sample_count)

        PHASE4_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = PHASE4_DIR / "video_evaluation.csv"

        rows: list[dict] = []
        frame_idx = 0
        sample_idx = 0

        with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
            fieldnames = ["time_sec", "class_name", "confidence", "x1", "y1", "x2", "y2"]
            writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
            writer.writeheader()

            while True:
                if _phase4_stop.is_set():
                    append_log("Evaluation stopped by user.")
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    time_sec = round(frame_idx / fps, 3)
                    results = model(frame, verbose=False)
                    for r in results:
                        if r.boxes is not None:
                            for box in r.boxes:
                                cls_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                coords = [round(v, 1) for v in box.xyxy[0].tolist()]
                                cls_name = model.names.get(cls_id, str(cls_id))
                                row = {
                                    "time_sec": time_sec,
                                    "class_name": cls_name,
                                    "confidence": round(conf, 4),
                                    "x1": coords[0], "y1": coords[1],
                                    "x2": coords[2], "y2": coords[3],
                                }
                                rows.append(row)
                                writer.writerow(row)

                    sample_idx += 1
                    pct = 10 + round(85 * sample_idx / max(sample_count, 1))
                    with phase4_lock:
                        _phase4_job["frame"] = sample_idx
                        _phase4_job["results_count"] = len(rows)
                    set_job(
                        stage=f"Evaluating {sample_idx}/{sample_count}",
                        percent=min(95, pct),
                        message=f"t={time_sec:.1f}s — {len(rows)} detection(s) so far",
                    )
                frame_idx += 1

        per_class: dict[str, int] = {}
        for row in rows:
            per_class[row["class_name"]] = per_class.get(row["class_name"], 0) + 1

        summary = {
            "total_detections": len(rows),
            "per_class": per_class,
            "seconds_sampled": sample_idx,
            "duration_sec": round(total_frames / fps, 2) if fps > 0 else 0,
        }

        elapsed = format_duration(time.perf_counter() - started_at)
        msg = (
            f"Evaluation complete in {elapsed}: "
            f"{len(rows)} detection(s) across {sample_idx} sample(s)"
        )
        append_log(msg)

        with phase4_lock:
            _phase4_job.update(
                running=False, stage="Complete",
                csv_path=str(csv_path),
                results_count=len(rows),
                summary=summary,
            )
        set_job(stage="Complete", percent=100, message=msg)

    except Exception as exc:
        err_msg = str(exc)
        append_log(err_msg)
        set_job(stage="Error", error=err_msg, message=err_msg)
        with phase4_lock:
            _phase4_job.update(running=False, stage="Error", error=err_msg)
    finally:
        if cap is not None:
            cap.release()
        set_job(running=False)
        with phase4_lock:
            _phase4_job["running"] = False


@app.post("/phase4/evaluate")
def phase4_evaluate():
    if snapshot_job()["running"] or snapshot_phase3()["running"] or snapshot_phase4()["running"]:
        return jsonify({"error": "A job is already running."}), 409

    video_file = request.files.get("video")
    model_file = request.files.get("model")

    if not video_file or not video_file.filename:
        return jsonify({"error": "No video file provided."}), 400
    if Path(video_file.filename).suffix.lower() not in VIDEO_EXTENSIONS:
        return jsonify({"error": "Unsupported video format."}), 400

    PHASE4_DIR.mkdir(parents=True, exist_ok=True)
    video_path = PHASE4_DIR / secure_filename(video_file.filename)
    video_file.save(str(video_path))

    if model_file and model_file.filename:
        model_path = PHASE4_DIR / secure_filename(model_file.filename)
        model_file.save(str(model_path))
    else:
        best = snapshot_phase3().get("best_pt")
        if not best:
            return jsonify({
                "error": "No trained model available. Train a model in Phase 3, or upload a custom model."
            }), 400
        model_path = Path(best)
        if not model_path.exists():
            return jsonify({
                "error": "Phase 3 best.pt not found on disk. Retrain or upload a custom model."
            }), 400

    threading.Thread(target=evaluate_video, args=(video_path, model_path), daemon=True).start()
    return jsonify({"ok": True})


@app.get("/phase4/status")
def phase4_status():
    return jsonify(snapshot_phase4())


@app.post("/phase4/stop")
def phase4_stop_route():
    _phase4_stop.set()
    return jsonify({"ok": True})


@app.get("/phase4/results")
def phase4_results():
    csv_p = snapshot_phase4().get("csv_path")
    if not csv_p:
        return jsonify({"error": "No evaluation results. Run evaluation first."}), 404
    path = Path(csv_p)
    if not path.exists():
        return jsonify({"error": "Results CSV not found on disk."}), 404
    return send_file(str(path), as_attachment=True, download_name="video_evaluation.csv")


# ── Phase 4: ROS live detection ───────────────────────────────────────────────

_ros_job_lock = threading.Lock()
_ros_job: dict = {
    "running": False,
    "stage": "Idle",
    "frame": 0,
    "results_count": 0,
    "csv_path": None,
    "error": None,
    "last_detections": [],
}
_ros_stop = threading.Event()


def snapshot_ros() -> dict:
    with _ros_job_lock:
        return dict(_ros_job)


def run_ros_detection(model_path: Path) -> None:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge

    _ros_stop.clear()
    with _ros_job_lock:
        _ros_job.update(
            running=True, stage="Loading model", frame=0,
            results_count=0, csv_path=None, error=None, last_detections=[],
        )
    set_job(running=True, stage="ROS Detection", percent=5,
            message="Loading model…", error=None, log=[])

    try:
        from ultralytics import YOLO as _YOLO
        model = _YOLO(str(model_path))
    except Exception as exc:
        err = f"Failed to load model: {exc}"
        append_log(err)
        with _ros_job_lock:
            _ros_job.update(running=False, stage="Error", error=err)
        set_job(running=False, stage="Error", error=err, message=err)
        return

    append_log(f"Model loaded: {model_path.name}")

    PHASE4_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PHASE4_DIR / "ros_detection.csv"

    conf_threshold = float(get_cfg("phase4", "confidence_threshold", default=0.75))
    bridge = CvBridge()
    frame_count = 0
    rows: list[dict] = []

    try:
        rclpy.init(args=None)
    except Exception:
        pass  # already initialized

    class DetectorNode(Node):
        def __init__(self):
            super().__init__("skuba_detector")
            self.subscription = self.create_subscription(
                Image, ROS_IMAGE_TOPIC, self._callback, 10,
            )
            self._fcsv = open(csv_path, "w", newline="", encoding="utf-8")
            fieldnames = ["frame", "timestamp", "class_name", "confidence", "x1", "y1", "x2", "y2"]
            self._writer = csv.DictWriter(self._fcsv, fieldnames=fieldnames)
            self._writer.writeheader()
            self._frame = 0
            self._rows: list[dict] = []

        def _callback(self, msg: Image):
            if _ros_stop.is_set():
                return
            try:
                frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception as exc:
                append_log(f"cv_bridge error: {exc}")
                return

            self._frame += 1
            ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            results = model(frame, verbose=False)
            detections = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < conf_threshold:
                        continue
                    cls_id = int(box.cls[0])
                    cls_name = model.names.get(cls_id, str(cls_id))
                    coords = [round(v, 1) for v in box.xyxy[0].tolist()]
                    row = {
                        "frame": self._frame,
                        "timestamp": round(ts, 4),
                        "class_name": cls_name,
                        "confidence": round(conf, 4),
                        "x1": coords[0], "y1": coords[1],
                        "x2": coords[2], "y2": coords[3],
                    }
                    self._rows.append(row)
                    self._writer.writerow(row)
                    detections.append({"class_name": cls_name, "confidence": round(conf, 4),
                                       "x1": coords[0], "y1": coords[1],
                                       "x2": coords[2], "y2": coords[3]})

            self._fcsv.flush()
            with _ros_job_lock:
                _ros_job["frame"] = self._frame
                _ros_job["results_count"] = len(self._rows)
                _ros_job["last_detections"] = detections
            set_job(
                stage=f"Live detection (frame {self._frame})",
                percent=50,
                message=f"Frame {self._frame} — {len(self._rows)} detection(s) total",
            )

        def shutdown(self):
            self._fcsv.close()

    node = DetectorNode()
    append_log(f"Subscribed to {ROS_IMAGE_TOPIC} — streaming detections (conf≥{conf_threshold})")
    with _ros_job_lock:
        _ros_job.update(stage="Running", csv_path=str(csv_path))
    set_job(stage="ROS Detection running", percent=20,
            message=f"Listening on {ROS_IMAGE_TOPIC}…")

    try:
        while not _ros_stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except Exception as exc:
        append_log(f"ROS spin error: {exc}")
    finally:
        node.shutdown()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

    with _ros_job_lock:
        n = _ros_job["results_count"]
        _ros_job.update(running=False, stage="Complete")
    msg_done = f"ROS detection stopped — {n} detection(s) saved to {csv_path}"
    append_log(msg_done)
    set_job(running=False, stage="Complete", percent=100, message=msg_done)


@app.post("/phase4/ros/start")
def phase4_ros_start():
    if snapshot_ros()["running"]:
        return jsonify({"error": "ROS detection is already running."}), 409
    if snapshot_job()["running"] or snapshot_phase3()["running"] or snapshot_phase4()["running"]:
        return jsonify({"error": "Another job is already running."}), 409

    model_file = request.files.get("model")
    if model_file and model_file.filename:
        PHASE4_DIR.mkdir(parents=True, exist_ok=True)
        model_path = PHASE4_DIR / secure_filename(model_file.filename)
        model_file.save(str(model_path))
    elif ROS_MODEL_PATH.exists():
        model_path = ROS_MODEL_PATH
    else:
        best = snapshot_phase3().get("best_pt")
        if not best or not Path(best).exists():
            return jsonify({"error": f"Model not found at {ROS_MODEL_PATH}. Train a model first or place best(1).pt in the project root."}), 400
        model_path = Path(best)

    threading.Thread(target=run_ros_detection, args=(model_path,), daemon=True).start()
    return jsonify({"ok": True, "model": str(model_path), "topic": ROS_IMAGE_TOPIC})


@app.post("/phase4/ros/stop")
def phase4_ros_stop():
    _ros_stop.set()
    return jsonify({"ok": True})


@app.get("/phase4/ros/status")
def phase4_ros_status():
    return jsonify(snapshot_ros())


@app.get("/phase4/ros/results")
def phase4_ros_results():
    csv_p = snapshot_ros().get("csv_path")
    if not csv_p:
        return jsonify({"error": "No ROS detection results yet."}), 404
    path = Path(csv_p)
    if not path.exists():
        return jsonify({"error": "Results CSV not found on disk."}), 404
    return send_file(str(path), as_attachment=True, download_name="ros_detection.csv")


@app.get("/api/health")
def api_health():
    cfg = load_config()
    errors: list[str] = []
    warnings: list[str] = []
    info: dict = {}

    # Backgrounds
    all_bg = image_files(BG_DIR) if BG_DIR.exists() else []
    cats = _list_bg_categories()
    cat_counts: dict[str, int] = {}
    for cat in cats:
        n = len(image_files(BG_DIR / cat))
        cat_counts[cat] = n
        if n < 15:
            warnings.append(f"Background category '{cat}' has {n} image(s) — aim for 15+ to avoid repetition")
    flat_bg = [f for f in all_bg if f.parent == BG_DIR]
    if not all_bg:
        errors.append("No background images found in assets/backgrounds/ — add .jpg/.png files there")
    info["backgrounds"] = {"total": len(all_bg), "flat": len(flat_bg), "categories": cat_counts}

    # Model weights
    model_rel = (cfg.get("phase3") or {}).get("model", "")
    model_path = (PROJECT_DIR / model_rel) if model_rel and not Path(model_rel).is_absolute() else Path(model_rel or "")
    if not model_rel:
        errors.append("config.yaml phase3.model is empty")
    elif not model_path.exists():
        errors.append(f"Model weights not found: {model_rel} (run setup.sh or download manually)")
    info["model"] = {"path": model_rel, "exists": model_path.exists()}

    # Required scripts
    for label, path in [("frame_stride.py", FRAME_STRIDE_SCRIPT), ("rembg.py", REMBG_SCRIPT)]:
        if not path.exists():
            errors.append(f"{label} missing at {path}")
    info["scripts_ok"] = FRAME_STRIDE_SCRIPT.exists() and REMBG_SCRIPT.exists()

    # Config split percentages
    p2 = cfg.get("phase2") or {}
    total_pct = (p2.get("train_pct") or 0) + (p2.get("val_pct") or 0) + (p2.get("test_pct") or 0)
    if total_pct != 100:
        errors.append(f"train_pct + val_pct + test_pct = {total_pct} (must equal 100)")
    info["split_pct"] = {"train": p2.get("train_pct"), "val": p2.get("val_pct"), "test": p2.get("test_pct"), "total": total_pct}

    # Phase 1 classes and their cutouts
    classes = sorted(d.name for d in PHASE1_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")) if PHASE1_DIR.exists() else []
    class_info: dict[str, dict] = {}
    for cls in classes:
        cutouts = len(image_files(REMBG_DIR / cls)) if (REMBG_DIR / cls).exists() else 0
        if cutouts == 0:
            warnings.append(f"Class '{cls}' has no background-removed cutouts — run Phase 1 (Background Removal)")
        class_info[cls] = {"cutouts": cutouts}
    info["classes"] = class_info

    # ffmpeg
    import shutil as _shutil
    ffmpeg_ok = bool(_shutil.which("ffmpeg")) or (PROJECT_DIR / "bin" / "ffmpeg").exists()
    if not ffmpeg_ok:
        warnings.append("ffmpeg not found — CUDA-accelerated frame extraction unavailable (CPU fallback will be used)")
    info["ffmpeg"] = ffmpeg_ok

    ok = len(errors) == 0
    return jsonify({"ok": ok, "errors": errors, "warnings": warnings, "info": info})


if __name__ == "__main__":
    ensure_config_defaults()
    load_config()
    ensure_dirs()
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
