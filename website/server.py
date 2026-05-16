from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_functional


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
VIDEO_DIR = BASE_DIR / "dataset_video"
IMAGE_DIR = BASE_DIR / "dataset_image"
MASK_DIR = BASE_DIR / "dataset_mask"
REMBG_DIR = BASE_DIR / "dataset_rembg"
BG_DIR = PROJECT_DIR / "bg"
DATA_DIR = BASE_DIR / "data"
SYNTH_IMAGE_DIR = DATA_DIR / "sythesized_data"
YOLO_IMAGE_ALIAS_DIR = DATA_DIR / "images"
SYNTH_LABEL_DIR = DATA_DIR / "labels"
YOLO_DATASET_YAML = DATA_DIR / "sythesized_data.yaml"
OBJECT_PROFILE_JSON = DATA_DIR / "object_profiles.json"
FRAME_STRIDE_SCRIPT = BASE_DIR / "frame_stride.py"
REMBG_SCRIPT = PROJECT_DIR / "mask" / "rembg.py"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
BRIGHTNESS_PROFILE_VARIATION = 0.10

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024 * 1024

job_lock = threading.Lock()
job = {
    "id": None,
    "running": False,
    "stage": "Idle",
    "percent": 0,
    "message": "Upload a folder of videos to begin.",
    "log": [],
    "error": None,
}


def ensure_dirs() -> None:
    for path in (VIDEO_DIR, IMAGE_DIR, MASK_DIR, REMBG_DIR, DATA_DIR, SYNTH_IMAGE_DIR, SYNTH_LABEL_DIR):
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


def load_object_profiles() -> dict:
    if not OBJECT_PROFILE_JSON.exists():
        return {}
    try:
        return json.loads(OBJECT_PROFILE_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_object_profiles(profiles: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OBJECT_PROFILE_JSON.write_text(json.dumps(profiles, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        )
        ensure_dirs()
        shutil.rmtree(IMAGE_DIR, ignore_errors=True)
        shutil.rmtree(MASK_DIR, ignore_errors=True)
        shutil.rmtree(REMBG_DIR, ignore_errors=True)
        ensure_dirs()

        run_step(
            "Frame stride",
            20,
            [
                sys.executable,
                str(FRAME_STRIDE_SCRIPT),
                "--input",
                str(VIDEO_DIR),
                "--output",
                str(IMAGE_DIR),
                "--images-per-video",
                str(images_per_video),
                "--device",
                "cuda",
            ],
        )
        set_job(stage="Frame stride", percent=55, message="Frames extracted")

        run_step(
            "Background remove",
            60,
            [
                sys.executable,
                str(REMBG_SCRIPT),
                "--input",
                str(IMAGE_DIR),
                "--output",
                str(REMBG_DIR),
                "--mask-output",
                str(MASK_DIR),
                "--device",
                "cuda",
            ],
        )
        total_elapsed = format_duration(time.perf_counter() - total_started_at)
        append_log(f"Total processing time: {total_elapsed}.")
        set_job(stage="Complete", percent=100, message=f"Processing complete in {total_elapsed}")
    except Exception as exc:
        append_log(str(exc))
        set_job(stage="Error", error=str(exc), message=str(exc))
    finally:
        set_job(running=False)


def image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def class_names() -> list[str]:
    root = IMAGE_DIR / "input_video"
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def source_split_dir(class_name: str, split: str, root: Path) -> Path:
    class_dir = root / "input_video" / class_name
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
    return max(0.0, value * (1.0 - BRIGHTNESS_PROFILE_VARIATION)), min(
        1.0, value * (1.0 + BRIGHTNESS_PROFILE_VARIATION)
    )


def adjust_object_brightness(image: np.ndarray, mask: np.ndarray, rng: random.Random, profile: dict | None) -> np.ndarray:
    if not profile:
        return image

    brightness_range = brightness_target_range(profile)
    if brightness_range is None:
        return image

    object_pixels = mask < 255
    if not np.any(object_pixels):
        return image

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    masked_hsv = hsv[object_pixels]

    current_v = max(float(masked_hsv[:, 2].mean()) / 255.0, 1e-6)
    target_v = rng.uniform(*brightness_range)
    hsv[..., 2] = np.clip(hsv[..., 2] * (target_v / current_v), 0.0, 255.0)

    adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return np.where(object_pixels[..., None], adjusted, image)


def load_object(image_path: Path, mask_path: Path, device: torch.device, rng: random.Random, profile: dict | None):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        return None
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    image = adjust_object_brightness(image, mask, rng, profile)
    if device.type == "cuda":
        image_tensor = torch.from_numpy(image).to(device=device, dtype=torch.float32).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).to(device=device)
        object_mask = mask_tensor < 255
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
    _, binary = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY_INV)
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
    max_attempts: int = 100,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    canvas_h, canvas_w = canvas_shape(canvas)
    obj_h, obj_w = object_shape(object_image)
    min_side = max(24, int(canvas_w * 0.12))
    max_side = max(min_side, int(canvas_w * 0.24))

    placed = None
    for _ in range(max_attempts):
        if size_profile:
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
        x = rng.randint(0, max(0, canvas_w - new_w))
        y = rng.randint(0, max(0, canvas_h - new_h))
        candidate = (x, y, new_w, new_h)
        if not any(boxes_overlap(candidate, box) for box in occupied_boxes):
            placed = candidate
            break

    if placed is None:
        return None

    x, y, new_w, new_h = placed
    if device.type == "cuda":
        paste_object_cuda(canvas, object_image, object_mask, x, y, new_w, new_h, device)
    else:
        object_image = cv2.resize(object_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        object_mask = cv2.resize(object_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        alpha = (object_mask.astype(np.float32) / 255.0)[..., None]
        roi = canvas[y : y + new_h, x : x + new_w].astype(np.float32)
        blended = object_image.astype(np.float32) * alpha + roi * (1.0 - alpha)
        canvas[y : y + new_h, x : x + new_w] = blended.astype(np.uint8)
    return canvas, placed


def paste_object_cuda(
    canvas: torch.Tensor,
    object_image: torch.Tensor,
    object_mask: torch.Tensor,
    x: int,
    y: int,
    width: int,
    height: int,
    device: torch.device,
) -> None:
    object_image = object_image.to(device=device, dtype=torch.float32)
    object_mask = object_mask.to(device=device, dtype=torch.float32)
    resized_image = torch_functional.interpolate(
        object_image.unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    resized_mask = torch_functional.interpolate(object_mask.unsqueeze(0), size=(height, width), mode="nearest")
    roi = canvas[:, y : y + height, x : x + width].unsqueeze(0)
    blended = resized_image * resized_mask + roi * (1.0 - resized_mask)
    canvas[:, y : y + height, x : x + width] = blended.squeeze(0)


def boxes_overlap(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    first_x, first_y, first_w, first_h = first
    second_x, second_y, second_w, second_h = second
    return not (
        first_x + first_w <= second_x
        or second_x + second_w <= first_x
        or first_y + first_h <= second_y
        or second_y + second_h <= first_y
    )


def write_yaml(names: list[str]) -> None:
    ensure_yolo_image_alias()
    lines = [
        f"path: {DATA_DIR}",
        "train: images/train",
        "val: images/validate",
        f"nc: {len(names)}",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(names))
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
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synthesize_split(
    split: str,
    count: int,
    names: list[str],
    backgrounds: list[Path],
    rng: random.Random,
    device: torch.device,
    object_profiles: dict,
    progress_start: int,
    progress_end: int,
) -> int:
    image_output = SYNTH_IMAGE_DIR / split
    label_output = SYNTH_LABEL_DIR / split
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    sources: dict[str, list[tuple[Path, Path]]] = {}
    for name in names:
        candidates = []
        for image_path in image_files(source_split_dir(name, split, IMAGE_DIR)):
            mask_path = match_mask(image_path)
            if mask_path is not None:
                candidates.append((image_path, mask_path))
        if not candidates:
            raise RuntimeError(f"No image/mask pairs found for class '{name}' in {split}.")
        sources[name] = candidates

    saved = 0
    attempts = 0
    max_image_attempts = max(count * 20, 20)
    while saved < count and attempts < max_image_attempts:
        attempts += 1
        background_path = rng.choice(backgrounds)
        background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
        if background is None:
            continue
        canvas = canvas_from_background(background, device)
        labels = []
        occupied_boxes = []
        for class_id, name in enumerate(names):
            object_pair = rng.choice(sources[name])
            object_profile = object_profiles.get(name)
            loaded = load_object(*object_pair, device, rng, object_profile)
            if loaded is None:
                continue
            pasted = paste_object(canvas, loaded[0], loaded[1], rng, occupied_boxes, device, object_profile)
            if pasted is None:
                break
            canvas, (x, y, w, h) = pasted
            occupied_boxes.append((x, y, w, h))
            canvas_h, canvas_w = canvas_shape(canvas)
            x_center = (x + w / 2) / canvas_w
            y_center = (y + h / 2) / canvas_h
            labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w / canvas_w:.6f} {h / canvas_h:.6f}")

        if len(labels) != len(names):
            continue
        stem = f"{split}_{saved:06d}"
        cv2.imwrite(str(image_output / f"{stem}.jpg"), canvas_to_image(canvas))
        (label_output / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
        saved += 1
        if count > 0:
            percent = progress_start + round((progress_end - progress_start) * saved / count)
            set_job(
                stage=f"Synthesizing {split}",
                percent=min(progress_end, percent),
                message=f"{split}: saved {saved}/{count} image(s)",
            )
    if saved < count:
        raise RuntimeError(f"Only saved {saved}/{count} {split} image(s) without overlap.")
    return saved


def synthesize_dataset(total_images: int) -> None:
    total_started_at = time.perf_counter()
    try:
        set_job(
            running=True,
            stage="Synthesizing",
            percent=5,
            message="Preparing synthesized dataset",
            error=None,
            log=[],
        )
        ensure_dirs()
        shutil.rmtree(SYNTH_IMAGE_DIR, ignore_errors=True)
        shutil.rmtree(SYNTH_LABEL_DIR, ignore_errors=True)
        SYNTH_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        SYNTH_LABEL_DIR.mkdir(parents=True, exist_ok=True)

        names = class_names()
        if not names:
            raise RuntimeError(f"No classes found in {IMAGE_DIR / 'input_video'}")
        backgrounds = image_files(BG_DIR)
        if not backgrounds:
            raise RuntimeError(f"No background images found in {BG_DIR}")

        train_count = max(1, round(total_images * 0.6))
        validate_count = max(1, total_images - train_count)
        if total_images == 1:
            train_count, validate_count = 1, 0

        rng = random.Random()
        device = datagen_device()
        object_profiles = load_object_profiles()
        append_log(f"Classes: {', '.join(names)}")
        append_log(f"Datagen compositor device: {device}")
        if device.type == "cuda":
            append_log(f"CUDA datagen GPU: {torch.cuda.get_device_name(device)}")
        if object_profiles:
            append_log(f"Using object size and brightness profile(s): {', '.join(sorted(object_profiles))}")
        append_log(f"Creating {train_count} train and {validate_count} validate image(s).")
        train_started_at = time.perf_counter()
        train_saved = synthesize_split("train", train_count, names, backgrounds, rng, device, object_profiles, 10, 60)
        train_elapsed = format_duration(time.perf_counter() - train_started_at)
        append_log(f"Train synthesis completed in {train_elapsed}.")
        set_job(stage="Synthesizing", percent=60, message=f"Saved {train_saved} train image(s) in {train_elapsed}")
        validate_started_at = time.perf_counter()
        validate_saved = synthesize_split(
            "validate",
            validate_count,
            names,
            backgrounds,
            rng,
            device,
            object_profiles,
            60,
            95,
        )
        validate_elapsed = format_duration(time.perf_counter() - validate_started_at)
        append_log(f"Validate synthesis completed in {validate_elapsed}.")
        write_yaml(names)
        total_elapsed = format_duration(time.perf_counter() - total_started_at)
        append_log(f"Total synthesis time: {total_elapsed}.")
        set_job(
            stage="Complete",
            percent=100,
            message=(
                f"Synthesis complete in {total_elapsed}: {train_saved} train, {validate_saved} validate. "
                f"YAML: {YOLO_DATASET_YAML}"
            ),
        )
    except Exception as exc:
        append_log(str(exc))
        set_job(stage="Error", error=str(exc), message=str(exc))
    finally:
        set_job(running=False)


@app.route("/")
def index():
    ensure_dirs()
    return render_template("index.html")


@app.post("/upload")
def upload():
    ensure_dirs()
    files = request.files.getlist("videos")
    if not files:
        return jsonify({"error": "No files were uploaded."}), 400

    saved = 0
    skipped = 0
    for file_storage in files:
        raw_name = file_storage.filename or ""
        suffix = Path(raw_name).suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            skipped += 1
            continue
        relative_path = safe_relative_path(raw_name)
        output_path = VIDEO_DIR / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_storage.save(output_path)
        saved += 1

    if saved == 0:
        return jsonify({"error": "No supported video files were uploaded."}), 400

    set_job(
        id=None,
        running=False,
        stage="Uploaded",
        percent=0,
        message=f"Saved {saved} video file(s).",
        error=None,
        log=[f"Saved {saved} video file(s), skipped {skipped} file(s)."],
    )
    return jsonify({"saved": saved, "skipped": skipped})


@app.post("/start")
def start():
    current = snapshot_job()
    if current["running"]:
        return jsonify({"error": "A job is already running."}), 409

    try:
        images_per_video = int(
            request.form.get("images_per_video", request.json.get("images_per_video") if request.is_json else 108)
        )
    except Exception:
        images_per_video = 108
    images_per_video = max(1, images_per_video)

    videos = [p for p in VIDEO_DIR.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    if not videos:
        return jsonify({"error": "Upload videos before starting."}), 400

    job_id = uuid.uuid4().hex
    set_job(id=job_id, running=True, stage="Queued", percent=1, message="Job queued", error=None, log=[])
    thread = threading.Thread(target=process_dataset, args=(images_per_video,), daemon=True)
    thread.start()
    return jsonify({"job_id": job_id})


@app.post("/synthesize")
def synthesize():
    current = snapshot_job()
    if current["running"]:
        return jsonify({"error": "A job is already running."}), 409

    try:
        total_images = int(request.form.get("total_images", request.json.get("total_images") if request.is_json else 100))
    except Exception:
        total_images = 100
    total_images = max(1, total_images)

    if not IMAGE_DIR.exists() or not MASK_DIR.exists():
        return jsonify({"error": "Create frames and masks before synthesizing."}), 400

    job_id = uuid.uuid4().hex
    set_job(id=job_id, running=True, stage="Queued", percent=1, message="Synthesis queued", error=None, log=[])
    thread = threading.Thread(target=synthesize_dataset, args=(total_images,), daemon=True)
    thread.start()
    return jsonify({"job_id": job_id})


@app.get("/classes")
def classes():
    names = class_names()
    if not names:
        names = sorted(load_object_profiles().keys())
    return jsonify({"classes": names})


@app.get("/object-profiles")
def object_profiles():
    return jsonify(load_object_profiles())


@app.post("/object-profile")
def object_profile():
    data = request.get_json(silent=True) or {}
    class_name = secure_filename(str(data.get("class_name", ""))).strip()
    try:
        width = int(round(float(data.get("width", 0))))
        height = int(round(float(data.get("height", 0))))
    except Exception:
        width = 0
        height = 0
    try:
        source_width = int(round(float(data.get("source_width", 0))))
        source_height = int(round(float(data.get("source_height", 0))))
        target_width = int(round(float(data.get("target_width", 640))))
        target_height = int(round(float(data.get("target_height", 640))))
        bbox_x = int(round(float(data.get("bbox_x", 0))))
        bbox_y = int(round(float(data.get("bbox_y", 0))))
    except Exception:
        source_width = 0
        source_height = 0
        target_width = 640
        target_height = 640
        bbox_x = 0
        bbox_y = 0
    if not class_name:
        return jsonify({"error": "Select a class before saving the object profile."}), 400
    if width <= 1 or height <= 1:
        return jsonify({"error": "Draw a valid bounding box before saving."}), 400

    try:
        brightness = float(data.get("brightness", 0))
    except (TypeError, ValueError):
        brightness = 0.0
    close_sample = distance_sample(data, "close_sample")
    far_sample = distance_sample(data, "far_sample")
    try:
        wanted_distance_cm = float(data.get("wanted_distance_cm", 0))
    except (TypeError, ValueError):
        wanted_distance_cm = 0.0
    try:
        min_distance_cm = float(data.get("min_distance_cm", 0))
        max_distance_cm = float(data.get("max_distance_cm", 0))
        min_width = int(round(float(data.get("min_width", width))))
        max_width = int(round(float(data.get("max_width", width))))
        min_height = int(round(float(data.get("min_height", height))))
        max_height = int(round(float(data.get("max_height", height))))
    except (TypeError, ValueError):
        min_distance_cm = 0.0
        max_distance_cm = 0.0
        min_width = width
        max_width = width
        min_height = height
        max_height = height
    profile = {
        "width": width,
        "height": height,
        "min_width": max(2, min_width),
        "max_width": max(2, max_width),
        "min_height": max(2, min_height),
        "max_height": max(2, max_height),
        "bbox_x": bbox_x,
        "bbox_y": bbox_y,
        "source_width": source_width,
        "source_height": source_height,
        "target_width": target_width,
        "target_height": target_height,
        "close_sample": close_sample,
        "far_sample": far_sample,
        "wanted_distance_cm": round(wanted_distance_cm, 3) if wanted_distance_cm > 0 else None,
        "min_distance_cm": round(min_distance_cm, 3) if min_distance_cm > 0 else None,
        "max_distance_cm": round(max_distance_cm, 3) if max_distance_cm > 0 else None,
        "brightness": round(max(0.0, min(1.0, brightness)), 6),
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
            if not snapshot_job()["running"]:
                time.sleep(0.5)
            else:
                time.sleep(0.25)

    response = Response(stream(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@app.post("/clear")
def clear():
    current = snapshot_job()
    if current["running"]:
        return jsonify({"error": "Cannot clear while a job is running."}), 409
    if YOLO_IMAGE_ALIAS_DIR.is_symlink():
        YOLO_IMAGE_ALIAS_DIR.unlink()
    for path in (VIDEO_DIR, IMAGE_DIR, MASK_DIR, REMBG_DIR, SYNTH_IMAGE_DIR, SYNTH_LABEL_DIR, YOLO_IMAGE_ALIAS_DIR):
        shutil.rmtree(path, ignore_errors=True)
    if YOLO_DATASET_YAML.exists():
        YOLO_DATASET_YAML.unlink()
    ensure_dirs()
    set_job(
        id=None,
        running=False,
        stage="Idle",
        percent=0,
        message="Cleared dataset folders.",
        error=None,
        log=["Cleared dataset folders."],
    )
    return jsonify({"ok": True})


if __name__ == "__main__":
    ensure_dirs()
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
