# datagen

A browser-based pipeline for generating synthetic YOLO object-detection datasets from real object videos. Record short clips of your objects, drop them in, and the tool handles frame extraction, background removal, dataset synthesis, model training, and inference — all from a single web UI.

---

## Table of Contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project structure](#project-structure)
- [Tutorial](#tutorial)
  - [Step 1 — Prepare your inputs](#step-1--prepare-your-inputs)
  - [Step 2 — Start the server](#step-2--start-the-server)
  - [Step 3 — Phase 0: Video upload & frame extraction](#step-3--phase-0-video-upload--frame-extraction)
  - [Step 4 — Phase 1: Background removal](#step-4--phase-1-background-removal)
  - [Step 5 — Phase 2: Dataset synthesis](#step-5--phase-2-dataset-synthesis)
  - [Step 6 — Phase 3: Training](#step-6--phase-3-training)
  - [Step 7 — Phase 4: Evaluation](#step-7--phase-4-evaluation)
- [Configuration reference](#configuration-reference)
- [Background categories](#background-categories)
- [Advanced: hyperparameter tuning](#advanced-hyperparameter-tuning)
- [Pre-flight check](#pre-flight-check)

---

## How it works

```
Your videos  →  Frame extraction  →  Background removal  →  Synthetic compositing
                (Phase 0)             (Phase 1)               (Phase 2)

YOLO training  →  Model evaluation
(Phase 3)          (Phase 4)
```

Each object class is a folder of short videos. The pipeline extracts clean frames, removes the background with `rembg`, then composites the cut-out objects onto background images in random positions, scales, and lighting conditions. The resulting dataset is ready to train a YOLOv8/v11 model without any manual labelling.

---

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.9+ |
| CUDA (recommended) | 11.8 or 12.x |
| GPU VRAM | 4 GB+ for training |
| Disk space | 10 GB+ for a typical project |

**Python packages** — installed automatically by `setup.sh`:

```
flask, pyyaml, opencv-python, numpy, torch, torchvision, ultralytics, rembg, onnxruntime, werkzeug
```

---

## Installation

**1. Clone or copy the project**

```bash
cd ~/your_workspace/src
git clone <repo-url> datagen
cd datagen
```

**2. Run the setup script**

```bash
bash setup.sh
```

This will:
- Install all Python dependencies via `pip`
- Upgrade to `onnxruntime-gpu` if CUDA is detected
- Create all required workspace directories
- Write default values to `config.yaml` for any missing keys

**3. Verify the setup (optional but recommended)**

```bash
python3 scripts/check_setup.py
```

You shouldsee all green `PASS` lines. Warnings about missing backgrounds or classes are expected at this stage.

**4. Download YOLO weights** (if not already present)

The default model is `models/yolov8s.pt`. Download it from Ultralytics if it is not in the `models/` folder:

```bash
python3 -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
mv yolov8s.pt models/
```

Or download `yolo11n.pt` for a lighter model:

```bash
python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
mv yolo11n.pt models/
# Then update config.yaml → phase3.model: models/yolo11n.pt
```

---

## Project structure

```
datagen/
├── config.yaml               ← all pipeline settings (edit via UI or directly)
├── requirements.txt
├── setup.sh                  ← one-time install script
│
├── assets/
│   └── backgrounds/          ← put background images here (jpg/png)
│       ├── indoor/           ← optional: named category subfolders
│       └── outdoor/
│
├── app/                      ← web server (do not edit unless developing)
│   ├── server.py
│   └── templates/index.html
│
├── bin/
│   └── ffmpeg                ← optional local ffmpeg binary for CUDA extraction
│
├── models/
│   ├── yolov8s.pt            ← base model for training
│   └── yolo11n.pt
│
├── runs/                     ← YOLO training output (auto-generated)
│
├── scripts/
│   ├── check_setup.py        ← pre-flight health checker
│   ├── frame_stride.py       ← frame extraction utility
│   ├── train_with_tuning.py  ← advanced: hyperparameter tuning + training
│   └── run_training.sh       ← advanced: launch tuned training from CLI
│
├── tools/
│   ├── rembg.py              ← background removal (rembg)
│   └── sam2.py               ← background removal (SAM 2, optional)
│
└── workspace/                ← all pipeline working data (auto-managed)
    ├── input/                ← Phase 0: uploaded videos land here
    ├── frames/               ← Phase 0: extracted frames
    ├── masks/                ← Phase 1: segmentation masks
    ├── cutouts/              ← Phase 1: background-removed PNG cutouts
    └── data/
        ├── phase1/           ← accepted class images
        ├── images/           ← YOLO dataset images (train/val/test)
        ├── labels/           ← YOLO dataset labels
        ├── sythesized_data/  ← raw synthesis output
        ├── sythesized_data.yaml
        ├── object_profiles.json
        └── phase4/           ← Phase 4 evaluation results
```

---

## Tutorial

### Step 1 — Prepare your inputs

**Record your object videos**

Film 2–3 short videos (15–30 seconds each) of each object you want to detect. Tips:
- Rotate and tilt the object to capture different angles
- Use consistent, even lighting
- Film against a **solid-colour background** (white, green, or black) for the best background-removal quality
- Keep the object in frame throughout

**Add background images**

Copy background photos into `assets/backgrounds/`. These are the scenes your objects will be composited onto. Aim for **15+ images per scene type**.

To restrict a class to specific backgrounds (e.g. put a kitchen item only on kitchen backgrounds), create named subfolders:

```
assets/backgrounds/
├── indoor/        ← 15+ indoor photos
├── outdoor/       ← 15+ outdoor photos
└── shelves/       ← 15+ shelf photos
```

---

### Step 2 — Start the server

```bash
python3 app/server.py
```

Open your browser at **http://127.0.0.1:5000**.

> **Note:** The server must keep running while you work through the phases. Keep the terminal open.

---

### Step 3 — Phase 0: Video upload & frame extraction

1. Click the **Phase 0** tab.
2. Under **Video Upload & Frame Extraction**, click the **Video folder** input and select the folder containing your videos. Organise your videos into subfolders named after each class:
   ```
   my_videos/
   ├── catalog/
   │   ├── clip1.mp4
   │   └── clip2.mp4
   └── detergent/
       ├── clip1.mp4
       └── clip2.mp4
   ```
3. Click **Upload Folder** and wait for the upload to finish.
4. Click **Process Videos** to extract frames. Progress is shown in real time.
5. After processing, the **Class Status** panel shows each detected class. Review and accept each one.

> **Tip:** If the extracted frame count differs greatly from the target, you will see a warning banner. Adjust `images_per_video` in Settings if needed.

**Distance calibration (optional)**

If you want distance-based object sizing for synthesis, use the **Distance Calibration & Object Profile** tool. Enter the wanted distance range and optional object max range in centimeters. When an object max range is saved, synthesis samples a camera-to-object distance up to that range and scales the original object pixels by `object_max_range_cm / sampled_distance_cm`. This is optional — synthesis will work without it.

**Background categories (optional)**

In the **Background Categories** panel, select a class and tick which background subfolders it is allowed to use. Leave all unticked to allow any background.

---

### Step 4 — Phase 1: Background removal

1. Click the **Phase 1** tab.
2. Click **Remove Backgrounds**. The pipeline runs `rembg` on every accepted frame and produces PNG cutouts with transparent backgrounds.
3. Review the results in the gallery. Re-run if the quality is poor (better lighting helps).
4. Click **Accept All** (or accept classes individually) to confirm.

> **Tip:** For complex objects, re-recording against a **plain white or green background** significantly improves cutout quality.

---

### Step 5 — Phase 2: Dataset synthesis

1. Click the **Phase 2** tab.
2. Adjust **Synthesis Settings** if needed (number of objects per image, augmentation strength, output resolution).
3. Click **Preview** to see a sample composite image before generating the full dataset.
4. When happy with the preview, click **Generate Dataset**.

The pipeline composites cut-out objects onto your background images with random position, scale, rotation, blur, and brightness variation, and writes YOLO-format labels automatically.

**Output split (default):**
| Split | % |
|---|---|
| Train | 80 % |
| Validation | 15 % |
| Test | 5 % |

---

### Step 6 — Phase 3: Training

1. Click the **Phase 3** tab.
2. Review **Training Configuration** — the key settings are:

   | Setting | Default | Notes |
   |---|---|---|
   | Model | `models/yolov8s.pt` | Base weights |
   | Epochs | 3 | Increase to 50–200 for real training |
   | Batch | 4 | Increase if GPU has more VRAM |
   | Image size | 640 | Match your inference resolution |
   | Device | `0` | GPU index; use `cpu` for CPU-only |
   | Patience | 20 | Early-stop patience |

3. Click **Start Training**. Live epoch progress and loss curves are shown as training runs.
4. When training finishes, the best weights path is displayed. The weights are saved under `runs/detect/train/weights/best.pt`.

---

### Step 7 — Phase 4: Evaluation

1. Click the **Phase 4** tab.
2. Upload a test video (mp4/avi/mov).
3. Set the **Confidence threshold** (default 0.75) and **Sample FPS**.
4. Click **Run Evaluation**. The model runs inference on the video and logs detections.
5. Download the results CSV when complete.

---

## Configuration reference

All settings are editable via the **Settings** tab in the UI, or directly in `config.yaml`.

### Phase 0 — Frame extraction

| Key | Default | Description |
|---|---|---|
| `images_per_video` | 108 | Target frames to extract per video |
| `target_frame_count` | 108 | Expected total frames per class |
| `frame_count_tolerance_pct` | 20 | % tolerance before showing a warning |
| `ssim_dedup_threshold` | 0.97 | SSIM similarity above which near-duplicate frames are dropped |

### Phase 1 — Background removal

| Key | Default | Description |
|---|---|---|
| `expected_images_per_class` | 108 | Minimum accepted images per class |
| `min_backgrounds_per_category` | 15 | Warning threshold for background category size |

### Phase 2 — Synthesis

| Key | Default | Description |
|---|---|---|
| `train_pct` / `val_pct` / `test_pct` | 80 / 15 / 5 | Dataset split percentages (must sum to 100) |
| `max_objects_per_image` | 15 | Maximum objects composited per image |
| `output_resolution` | 640 × 640 | Output image size in pixels |
| `placement_mode` | `random` | Object placement strategy |
| `overlap_threshold_pct` | 20 | Max bounding-box overlap % before skipping a placement |
| `out_of_frame_pct` | 10 | % of objects allowed to be partially outside the frame |
| `blur_max_pct` | 15 | Maximum blur strength (% of image width) |
| `brightness_variation_pct` | 10 | Brightness jitter range (±%) |
| `contrast_variation_pct` | 20 | Contrast jitter range (±%) |

### Phase 3 — Training

| Key | Default | Description |
|---|---|---|
| `model` | `models/yolov8s.pt` | Path to base YOLO weights |
| `epochs` | 3 | Training epochs |
| `batch` | 4 | Batch size |
| `imgsz` | 640 | Training image size |
| `device` | `0` | GPU index or `cpu` |
| `patience` | 20 | Early-stop patience (epochs without improvement) |
| `project` | `runs/detect` | Output directory for training runs |
| `name` | `train` | Name for this training run |

### Phase 4 — Evaluation

| Key | Default | Description |
|---|---|---|
| `confidence_threshold` | 0.75 | Minimum detection confidence |
| `sample_fps` | 1 | Frames per second to sample from the test video |
| `accepted_formats` | mp4, avi, mov | Allowed video file extensions |

---

## Background categories

Background categories let you restrict each class to specific background subfolders, so (for example) a kitchen object is never placed on an outdoor background.

**Setup:**

1. Create named subfolders inside `assets/backgrounds/`:
   ```
   assets/backgrounds/
   ├── kitchen/    ← 15+ kitchen photos
   ├── shelf/      ← 15+ shelf photos
   └── outdoor/    ← 15+ outdoor photos
   ```

2. In the UI → **Phase 0** → **Background Categories**, select a class, tick the allowed categories, and click **Save Categories**.

**Rules:**
- A class with **no categories ticked** may use any background image.
- When multiple classes appear in the same image, the synthesiser picks from the **intersection** of their allowed categories. If the intersection is empty (conflicting restrictions), it falls back to the union.

---

## Advanced: hyperparameter tuning

For best accuracy, use Ultralytics' built-in hyperparameter tuner before the final training run:

```bash
bash scripts/run_training.sh
```

Or directly:

```bash
python3 scripts/train_with_tuning.py \
  --epochs 150 \
  --tune-epochs 10 \
  --tune-iterations 30 \
  --batch 16 \
  --device 0
```

Results and the best weights are saved to `evaluation/`.

---

## Pre-flight check

Run this before starting a new project to catch missing dependencies, misconfigured paths, or low background counts:

```bash
python3 scripts/check_setup.py
```

Example output:

```
══════════════════════════════════════════════
   datagen Pre-flight Check
══════════════════════════════════════════════
   Project: /your/path/datagen

Python
  PASS  Python 3.12.3

Python packages
  PASS  flask 3.0.2
  PASS  torch 2.5.0+cu124
  PASS  CUDA  device 0 = NVIDIA GeForce RTX 3080
  ...

Assets
  PASS  backgrounds  42 image(s) total
  WARN  backgrounds/indoor/  8 image(s) — aim for 15+ to avoid repetition

Model weights
  PASS  models/yolov8s.pt  21.5 MB

══════════════════════════════════════════════
  Result: 1 warning(s) — setup is usable but not ideal
══════════════════════════════════════════════
```

A **System Status** panel on the Phase 0 tab in the UI shows the same checks at a glance.
