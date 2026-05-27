#!/usr/bin/env python3
"""Pre-flight checker for the datagen pipeline.

Run from the project root:
    python3 scripts/check_setup.py
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

_passes = _warns = _errors = 0

def _ok(label: str, detail: str = "") -> None:
    global _passes
    _passes += 1
    suffix = f"  {detail}" if detail else ""
    print(f"  {GREEN}PASS{RESET}  {label}{suffix}")

def _warn(label: str, detail: str = "") -> None:
    global _warns
    _warns += 1
    suffix = f"  {detail}" if detail else ""
    print(f"  {YELLOW}WARN{RESET}  {label}{suffix}")

def _fail(label: str, detail: str = "") -> None:
    global _errors
    _errors += 1
    suffix = f"  {detail}" if detail else ""
    print(f"  {RED}FAIL{RESET}  {label}{suffix}")


def check_python() -> None:
    print(f"\n{BOLD}Python{RESET}")
    v = sys.version_info
    if v >= (3, 9):
        _ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        _fail(f"Python {v.major}.{v.minor}.{v.micro}", "Python 3.9+ required")


def check_packages() -> None:
    print(f"\n{BOLD}Python packages{RESET}")
    packages = [
        ("flask",       "flask",          "flask"),
        ("yaml",        "pyyaml",         "PyYAML"),
        ("cv2",         "opencv-python",  "opencv-python"),
        ("numpy",       "numpy",          "numpy"),
        ("torch",       "torch",          "torch"),
        ("ultralytics", "ultralytics",    "ultralytics"),
        ("rembg",       "rembg",          "rembg"),
        ("werkzeug",    "Werkzeug",       "werkzeug"),
    ]
    for mod, pip_name, dist_name in packages:
        spec = importlib.util.find_spec(mod)
        if spec is None:
            _fail(f"{pip_name}", f"not installed — run: pip install {pip_name}")
        else:
            try:
                ver = importlib.metadata.version(dist_name)
            except importlib.metadata.PackageNotFoundError:
                m = importlib.import_module(mod)
                ver = getattr(m, "__version__", "?")
            _ok(f"{pip_name} {ver}")

    # CUDA
    try:
        import torch
        if torch.cuda.is_available():
            _ok("CUDA", f"device 0 = {torch.cuda.get_device_name(0)}")
        else:
            _warn("CUDA", "not available — training will use CPU (slower)")
    except Exception:
        pass


def check_structure() -> None:
    print(f"\n{BOLD}File / directory structure{RESET}")
    required_dirs = [
        "assets/backgrounds",
        "bin",
        "workspace/input",
        "workspace/frames",
        "workspace/masks",
        "workspace/cutouts",
        "workspace/data",
        "workspace/data/phase1",
        "workspace/data/phase4",
        "app",
        "app/templates",
        "scripts",
        "tools",
    ]
    required_files = [
        "config.yaml",
        "requirements.txt",
        "app/server.py",
        "app/templates/index.html",
        "scripts/frame_stride.py",
        "tools/rembg.py",
    ]
    for d in required_dirs:
        p = PROJECT_DIR / d
        if p.exists():
            _ok(d + "/")
        else:
            _fail(d + "/", "missing — run setup.sh")
    for f in required_files:
        p = PROJECT_DIR / f
        if p.exists():
            _ok(f)
        else:
            _fail(f, "missing")


def check_assets() -> None:
    print(f"\n{BOLD}Assets{RESET}")
    bg_dir = PROJECT_DIR / "assets" / "backgrounds"
    IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not bg_dir.exists():
        _fail("assets/backgrounds/", "directory missing")
        return

    all_images = [p for p in bg_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT]
    if not all_images:
        _fail("backgrounds", "no images found — add .jpg/.png files to assets/backgrounds/")
    else:
        _ok("backgrounds", f"{len(all_images)} image(s) total")

    categories = sorted(d.name for d in bg_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    for cat in categories:
        cat_images = [p for p in (bg_dir / cat).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT]
        if len(cat_images) < 15:
            _warn(f"backgrounds/{cat}/", f"{len(cat_images)} image(s) — aim for 15+ to avoid repetition")
        else:
            _ok(f"backgrounds/{cat}/", f"{len(cat_images)} image(s)")

    # ffmpeg
    import shutil
    ffmpeg_ok = bool(shutil.which("ffmpeg")) or (PROJECT_DIR / "bin" / "ffmpeg").exists()
    if ffmpeg_ok:
        _ok("ffmpeg", "found")
    else:
        _warn("ffmpeg", "not found — CUDA frame extraction unavailable (CPU fallback will be used)")


def check_model() -> None:
    print(f"\n{BOLD}Model weights{RESET}")
    import yaml
    config_path = PROJECT_DIR / "config.yaml"
    if not config_path.exists():
        _fail("config.yaml", "missing")
        return

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    model_rel = (cfg.get("phase3") or {}).get("model", "")
    if not model_rel:
        _fail("phase3.model", "not set in config.yaml")
        return

    model_path = PROJECT_DIR / model_rel if not Path(model_rel).is_absolute() else Path(model_rel)
    if model_path.exists():
        size_mb = model_path.stat().st_size / 1_048_576
        _ok(model_rel, f"{size_mb:.1f} MB")
    else:
        _fail(model_rel, "not found — download from ultralytics or run: yolo export model=yolov8s.pt")


def check_config() -> None:
    print(f"\n{BOLD}Configuration (config.yaml){RESET}")
    import yaml
    config_path = PROJECT_DIR / "config.yaml"
    if not config_path.exists():
        _fail("config.yaml", "missing")
        return

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    required_sections = ["phase0", "phase1", "phase2", "phase3", "phase4"]
    for sec in required_sections:
        if sec in cfg:
            _ok(f"section [{sec}]")
        else:
            _fail(f"section [{sec}]", "missing — run setup.sh to add defaults")

    p2 = cfg.get("phase2") or {}
    total = (p2.get("train_pct") or 0) + (p2.get("val_pct") or 0) + (p2.get("test_pct") or 0)
    if total == 100:
        _ok("dataset split", f"train={p2.get('train_pct')}% + val={p2.get('val_pct')}% + test={p2.get('test_pct')}% = 100%")
    else:
        _fail("dataset split", f"train+val+test = {total}% (must equal 100%)")


def check_phase1() -> None:
    print(f"\n{BOLD}Phase 1 classes{RESET}")
    phase1_dir = PROJECT_DIR / "workspace" / "data" / "phase1"
    rembg_dir  = PROJECT_DIR / "workspace" / "cutouts"
    IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if not phase1_dir.exists():
        _warn("workspace/data/phase1/", "does not exist yet — upload and process videos first")
        return

    classes = sorted(d.name for d in phase1_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not classes:
        _warn("classes", "no classes found — process videos in Phase 0 first")
        return

    for cls in classes:
        cls_dir = phase1_dir / cls
        frames = [p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT] if cls_dir.exists() else []
        rembg_cls = rembg_dir / cls
        cutouts = [p for p in rembg_cls.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT] if rembg_cls.exists() else []

        if not frames:
            _warn(f"class '{cls}'", "no frames (run Phase 0)")
        elif not cutouts:
            _warn(f"class '{cls}'", f"{len(frames)} frame(s), 0 cutouts — run Phase 1 (Background Removal)")
        else:
            _ok(f"class '{cls}'", f"{len(frames)} frame(s), {len(cutouts)} cutout(s)")


def main() -> int:
    print(f"\n{BOLD}{'═' * 46}{RESET}")
    print(f"{BOLD}   datagen Pre-flight Check{RESET}")
    print(f"{BOLD}{'═' * 46}{RESET}")
    print(f"   Project: {PROJECT_DIR}")

    check_python()
    check_packages()
    check_structure()
    check_assets()
    check_model()
    check_config()
    check_phase1()

    print(f"\n{BOLD}{'═' * 46}{RESET}")
    if _errors:
        print(f"{BOLD}{RED}  Result: {_errors} error(s), {_warns} warning(s) — fix errors before running{RESET}")
    elif _warns:
        print(f"{BOLD}{YELLOW}  Result: {_warns} warning(s) — setup is usable but not ideal{RESET}")
    else:
        print(f"{BOLD}{GREEN}  Result: All {_passes} checks passed — ready to run!{RESET}")
    print(f"{BOLD}{'═' * 46}{RESET}\n")

    return 1 if _errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
