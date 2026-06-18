from __future__ import annotations

import argparse
import os
import site
import sys
import typing
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_COVERAGE", "1")
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [path for path in sys.path if Path(path or ".").resolve() != SCRIPT_DIR]


def configure_nvidia_library_path() -> None:
    if os.environ.get("REMBG_NVIDIA_LIB_PATH_READY") == "1":
        return

    site_roots = [Path(site.getusersitepackages())]
    site_roots.extend(Path(path) for path in site.getsitepackages())
    nvidia_lib_dirs = []
    for root in site_roots:
        nvidia_root = root / "nvidia"
        if not nvidia_root.exists():
            continue
        nvidia_lib_dirs.extend(str(path) for path in nvidia_root.glob("*/lib") if path.is_dir())

    if not nvidia_lib_dirs:
        os.environ["REMBG_NVIDIA_LIB_PATH_READY"] = "1"
        return

    existing_paths = [path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path]
    merged_paths = nvidia_lib_dirs + [path for path in existing_paths if path not in nvidia_lib_dirs]
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged_paths)
    os.environ["REMBG_NVIDIA_LIB_PATH_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)


configure_nvidia_library_path()

try:
    import coverage.types

    if not hasattr(coverage.types, "Tracer") and hasattr(coverage.types, "TracerCore"):
        coverage.types.Tracer = coverage.types.TracerCore
    coverage.types.TShouldTraceFn = getattr(coverage.types, "TShouldTraceFn", typing.Any)
    coverage.types.TShouldStartContextFn = getattr(coverage.types, "TShouldStartContextFn", typing.Any)
except Exception:
    pass

from PIL import Image, ImageFilter
from rembg import new_session, remove
import onnxruntime as ort


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove image backgrounds and create alpha-derived masks.")
    parser.add_argument("--input", required=True, type=Path, help="Directory containing extracted images.")
    parser.add_argument("--output", required=True, type=Path, help="Directory for rembg RGBA PNG output.")
    parser.add_argument("--mask-output", required=True, type=Path, help="Directory for white-background black-object masks.")
    parser.add_argument("--model", default="u2net", help="rembg model name.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "rocm"],
        help="Inference device. Use cuda for NVIDIA GPU when onnxruntime-gpu is installed.",
    )
    parser.add_argument(
        "--alpha-threshold",
        default=80,
        type=int,
        help="Drop semi-transparent edge pixels below this alpha value.",
    )
    parser.add_argument(
        "--erode-pixels",
        default=2,
        type=int,
        help="Shrink the object alpha by this many pixels to remove background borders.",
    )
    return parser.parse_args()


def providers_for_device(device: str) -> list[str] | None:
    available = ort.get_available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print(
            "CUDAExecutionProvider is not available. Falling back to CPUExecutionProvider. "
            "Install onnxruntime-gpu with matching CUDA/cuDNN to use the GPU.",
            flush=True,
        )
        return ["CPUExecutionProvider"]
    if device == "rocm":
        if "ROCMExecutionProvider" in available:
            return ["ROCMExecutionProvider", "CPUExecutionProvider"]
        print("ROCMExecutionProvider is not available. Falling back to CPUExecutionProvider.", flush=True)
        return ["CPUExecutionProvider"]
    return None


def image_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_mask_from_alpha(rgba: Image.Image, output_path: Path) -> None:
    alpha = rgba.getchannel("A")
    mask = Image.new("L", alpha.size, 255)
    object_pixels = alpha.point(lambda value: 0 if value > 0 else 255)
    mask.paste(object_pixels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output_path)


def sharpen_alpha(rgba: Image.Image, alpha_threshold: int, erode_pixels: int) -> Image.Image:
    alpha_threshold = max(0, min(255, alpha_threshold))
    alpha = rgba.getchannel("A").point(lambda value: 255 if value >= alpha_threshold else 0)
    for _ in range(max(0, erode_pixels)):
        alpha = alpha.filter(ImageFilter.MinFilter(3))
    result = rgba.copy()
    result.putalpha(alpha)
    return result


def process_image(
    image_path: Path,
    input_dir: Path,
    output_dir: Path,
    mask_dir: Path,
    session,
    alpha_threshold: int,
    erode_pixels: int,
) -> None:
    relative = image_path.relative_to(input_dir)
    rembg_path = (output_dir / relative).with_suffix(".png")
    mask_path = (mask_dir / relative).with_suffix(".png")

    with Image.open(image_path) as image:
        rgba = remove(image.convert("RGBA"), session=session)
        if not isinstance(rgba, Image.Image):
            rgba = Image.open(rgba).convert("RGBA")
        else:
            rgba = rgba.convert("RGBA")
        rgba = sharpen_alpha(rgba, alpha_threshold, erode_pixels)

    rembg_path.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(rembg_path)
    write_mask_from_alpha(rgba, mask_path)
    print(f"{relative}: wrote rembg and mask", flush=True)


def main() -> int:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    mask_dir = args.mask_output.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    images = image_files(input_dir)
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    providers = providers_for_device(args.device)
    if providers is None:
        session = new_session(args.model)
        provider_message = "auto"
    else:
        session = new_session(args.model, providers=providers)
        provider_message = ", ".join(providers)
    active_providers = session.inner_session.get_providers()

    print(
        f"Found {len(images)} image(s). Removing backgrounds with {args.model}. "
        f"Requested providers: {provider_message}. Active providers: {', '.join(active_providers)}.",
        flush=True,
    )
    for index, image_path in enumerate(images, start=1):
        process_image(
            image_path,
            input_dir,
            output_dir,
            mask_dir,
            session,
            args.alpha_threshold,
            args.erode_pixels,
        )
        print(f"Processed {index}/{len(images)}", flush=True)

    print(f"Background removal complete. Masks saved to {mask_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())