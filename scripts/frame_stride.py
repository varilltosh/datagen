from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".mp4", ".mov", ".move", ".avi", ".mkv", ".webm", ".m4v"}
PROJECT_DIR = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument("--input", required=True, type=Path, help="Directory containing input videos.")
    parser.add_argument("--output", required=True, type=Path, help="Directory that will receive extracted frames.")
    parser.add_argument("--stride", default=10, type=int, help="Save one frame every N frames.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Frame extraction device. cuda uses ffmpeg NVDEC when ffmpeg is installed.",
    )
    parser.add_argument(
        "--images-per-video",
        type=int,
        help="Save this many evenly spaced frames from each video. Overrides --stride.",
    )
    parser.add_argument("--image-ext", default=".png", choices=[".png", ".jpg", ".jpeg"], help="Output image extension.")
    return parser.parse_args()


def video_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def frame_indices(frame_count: int, stride: int, images_per_video: int | None) -> set[int]:
    if frame_count <= 0:
        return set()
    if images_per_video is None:
        return set(range(0, frame_count, max(1, stride)))
    target = max(1, min(images_per_video, frame_count))
    if target == 1:
        return {0}
    return {round(i * (frame_count - 1) / (target - 1)) for i in range(target)}


def ffmpeg_path() -> str | None:
    candidates = [
        shutil.which("ffmpeg"),
        str(PROJECT_DIR / "bin" / "ffmpeg"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def ffmpeg_has_cuda(ffmpeg: str) -> bool:
    process = subprocess.run(
        [ffmpeg, "-hide_banner", "-hwaccels"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return "cuda" in process.stdout.lower()


def video_frame_count(video_path: Path) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return frame_count


def extract_video_cuda(
    video_path: Path,
    input_dir: Path,
    output_dir: Path,
    stride: int,
    images_per_video: int | None,
    image_ext: str,
    ffmpeg: str,
) -> int:
    frame_count = video_frame_count(video_path)
    if frame_count <= 0:
        print(f"Skipping unreadable video: {video_path}", flush=True)
        return 0

    selected_indices = sorted(frame_indices(frame_count, stride, images_per_video))
    if not selected_indices:
        return 0

    relative_parent = video_path.relative_to(input_dir).parent
    frame_dir = output_dir / relative_parent
    frame_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    select_expr = "+".join(f"eq(n\\,{index})" for index in selected_indices)

    with tempfile.TemporaryDirectory(prefix=f"{stem}_frames_") as temp_dir:
        temp_pattern = str(Path(temp_dir) / f"{stem}_tmp_%06d{image_ext}")
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-hwaccel",
            "cuda",
            "-i",
            str(video_path),
            "-vf",
            f"select='{select_expr}'",
            "-vsync",
            "0",
            "-start_number",
            "0",
            temp_pattern,
        ]
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg CUDA extraction failed for {video_path.name}: {process.stdout.strip()}")

        temp_files = sorted(Path(temp_dir).glob(f"{stem}_tmp_*{image_ext}"))
        for selected_index, temp_file in zip(selected_indices, temp_files):
            output_path = frame_dir / f"{stem}_frame_{selected_index:06d}{image_ext}"
            shutil.move(str(temp_file), output_path)

    saved_count = min(len(selected_indices), len(temp_files))
    print(f"{video_path.name}: saved {saved_count} frame(s) with CUDA ffmpeg", flush=True)
    return saved_count


def extract_video(
    video_path: Path,
    input_dir: Path,
    output_dir: Path,
    stride: int,
    images_per_video: int | None,
    image_ext: str,
) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Skipping unreadable video: {video_path}", flush=True)
        return 0

    selected_indices = frame_indices(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), stride, images_per_video)
    relative_parent = video_path.relative_to(input_dir).parent
    stem = video_path.stem
    frame_index = 0
    saved_count = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in selected_indices:
            frame_dir = output_dir / relative_parent
            frame_dir.mkdir(parents=True, exist_ok=True)
            image_name = f"{stem}_frame_{frame_index:06d}{image_ext}"
            output_path = frame_dir / image_name
            cv2.imwrite(str(output_path), frame)
            saved_count += 1
        frame_index += 1

    capture.release()
    print(f"{video_path.name}: saved {saved_count} frame(s)", flush=True)
    return saved_count


def extract_video_with_device(
    video_path: Path,
    input_dir: Path,
    output_dir: Path,
    stride: int,
    images_per_video: int | None,
    image_ext: str,
    device: str,
    ffmpeg: str | None,
) -> int:
    if device == "cuda":
        if ffmpeg is None:
            raise RuntimeError("CUDA frame extraction requested, but ffmpeg is not installed.")
        if not ffmpeg_has_cuda(ffmpeg):
            raise RuntimeError("CUDA frame extraction requested, but ffmpeg does not list cuda in -hwaccels.")
        return extract_video_cuda(video_path, input_dir, output_dir, stride, images_per_video, image_ext, ffmpeg)
    if device == "auto" and ffmpeg is not None and ffmpeg_has_cuda(ffmpeg):
        try:
            return extract_video_cuda(video_path, input_dir, output_dir, stride, images_per_video, image_ext, ffmpeg)
        except Exception as exc:
            print(f"CUDA frame extraction unavailable for {video_path.name}; falling back to CPU: {exc}", flush=True)
    return extract_video(video_path, input_dir, output_dir, stride, images_per_video, image_ext)


def main() -> int:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    stride = max(1, args.stride)
    ffmpeg = ffmpeg_path()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    videos = video_files(input_dir)
    if not videos:
        raise RuntimeError(f"No videos found in {input_dir}")

    total = 0
    if args.images_per_video is None:
        print(f"Found {len(videos)} video(s). Extracting every {stride} frame(s).", flush=True)
    else:
        print(f"Found {len(videos)} video(s). Extracting {max(1, args.images_per_video)} image(s) per video.", flush=True)
    for video_path in videos:
        total += extract_video_with_device(
            video_path,
            input_dir,
            output_dir,
            stride,
            args.images_per_video,
            args.image_ext,
            args.device,
            ffmpeg,
        )

    print(f"Frame extraction complete. Saved {total} image(s) to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
