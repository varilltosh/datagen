from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.engine.tuner import Tuner
from ultralytics.utils import YAML


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_DIR / "workspace" / "data" / "sythesized_data.yaml"
DEFAULT_EVALUATION_DIR = PROJECT_DIR / "evaluation"
IMAGE_EXTENSIONS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm", ".heic"}
INT_HYPERPARAMETERS = {"close_mosaic"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune YOLO hyperparameters, then train with the best tuned settings."
    )
    parser.add_argument("--model", default=str(PROJECT_DIR / "models" / "yolov8s.pt"), help="Base YOLO model or checkpoint.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Dataset YAML path.")
    parser.add_argument("--project", type=Path, default=DEFAULT_EVALUATION_DIR, help="Directory for all results.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Final training and tuning batch size.")
    parser.add_argument("--epochs", type=int, default=200, help="Final training epochs.")
    parser.add_argument("--patience", type=int, default=20, help="Final training early-stop patience.")
    parser.add_argument("--tune-epochs", type=int, default=10, help="Epochs per tuning iteration.")
    parser.add_argument("--tune-iterations", type=int, default=30, help="Number of tuning iterations.")
    parser.add_argument("--device", default=0, help="Training device, for example 0, cpu, or cuda:0.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--skip-tune",
        action="store_true",
        help="Skip tuning and train directly from an existing best_hyperparameters.yaml.",
    )
    parser.add_argument(
        "--tune-dir",
        type=Path,
        default=None,
        help="Tune output directory containing best_hyperparameters.yaml. Defaults to PROJECT/tune.",
    )
    parser.add_argument("--resume-tune", action="store_true", help="Resume an existing tune run if present.")
    parser.add_argument(
        "--keep-tune-weights",
        action="store_true",
        help="Keep all tuning iteration weights instead of only the best weights.",
    )
    return parser.parse_args()


def validate_paths(data_path: Path, project_dir: Path) -> tuple[Path, Path]:
    data_path = data_path.expanduser().resolve()
    project_dir = project_dir.expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data_path}")
    project_dir.mkdir(parents=True, exist_ok=True)
    return data_path, project_dir


def remove_dataset_caches(dataset_root: Path) -> None:
    for cache_path in dataset_root.rglob("*.cache"):
        cache_path.unlink()
    for cache_path in dataset_root.rglob("*.cache.npy"):
        cache_path.unlink()


def image_files(path: Path) -> list[Path]:
    return sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def link_split_images(source_dir: Path, output_dir: Path) -> int:
    if output_dir.exists() or output_dir.is_symlink():
        if output_dir.is_symlink() or output_dir.is_file():
            output_dir.unlink()
        else:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    linked = 0
    for source_image in image_files(source_dir):
        target = output_dir / source_image.name
        try:
            target.symlink_to(source_image.resolve())
        except OSError:
            shutil.copy2(source_image, target)
        linked += 1
    return linked


def prepare_yolo_dataset_layout(data_path: Path) -> None:
    data = YAML.load(data_path)
    dataset_root = Path(data.get("path", data_path.parent)).expanduser().resolve()
    source_root = dataset_root / "sythesized_data"
    labels_root = dataset_root / "labels"
    images_root = dataset_root / "images"

    if not source_root.exists() or not labels_root.exists():
        return

    if images_root.is_symlink() or images_root.is_file():
        images_root.unlink()
    images_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "validate"):
        source_dir = source_root / split
        label_dir = labels_root / split
        if not source_dir.exists() or not label_dir.exists():
            continue
        linked = link_split_images(source_dir, images_root / split)
        label_count = len(list(label_dir.glob("*.txt")))
        if linked == 0:
            raise RuntimeError(f"No images found for split '{split}' in {source_dir}")
        if label_count == 0:
            raise RuntimeError(f"No labels found for split '{split}' in {label_dir}")
        if linked != label_count:
            raise RuntimeError(
                f"Image/label count mismatch for split '{split}': {linked} image(s), {label_count} label(s)"
            )

    remove_dataset_caches(dataset_root)


def tuner_args(args: argparse.Namespace, data_path: Path, project_dir: Path) -> dict:
    cfg = dict(vars(DEFAULT_CFG))
    cfg.update(
        {
            "task": "detect",
            "mode": "train",
            "model": args.model,
            "data": str(data_path),
            "epochs": args.tune_epochs,
            "patience": args.tune_epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "device": args.device,
            "workers": args.workers,
            "project": str(project_dir),
            "name": "tune",
            "resume": args.resume_tune,
            "seed": args.seed,
            "plots": True,
            "save": True,
            "val": True,
        }
    )
    return cfg


def load_best_hyperparameters(tune_dir: Path) -> dict:
    best_yaml = tune_dir / "best_hyperparameters.yaml"
    if not best_yaml.exists():
        raise FileNotFoundError(f"Tuning did not produce best hyperparameters: {best_yaml}")
    best_hyp = YAML.load(best_yaml)
    if not isinstance(best_hyp, dict) or not best_hyp:
        raise RuntimeError(f"Best hyperparameters file is empty or invalid: {best_yaml}")
    normalize_hyperparameter_types(best_hyp)
    return best_hyp


def normalize_hyperparameter_types(hyperparameters: dict) -> None:
    for key in INT_HYPERPARAMETERS:
        value = hyperparameters.get(key)
        if isinstance(value, float) and value.is_integer():
            hyperparameters[key] = int(value)


def train_final(
    args: argparse.Namespace,
    data_path: Path,
    project_dir: Path,
    best_hyp: dict,
) -> Path:
    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=str(project_dir),
        name="train_best_tuned",
        exist_ok=True,
        seed=args.seed,
        plots=True,
        **best_hyp,
    )
    return Path(results.save_dir).resolve()


def write_summary(
    project_dir: Path,
    args: argparse.Namespace,
    data_path: Path,
    tune_dir: Path,
    train_dir: Path,
    best_hyp: dict,
) -> None:
    summary = {
        "model": args.model,
        "data": str(data_path),
        "tune": {
            "epochs_per_iteration": args.tune_epochs,
            "iterations": args.tune_iterations,
            "directory": str(tune_dir),
            "best_hyperparameters": best_hyp,
        },
        "train": {
            "epochs": args.epochs,
            "batch": args.batch,
            "patience": args.patience,
            "directory": str(train_dir),
        },
    }
    summary_path = project_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Summary saved to {summary_path}")


def main() -> int:
    args = parse_args()
    data_path, project_dir = validate_paths(args.data, args.project)
    prepare_yolo_dataset_layout(data_path)

    print(f"Dataset: {data_path}")
    print(f"Evaluation directory: {project_dir}")

    tune_dir = args.tune_dir.expanduser().resolve() if args.tune_dir else project_dir / "tune"
    if args.skip_tune:
        print(f"Skipping tuning; using hyperparameters from {tune_dir / 'best_hyperparameters.yaml'}")
    else:
        print(
            f"Starting tuning: {args.tune_iterations} iteration(s), "
            f"{args.tune_epochs} epoch(s) each, batch={args.batch}"
        )
        tuner = Tuner(args=tuner_args(args, data_path, project_dir))
        tuner(model=YOLO(args.model), iterations=args.tune_iterations, cleanup=not args.keep_tune_weights)
        tune_dir = Path(tuner.tune_dir).resolve()

    best_hyp = load_best_hyperparameters(tune_dir)
    print(f"Loaded best hyperparameters from {tune_dir / 'best_hyperparameters.yaml'}")
    print(f"Starting final training: epochs={args.epochs}, batch={args.batch}, patience={args.patience}")

    train_dir = train_final(args, data_path, project_dir, best_hyp)
    write_summary(project_dir, args, data_path, tune_dir.resolve(), train_dir, best_hyp)
    print(f"Final training results saved to {train_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
