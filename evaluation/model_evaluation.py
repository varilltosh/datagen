#!/usr/bin/env python3
"""Compare YOLO models on the collected ground-truth dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parents[1]
EVALUATION_DIR = PROJECT_DIR / "evaluation"
DEFAULT_TUNED_WEIGHTS = EVALUATION_DIR / "train_best_tuned" / "weights" / "best.pt"
DEFAULT_DEFAULT_WEIGHTS = EVALUATION_DIR / "train_default_param" / "yolov8s_default" / "weights" / "best.pt"
DEFAULT_GROUND_TRUTH_ROOT = PROJECT_DIR / "website" / "data" / "ground_truth"
DEFAULT_LABEL_ROOT = DEFAULT_GROUND_TRUTH_ROOT / "ground_truth_labels"
DEFAULT_DATASET_DIR = EVALUATION_DIR / "ground_truth_eval_dataset"
DEFAULT_OUTPUT_DIR = EVALUATION_DIR / "model_evaluation_results"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and compare YOLO models on ground-truth images.")
    parser.add_argument("--tuned-weights", type=Path, default=DEFAULT_TUNED_WEIGHTS)
    parser.add_argument("--default-weights", type=Path, default=DEFAULT_DEFAULT_WEIGHTS)
    parser.add_argument("--ground-truth-root", type=Path, default=DEFAULT_GROUND_TRUTH_ROOT)
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="Example: 0, cuda:0, or cpu. Default lets YOLO choose.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU for Ultralytics validation.")
    parser.add_argument("--conf", type=float, default=0.001, help="Validation confidence threshold.")
    return parser.parse_args()


def image_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("ground_truth_image"))


def image_files(path: Path) -> list[Path]:
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def load_class_names(label_root: Path) -> list[str]:
    for class_file in sorted(label_root.glob("*/classes.json")):
        names = json.loads(class_file.read_text(encoding="utf-8"))
        if isinstance(names, list) and names:
            return [str(name) for name in names]
    max_class_id = -1
    for label_file in label_root.glob("*/*.txt"):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                max_class_id = max(max_class_id, int(float(parts[0])))
            except ValueError:
                continue
    if max_class_id < 0:
        raise RuntimeError(f"No class names or YOLO labels found in {label_root}")
    return [str(index) for index in range(max_class_id + 1)]


def link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def prepare_dataset(ground_truth_root: Path, label_root: Path, dataset_dir: Path) -> tuple[Path, int, int]:
    if not ground_truth_root.is_dir():
        raise FileNotFoundError(f"Ground-truth image root not found: {ground_truth_root}")
    if not label_root.is_dir():
        raise FileNotFoundError(f"Ground-truth label root not found: {label_root}")

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    image_out = dataset_dir / "images" / "val"
    label_out = dataset_dir / "labels" / "val"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    image_count = 0
    missing_label_count = 0
    for folder in image_dirs(ground_truth_root):
        source_label_dir = label_root / folder.name
        for image_path in image_files(folder):
            stem = f"{folder.name}__{image_path.stem}"
            target_image = image_out / f"{stem}{image_path.suffix.lower()}"
            target_label = label_out / f"{stem}.txt"
            link_or_copy(image_path, target_image)
            source_label = source_label_dir / f"{image_path.stem}.txt"
            if source_label.exists():
                shutil.copy2(source_label, target_label)
            else:
                target_label.write_text("", encoding="utf-8")
                missing_label_count += 1
            image_count += 1

    if image_count == 0:
        raise RuntimeError(f"No ground-truth images found under {ground_truth_root}")

    class_names = load_class_names(label_root)
    data_yaml = dataset_dir / "ground_truth_eval.yaml"
    names_block = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_dir}",
                "train: images/val",
                "val: images/val",
                f"nc: {len(class_names)}",
                "names:",
                names_block,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml, image_count, missing_label_count


def get_metric(metrics: Any, key: str, default: float = 0.0) -> float:
    results = getattr(metrics, "results_dict", {}) or {}
    if key in results:
        return float(results[key])
    return default


def summarize_metrics(name: str, weights: Path, metrics: Any) -> dict[str, float | str]:
    box = getattr(metrics, "box", None)
    precision = float(getattr(box, "mp", get_metric(metrics, "metrics/precision(B)", 0.0)))
    recall = float(getattr(box, "mr", get_metric(metrics, "metrics/recall(B)", 0.0)))
    map50 = float(getattr(box, "map50", get_metric(metrics, "metrics/mAP50(B)", 0.0)))
    map5095 = float(getattr(box, "map", get_metric(metrics, "metrics/mAP50-95(B)", 0.0)))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    accuracy = 0.0
    if precision > 0 and recall > 0:
        # Detection accuracy here is TP / (TP + FP + FN), derivable from precision and recall.
        accuracy = 1.0 / ((1.0 / precision) + (1.0 / recall) - 1.0)
    return {
        "model": name,
        "weights": str(weights),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map50_95": map5095,
        "accuracy": accuracy,
    }


def evaluate_model(name: str, weights: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, float | str]:
    if not weights.is_file():
        raise FileNotFoundError(f"{name} weights not found: {weights}")
    model = YOLO(str(weights))
    val_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "split": "val",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "conf": args.conf,
        "iou": args.iou,
        "plots": False,
        "save_json": False,
        "project": str(args.output_dir),
        "name": name,
        "exist_ok": True,
        "verbose": False,
    }
    if args.device is not None:
        val_kwargs["device"] = args.device
    metrics = model.val(**val_kwargs)
    return summarize_metrics(name, weights, metrics)


def write_reports(rows: list[dict[str, float | str]], output_dir: Path, image_count: int, missing_label_count: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparison_metrics.json"
    csv_path = output_dir / "comparison_metrics.csv"
    md_path = output_dir / "comparison_metrics.md"

    payload = {
        "image_count": image_count,
        "missing_label_count": missing_label_count,
        "metrics": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    fields = ["model", "precision", "recall", "f1", "map50", "map50_95", "accuracy", "weights"]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Model Evaluation",
        "",
        f"- Images: {image_count}",
        f"- Missing labels replaced with empty files: {missing_label_count}",
        "",
        "| Model | Precision | Recall | F1 | mAP50 | mAP50-95 | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {map50:.4f} | {map50_95:.4f} | {accuracy:.4f} |".format(
                **row
            )
        )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.ground_truth_root = args.ground_truth_root.expanduser().resolve()
    args.label_root = args.label_root.expanduser().resolve()
    args.dataset_dir = args.dataset_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.tuned_weights = args.tuned_weights.expanduser().resolve()
    args.default_weights = args.default_weights.expanduser().resolve()

    data_yaml, image_count, missing_label_count = prepare_dataset(
        args.ground_truth_root,
        args.label_root,
        args.dataset_dir,
    )
    rows = [
        evaluate_model("best_tuned", args.tuned_weights, data_yaml, args),
        evaluate_model("default_param", args.default_weights, data_yaml, args),
    ]
    write_reports(rows, args.output_dir, image_count, missing_label_count)

    print(f"Evaluated {image_count} image(s). Reports saved to {args.output_dir}", flush=True)
    for row in rows:
        print(
            "{model}: precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}, "
            "map50={map50:.4f}, map50-95={map50_95:.4f}, accuracy={accuracy:.4f}".format(**row),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
