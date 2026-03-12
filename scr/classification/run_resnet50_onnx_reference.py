import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ONNX Runtime reference inference for the current ResNet-50 export"
    )
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "models/classification/resnet50.onnx")
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "data/evaluation/imagenet")
    parser.add_argument("--val-map", type=Path, default=None)
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=REPO_ROOT / "experiments/classification_onnx_reference/predictions.jsonl",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=REPO_ROOT / "experiments/classification_onnx_reference/results_summary.json",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=0, help="0 means all rows from val_map.txt")
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--providers", type=str, default=None)
    return parser.parse_args()


def load_export_metadata(model_path: Path) -> Dict:
    metadata_path = model_path.with_suffix(".json")
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_val_rows(path: Path) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image_name, label = line.split()
            rows.append((image_name, int(label)))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def resolve_providers(user_providers: str | None) -> Sequence[str]:
    available = ort.get_available_providers()
    if user_providers:
        selected = [p.strip() for p in user_providers.split(",") if p.strip() in available]
        if not selected:
            raise RuntimeError(f"No requested providers available. available={available}")
        return selected
    ordered = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    selected = [p for p in ordered if p in available]
    return selected or available


def resize_center_crop(image: Image.Image, size: int = 224, resize_shorter: int = 256) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = resize_shorter / min(width, height)
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))
    image = image.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return image.crop((left, top, left + size, top + size))


def preprocess_image(image_path: Path, metadata: Dict) -> np.ndarray:
    input_layout = metadata.get("input_layout", "nchw")
    input_value_range = metadata.get("input_value_range", "normalized")

    with Image.open(image_path) as image:
        image = resize_center_crop(image)
        arr = np.asarray(image, dtype=np.float32)

    if input_value_range == "uint8":
        pass
    elif input_value_range == "unit_float":
        arr = arr / 255.0
    elif input_value_range == "normalized":
        arr = arr / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    else:
        raise RuntimeError(f"Unsupported input_value_range: {input_value_range}")

    if input_layout == "nchw":
        arr = np.transpose(arr, (2, 0, 1))
    elif input_layout != "nhwc":
        raise RuntimeError(f"Unsupported input_layout: {input_layout}")

    return arr.astype(np.float32)


def top5_indices(logits: np.ndarray) -> np.ndarray:
    top5 = np.argpartition(logits, -5, axis=1)[:, -5:]
    top5_scores = np.take_along_axis(logits, top5, axis=1)
    order = np.argsort(-top5_scores, axis=1)
    return np.take_along_axis(top5, order, axis=1)


def resolve_effective_batch_size(session: ort.InferenceSession, metadata: Dict, requested_batch_size: int) -> int:
    input_meta = session.get_inputs()[0]
    input_shape = list(input_meta.shape)
    batch_dim = input_shape[0] if input_shape else None

    if isinstance(batch_dim, int) and batch_dim > 0:
        return batch_dim

    metadata_batch = metadata.get("batch_size")
    if isinstance(metadata_batch, int) and metadata_batch > 0:
        return metadata_batch

    return requested_batch_size


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {args.dataset_dir}")
    val_map = args.val_map or (args.dataset_dir / "val_map.txt")
    if not val_map.exists():
        raise FileNotFoundError(f"val_map.txt not found: {val_map}")
    if args.batch_size <= 0:
        raise RuntimeError("--batch-size must be > 0")

    metadata = load_export_metadata(args.model_path)
    rows = load_val_rows(val_map)
    if args.samples > 0:
        rows = rows[: min(args.samples, len(rows))]

    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)

    providers = list(resolve_providers(args.providers))
    session = ort.InferenceSession(args.model_path.as_posix(), providers=providers)
    input_name = session.get_inputs()[0].name
    active_providers = session.get_providers()
    effective_batch_size = resolve_effective_batch_size(session, metadata, args.batch_size)

    infer_time = 0.0
    measured_images = 0
    measured_batches = 0
    good_top1 = 0
    good_top5 = 0

    with args.predictions_out.open("w", encoding="utf-8") as out_file:
        for batch_idx, start in enumerate(range(0, len(rows), effective_batch_size)):
            batch_rows = rows[start : start + effective_batch_size]
            batch_inputs = []
            batch_labels = []
            batch_names = []
            for image_name, raw_label in batch_rows:
                image_path = args.dataset_dir / image_name
                if not image_path.exists():
                    raise FileNotFoundError(f"Image from val_map.txt not found: {image_path}")
                batch_inputs.append(preprocess_image(image_path, metadata))
                batch_labels.append(raw_label - 1)
                batch_names.append(image_name)

            x = np.stack(batch_inputs, axis=0).astype(np.float32)
            t0 = time.perf_counter()
            logits = session.run(None, {input_name: x})[0]
            t1 = time.perf_counter()

            if batch_idx >= args.warmup_batches:
                infer_time += t1 - t0
                measured_images += len(batch_rows)
                measured_batches += 1

            top5 = top5_indices(np.asarray(logits))
            top1 = top5[:, 0]
            for image_name, label, pred1, pred5 in zip(batch_names, batch_labels, top1, top5):
                pred1 = int(pred1)
                pred5_list = [int(v) for v in pred5.tolist()]
                if pred1 == label:
                    good_top1 += 1
                if label in pred5_list:
                    good_top5 += 1
                out_file.write(json.dumps({"image": image_name, "top5": pred5_list}) + "\n")

    total = len(rows)
    summary = {
        "model_path": args.model_path.as_posix(),
        "metadata": metadata or None,
        "dataset_dir": args.dataset_dir.as_posix(),
        "val_map": val_map.as_posix(),
        "effective_samples": total,
        "providers": active_providers,
        "batch_size": effective_batch_size,
        "requested_batch_size": args.batch_size,
        "warmup_batches": args.warmup_batches,
        "measured_inference_sec": infer_time,
        "throughput_img_per_sec": measured_images / max(infer_time, 1e-9),
        "avg_batch_latency_ms": (infer_time / max(measured_batches, 1)) * 1000.0,
        "predictions_file": args.predictions_out.as_posix(),
        "top1_accuracy": 100.0 * good_top1 / total,
        "top5_accuracy": 100.0 * good_top5 / total,
        "good_top1": good_top1,
        "good_top5": good_top5,
        "total": total,
    }
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
