import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import onnxruntime as ort
from pycocotools.coco import COCO
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FasterRCNN ONNX inference on COCO and save predictions")
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "experiments/detection_model/fasterrcnn_resnet50_fpn.onnx")
    parser.add_argument("--img-dir", type=Path, default=REPO_ROOT / "data/evaluation/MSCOCO2017/val2017")
    parser.add_argument("--ann-file", type=Path, default=REPO_ROOT / "data/evaluation/MSCOCO2017/annotations/instances_val2017.json")
    parser.add_argument("--predictions-out", type=Path, default=REPO_ROOT / "experiments/detection_model/predictions_onnx.json")
    parser.add_argument("--timing-out", type=Path, default=REPO_ROOT / "experiments/detection_model/inference_timing_onnx.json")
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup-images", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--providers", type=str, default=None)
    return parser.parse_args()


def resolve_providers(user_providers: Optional[str]) -> Sequence[str]:
    available = ort.get_available_providers()
    if user_providers:
        selected = [p.strip() for p in user_providers.split(",") if p.strip() in available]
        if not selected:
            raise RuntimeError(f"No requested providers available. available={available}")
        return selected
    return [p for p in ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]


def preprocess(path: Path, width: int, height: int) -> tuple[np.ndarray, tuple[int, int]]:
    img = cv2.imread(path.as_posix())
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img_h, img_w = img.shape[:2]
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    x = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(x, axis=0), (img_h, img_w)


def make_batch(tensors: List[np.ndarray], batch_size: int) -> np.ndarray:
    x = np.concatenate(tensors, axis=0)
    if x.shape[0] == batch_size:
        return x
    if x.shape[0] > batch_size:
        return x[:batch_size]
    pad = np.repeat(x[-1:], repeats=batch_size - x.shape[0], axis=0)
    return np.concatenate([x, pad], axis=0)


def normalize_detections_shape(dets: np.ndarray) -> np.ndarray:
    arr = np.asarray(dets)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)
    if arr.ndim != 3 or arr.shape[-1] != 6:
        raise RuntimeError(f"Unexpected detections shape: {arr.shape}")
    return arr


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise RuntimeError("--batch-size must be > 0")

    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.timing_out.parent.mkdir(parents=True, exist_ok=True)

    coco = COCO(args.ann_file.as_posix())
    valid_coco_ids = set(coco.getCatIds())
    img_ids = coco.getImgIds()
    if args.limit > 0:
        img_ids = img_ids[: args.limit]

    providers = resolve_providers(args.providers)
    session = ort.InferenceSession(args.model_path.as_posix(), providers=list(providers))
    active_providers = session.get_providers()
    print(f"Active providers: {active_providers}")

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = list(input_meta.shape)

    effective_batch_size = args.batch_size
    fixed_batch = input_shape[0]
    if isinstance(fixed_batch, int) and fixed_batch > 0 and fixed_batch != effective_batch_size:
        print(f"Requested batch_size={effective_batch_size}, but model batch is fixed to {fixed_batch}. Using {fixed_batch}.")
        effective_batch_size = fixed_batch

    results: List[Dict] = []
    infer_time = 0.0
    measured_images = 0

    num_batches = (len(img_ids) + effective_batch_size - 1) // effective_batch_size
    processed_images = 0

    for batch_idx in tqdm(range(num_batches), desc="Inference"):
        start = batch_idx * effective_batch_size
        end = min((batch_idx + 1) * effective_batch_size, len(img_ids))
        batch_ids = img_ids[start:end]

        batch_tensors: List[np.ndarray] = []
        metas: List[tuple[int, int, int]] = []
        for img_id in batch_ids:
            info = coco.loadImgs(img_id)[0]
            x, (orig_h, orig_w) = preprocess(args.img_dir / info["file_name"], width=args.width, height=args.height)
            batch_tensors.append(x)
            metas.append((int(img_id), orig_h, orig_w))

        if not batch_tensors:
            continue

        x_batch = make_batch(batch_tensors, effective_batch_size)
        t0 = time.perf_counter()
        out = session.run(None, {input_name: x_batch})[0]
        t1 = time.perf_counter()

        dets = normalize_detections_shape(out)[: len(metas)]

        for det, (img_id, orig_h, orig_w) in zip(dets, metas):
            sx = float(orig_w) / float(args.width)
            sy = float(orig_h) / float(args.height)
            for x1, y1, x2, y2, score, label in det:
                score = float(score)
                if score < args.conf_thres:
                    continue
                cat_id = int(round(float(label)))
                if cat_id not in valid_coco_ids:
                    continue
                x1 = max(0.0, min(float(x1) * sx, orig_w - 1.0))
                y1 = max(0.0, min(float(y1) * sy, orig_h - 1.0))
                x2 = max(0.0, min(float(x2) * sx, orig_w - 1.0))
                y2 = max(0.0, min(float(y2) * sy, orig_h - 1.0))
                w, h = x2 - x1, y2 - y1
                if w <= 1.0 or h <= 1.0:
                    continue
                results.append({"image_id": img_id, "category_id": cat_id, "bbox": [x1, y1, w, h], "score": score})

        processed_images += len(metas)
        if processed_images > args.warmup_images:
            measured_in_batch = len(metas)
            if processed_images - len(metas) < args.warmup_images:
                measured_in_batch = processed_images - args.warmup_images
            infer_time += (t1 - t0)
            measured_images += measured_in_batch

    with args.predictions_out.open("w", encoding="utf-8") as f:
        json.dump(results, f)

    timing = {
        "providers": list(active_providers),
        "batch_size": effective_batch_size,
        "requested_batch_size": args.batch_size,
        "images": len(img_ids),
        "warmup_images": args.warmup_images,
        "detections": len(results),
        "measured_inference_sec": infer_time,
        "throughput_img_per_sec": measured_images / max(infer_time, 1e-9),
        "predictions_file": args.predictions_out.as_posix(),
        "input_size": [args.height, args.width],
    }
    with args.timing_out.open("w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
