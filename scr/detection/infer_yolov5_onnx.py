import argparse
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

os.environ.setdefault("YOLO_AUTOINSTALL", "False")

from coco_utils import COCO80_TO_91, letterbox

if TYPE_CHECKING:
    from pycocotools.coco import COCO  # pragma: no cover

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv5 ONNX inference on COCO and save predictions")
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "experiments/detection/yolov5su.onnx")
    parser.add_argument("--img-dir", type=Path, default=REPO_ROOT / "data/evaluation/MSCOCO2017/val2017")
    parser.add_argument("--ann-file", type=Path, default=REPO_ROOT / "data/evaluation/MSCOCO2017/annotations/instances_val2017.json")
    parser.add_argument("--predictions-out", type=Path, default=REPO_ROOT / "experiments/detection/predictions.json")
    parser.add_argument("--timing-out", type=Path, default=REPO_ROOT / "experiments/detection/inference_timing.json")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.65)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup-images", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
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


def normalize_prediction_shape(pred: np.ndarray) -> np.ndarray:
    if pred.ndim != 3:
        raise ValueError(f"Unexpected prediction rank: {pred.ndim}")
    if pred.shape[1] in (84, 85):
        return pred
    if pred.shape[2] in (84, 85):
        return pred.transpose(0, 2, 1)
    raise ValueError(f"Unexpected prediction shape: {pred.shape}")


def main() -> None:
    args = parse_args()
    try:
        import torch
        from pycocotools.coco import COCO
        from ultralytics.utils.nms import non_max_suppression
    except Exception as e:
        raise RuntimeError("Missing dependencies: torch, pycocotools, ultralytics") from e

    if args.batch_size <= 0:
        raise RuntimeError("--batch-size must be > 0")
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.timing_out.parent.mkdir(parents=True, exist_ok=True)

    coco = COCO(args.ann_file.as_posix())
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
    if isinstance(fixed_batch, int) and fixed_batch > 0 and effective_batch_size != fixed_batch:
        print(
            f"Requested batch_size={effective_batch_size}, but model input batch is fixed to {fixed_batch}. "
            f"Using batch_size={fixed_batch}."
        )
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
        metas: List[Dict] = []
        for img_id in batch_ids:
            img_info = coco.loadImgs(img_id)[0]
            img = cv2.imread((args.img_dir / img_info["file_name"]).as_posix())
            if img is None:
                continue

            img_lb, scale, pad_x, pad_y = letterbox(img, args.img_size)
            x = img_lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
            batch_tensors.append(x)
            metas.append(
                {
                    "image_id": int(img_id),
                    "shape": img.shape,
                    "scale": scale,
                    "pad_x": pad_x,
                    "pad_y": pad_y,
                }
            )

        if not batch_tensors:
            continue

        x_batch = np.stack(batch_tensors, axis=0).astype(np.float32)

        t0 = time.perf_counter()
        pred = normalize_prediction_shape(session.run(None, {input_name: x_batch})[0])
        t1 = time.perf_counter()

        dets = non_max_suppression(torch.from_numpy(pred), conf_thres=args.conf_thres, iou_thres=args.iou_thres, max_det=args.max_det)
        for det_idx, det in enumerate(dets):
            if det_idx >= len(metas):
                break
            meta = metas[det_idx]
            if det is None or len(det) == 0:
                continue
            for x1, y1, x2, y2, conf, cls in det.cpu().numpy():
                cls_idx = int(cls)
                if cls_idx < 0 or cls_idx >= len(COCO80_TO_91):
                    continue
                x1 = (x1 - meta["pad_x"]) / meta["scale"]
                y1 = (y1 - meta["pad_y"]) / meta["scale"]
                x2 = (x2 - meta["pad_x"]) / meta["scale"]
                y2 = (y2 - meta["pad_y"]) / meta["scale"]
                img_h, img_w = meta["shape"][:2]
                x1 = max(0.0, min(float(x1), img_w - 1.0))
                y1 = max(0.0, min(float(y1), img_h - 1.0))
                x2 = max(0.0, min(float(x2), img_w - 1.0))
                y2 = max(0.0, min(float(y2), img_h - 1.0))
                w, h = x2 - x1, y2 - y1
                if w <= 1.0 or h <= 1.0:
                    continue
                results.append({"image_id": meta["image_id"], "category_id": COCO80_TO_91[cls_idx], "bbox": [x1, y1, w, h], "score": float(conf)})

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
    }
    with args.timing_out.open("w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
