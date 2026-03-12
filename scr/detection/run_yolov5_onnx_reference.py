import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import onnxruntime as ort
import torch
from pycocotools.coco import COCO
from torchvision.ops import batched_nms
from tqdm import tqdm

from coco_utils import COCO80_TO_91, letterbox


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classic YOLOv5s ONNX inference on COCO and save predictions")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT / "models/detection/yolov5s.onnx",
    )
    parser.add_argument(
        "--img-dir",
        type=Path,
        default=REPO_ROOT / "data/evaluation/MSCOCO2017/val2017",
    )
    parser.add_argument(
        "--ann-file",
        type=Path,
        default=REPO_ROOT / "data/evaluation/MSCOCO2017/annotations/instances_val2017.json",
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=REPO_ROOT / "experiments/detection_onnx_reference/predictions.json",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=REPO_ROOT / "experiments/detection_onnx_reference/results_summary.json",
    )
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.65)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--warmup-images", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--providers", type=str, default=None)
    return parser.parse_args()


def resolve_providers(user_providers: Optional[str]) -> Sequence[str]:
    available = ort.get_available_providers()
    if user_providers:
        selected = [provider.strip() for provider in user_providers.split(",") if provider.strip() in available]
        if not selected:
            raise RuntimeError(f"No requested providers available. available={available}")
        return selected
    return [provider for provider in ["CUDAExecutionProvider", "CPUExecutionProvider"] if provider in available]


def normalize_prediction_shape(pred: np.ndarray) -> np.ndarray:
    if pred.ndim != 3:
        raise ValueError(f"Unexpected prediction rank: {pred.ndim}")
    if pred.shape[2] >= 6:
        return pred
    if pred.shape[1] >= 6:
        return pred.transpose(0, 2, 1)
    raise ValueError(f"Unexpected prediction shape: {pred.shape}")


def preprocess(path: Path, img_size: int) -> tuple[np.ndarray, tuple[int, int], float, int, int]:
    img = cv2.imread(path.as_posix())
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img_h, img_w = img.shape[:2]
    img_lb, scale, pad_x, pad_y = letterbox(img, img_size)
    x = img_lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(x, axis=0), (img_h, img_w), scale, pad_x, pad_y


def make_batch(tensors: List[np.ndarray], batch_size: int) -> np.ndarray:
    x = np.concatenate(tensors, axis=0).astype(np.float32)
    if x.shape[0] == batch_size:
        return x
    if x.shape[0] > batch_size:
        return x[:batch_size]
    pad = np.repeat(x[-1:], repeats=batch_size - x.shape[0], axis=0)
    return np.concatenate([x, pad], axis=0)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    out = boxes.clone()
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return out


def non_max_suppression(
    prediction: np.ndarray,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
) -> List[torch.Tensor]:
    pred = torch.from_numpy(prediction)
    outputs: List[torch.Tensor] = []

    for image_pred in pred:
        if image_pred.numel() == 0:
            outputs.append(torch.zeros((0, 6), dtype=torch.float32))
            continue

        boxes = xywh_to_xyxy(image_pred[:, :4])
        objectness = image_pred[:, 4:5]
        class_scores = image_pred[:, 5:]
        scores = objectness * class_scores
        conf, class_ids = scores.max(dim=1)
        keep = conf > conf_thres

        if keep.sum().item() == 0:
            outputs.append(torch.zeros((0, 6), dtype=torch.float32))
            continue

        boxes = boxes[keep]
        conf = conf[keep]
        class_ids = class_ids[keep]
        keep_idx = batched_nms(boxes, conf, class_ids, iou_thres)
        keep_idx = keep_idx[:max_det]

        detections = torch.cat(
            [
                boxes[keep_idx],
                conf[keep_idx].unsqueeze(1),
                class_ids[keep_idx].float().unsqueeze(1),
            ],
            dim=1,
        )
        outputs.append(detections)

    return outputs


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise RuntimeError("--batch-size must be > 0")
    if not args.model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {args.model_path}")

    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)

    coco = COCO(args.ann_file.as_posix())
    img_ids = coco.getImgIds()
    if args.limit > 0:
        img_ids = img_ids[: args.limit]

    providers = resolve_providers(args.providers)
    session = ort.InferenceSession(args.model_path.as_posix(), providers=list(providers))
    active_providers = session.get_providers()
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

    for batch_idx in tqdm(range(num_batches), desc="ONNX Inference"):
        start = batch_idx * effective_batch_size
        end = min((batch_idx + 1) * effective_batch_size, len(img_ids))
        batch_ids = img_ids[start:end]

        batch_tensors: List[np.ndarray] = []
        metas: List[Dict] = []
        for img_id in batch_ids:
            img_info = coco.loadImgs(img_id)[0]
            x, (img_h, img_w), scale, pad_x, pad_y = preprocess(args.img_dir / img_info["file_name"], args.img_size)
            batch_tensors.append(x)
            metas.append(
                {
                    "image_id": int(img_id),
                    "img_h": img_h,
                    "img_w": img_w,
                    "scale": scale,
                    "pad_x": pad_x,
                    "pad_y": pad_y,
                }
            )

        if not batch_tensors:
            continue

        x_batch = make_batch(batch_tensors, effective_batch_size)

        t0 = time.perf_counter()
        pred_raw = session.run(None, {input_name: x_batch})[0]
        t1 = time.perf_counter()
        pred = normalize_prediction_shape(np.asarray(pred_raw))
        valid_pred = pred[: len(batch_ids)]
        dets = non_max_suppression(
            valid_pred,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            max_det=args.max_det,
        )

        for det_idx, det in enumerate(dets):
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
                x1 = max(0.0, min(float(x1), meta["img_w"] - 1.0))
                y1 = max(0.0, min(float(y1), meta["img_h"] - 1.0))
                x2 = max(0.0, min(float(x2), meta["img_w"] - 1.0))
                y2 = max(0.0, min(float(y2), meta["img_h"] - 1.0))
                w, h = x2 - x1, y2 - y1
                if w <= 1.0 or h <= 1.0:
                    continue
                results.append(
                    {
                        "image_id": meta["image_id"],
                        "category_id": COCO80_TO_91[cls_idx],
                        "bbox": [x1, y1, w, h],
                        "score": float(conf),
                    }
                )

        processed_images += len(metas)
        if processed_images > args.warmup_images:
            measured_in_batch = len(metas)
            if processed_images - len(metas) < args.warmup_images:
                measured_in_batch = processed_images - args.warmup_images
            infer_time += t1 - t0
            measured_images += measured_in_batch

    args.predictions_out.write_text(json.dumps(results), encoding="utf-8")

    timing = {
        "pipeline": "yolov5s_onnx_reference",
        "providers": list(active_providers),
        "batch_size": effective_batch_size,
        "requested_batch_size": args.batch_size,
        "images": len(img_ids),
        "warmup_images": args.warmup_images,
        "detections": len(results),
        "measured_inference_sec": infer_time,
        "throughput_img_per_sec": measured_images / max(infer_time, 1e-9),
        "predictions_file": args.predictions_out.as_posix(),
        "model_path": args.model_path.as_posix(),
        "input_name": input_name,
    }
    args.summary_out.write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
