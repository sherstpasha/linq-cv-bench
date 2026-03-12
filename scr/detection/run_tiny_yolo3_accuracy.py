import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torchvision.ops import batched_nms

from coco_utils import COCO80_TO_91, letterbox


REPO_ROOT = Path(__file__).resolve().parents[2]
BATCH_RE = re.compile(r"_b(\d+)")
DEFAULT_ANCHORS = [(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)]
DEFAULT_MASKS = [(3, 4, 5), (0, 1, 2)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vendor tiny_yolo3 TPU inference on COCO and compute bbox metrics")
    parser.add_argument(
        "--program-path",
        type=Path,
        default=Path("linq_files/tpu_programs/tiny_yolo3_b8_o5_128x128_asic.tpu"),
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
        default=REPO_ROOT / "experiments/detection/tiny_yolo3_vendor/predictions.json",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=REPO_ROOT / "experiments/detection/tiny_yolo3_vendor/accuracy_summary.json",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=REPO_ROOT / "experiments/detection/tiny_yolo3_vendor/metrics.json",
    )
    parser.add_argument(
        "--metrics-text",
        type=Path,
        default=REPO_ROOT / "experiments/detection/tiny_yolo3_vendor/metrics.txt",
    )
    parser.add_argument("--input-tensor-name", type=str, default=None)
    parser.add_argument("--output-tensor-name", type=str, default=None)
    parser.add_argument("--img-size", type=int, default=416)
    parser.add_argument("--input-layout", choices=["nchw", "nhwc"], default="nchw")
    parser.add_argument("--input-range", choices=["unit_float", "uint8"], default="unit_float")
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--warmup-images", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument(
        "--anchors",
        type=str,
        default="10,14 23,27 37,58 81,82 135,169 344,319",
        help="YOLO anchors as 'w,h w,h ...'",
    )
    parser.add_argument(
        "--masks",
        type=str,
        default="3,4,5|0,1,2",
        help="YOLO masks as '3,4,5|0,1,2' from coarse to fine head",
    )
    return parser.parse_args()


def parse_anchors(value: str) -> List[Tuple[float, float]]:
    anchors: List[Tuple[float, float]] = []
    for item in value.replace(";", " ").split():
        w_str, h_str = item.split(",")
        anchors.append((float(w_str), float(h_str)))
    if not anchors:
        raise RuntimeError("No anchors parsed")
    return anchors


def parse_masks(value: str) -> List[Tuple[int, ...]]:
    masks: List[Tuple[int, ...]] = []
    for group in value.split("|"):
        indices = tuple(int(x.strip()) for x in group.split(",") if x.strip())
        if indices:
            masks.append(indices)
    if not masks:
        raise RuntimeError("No masks parsed")
    return masks


def key_candidates(name: Optional[str]) -> List[str]:
    if not name:
        return []
    out = [name]
    if name.endswith(":0"):
        out.append(name[:-2])
    else:
        out.append(f"{name}:0")
    return out


def infer_batch_size(program_path: Path, requested_batch_size: int) -> int:
    if requested_batch_size > 0:
        return requested_batch_size
    match = BATCH_RE.search(program_path.name)
    if match:
        return int(match.group(1))
    return 1


def preprocess_image(path: Path, img_size: int, input_layout: str, input_range: str) -> Tuple[np.ndarray, Dict[str, float]]:
    img = cv2.imread(path.as_posix())
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img_h, img_w = img.shape[:2]
    img_lb, scale, pad_x, pad_y = letterbox(img, img_size)
    rgb = img_lb[:, :, ::-1]
    if input_layout == "nchw":
        x = rgb.transpose(2, 0, 1)
    else:
        x = rgb
    if input_range == "unit_float":
        x = x.astype(np.float32) / 255.0
    else:
        x = x.astype(np.uint8)
    x = np.expand_dims(x, axis=0)
    meta = {
        "img_h": float(img_h),
        "img_w": float(img_w),
        "scale": float(scale),
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
    }
    return x, meta


def make_batch(tensors: List[np.ndarray], batch_size: int) -> np.ndarray:
    x = np.concatenate(tensors, axis=0)
    if x.shape[0] == batch_size:
        return x
    if x.shape[0] > batch_size:
        return x[:batch_size]
    pad = np.repeat(x[-1:], repeats=batch_size - x.shape[0], axis=0)
    return np.concatenate([x, pad], axis=0)


def run_once(inference: object, input_name: str, x: np.ndarray) -> Dict[str, np.ndarray]:
    if hasattr(inference, "run"):
        return inference.run({input_name: x})  # type: ignore[attr-defined]
    return inference.sync({input_name: x})  # type: ignore[attr-defined]


def _as_name_list(obj: Any) -> List[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [str(k) for k in obj.keys()]
    if isinstance(obj, (list, tuple, set)):
        return [str(x) for x in obj]
    return []


def collect_runtime_input_hints(inference: object, tpu_program: object) -> List[str]:
    hints: List[str] = []
    for obj in (inference, tpu_program):
        for name in ("input_names", "inputs", "get_input_names", "get_inputs", "tensor_descriptions", "get_tensor_descriptions"):
            if not hasattr(obj, name):
                continue
            attr = getattr(obj, name)
            try:
                value = attr() if callable(attr) else attr
            except Exception:
                continue
            hints.extend(_as_name_list(value))
    hints.extend(["images", "images:0", "input", "input:0", "input.1", "Placeholder", "Placeholder:0"])

    uniq: List[str] = []
    seen = set()
    for item in hints:
        if item and item not in seen:
            seen.add(item)
            uniq.append(item)
    return uniq


def resolve_runtime_input_name(
    inference: object,
    preferred_input: Optional[str],
    probe_x: np.ndarray,
    runtime_hints: List[str],
) -> Tuple[str, Dict[str, np.ndarray]]:
    candidates = key_candidates(preferred_input) + runtime_hints
    tried = set()
    errors: Dict[str, str] = {}
    for name in candidates:
        if name in tried:
            continue
        tried.add(name)
        try:
            out = run_once(inference, name, probe_x)
            return name, out
        except Exception as error:
            errors[name] = str(error)
    sample_errors = {k: errors[k] for k in sorted(errors)[:4]}
    raise RuntimeError(f"Could not resolve input tensor name. Tried: {sorted(tried)}. Sample errors: {sample_errors}")


def candidate_outputs(output_dict: Dict[str, np.ndarray], preferred_name: Optional[str]) -> List[Tuple[str, np.ndarray]]:
    preferred = key_candidates(preferred_name)
    items = list(output_dict.items())
    if preferred:
        ordered: List[Tuple[str, np.ndarray]] = []
        seen = set()
        for key in preferred:
            if key in output_dict:
                ordered.append((key, output_dict[key]))
                seen.add(key)
        for key, value in items:
            if key not in seen:
                ordered.append((key, value))
        return ordered
    return items


def tensor_to_no_reshape(tensor: np.ndarray, num_classes: int) -> np.ndarray:
    no = num_classes + 5
    if tensor.ndim == 3:
        if tensor.shape[1] == no:
            return tensor.transpose(0, 2, 1)
        if tensor.shape[2] == no:
            return tensor
        raise RuntimeError(f"Unsupported combined tensor shape: {tensor.shape}")
    raise RuntimeError(f"Unsupported combined tensor rank: {tensor.ndim}")


def prepare_head_tensor(tensor: np.ndarray, num_classes: int) -> np.ndarray:
    no = num_classes + 5
    if tensor.ndim != 4:
        raise RuntimeError(f"Unsupported head rank: {tensor.ndim}, shape={tensor.shape}")
    if tensor.shape[1] % no == 0:
        b, c, h, w = tensor.shape
        na = c // no
        return tensor.reshape(b, na, no, h, w).transpose(0, 1, 3, 4, 2)
    if tensor.shape[-1] % no == 0:
        b, h, w, c = tensor.shape
        na = c // no
        return tensor.reshape(b, h, w, na, no).transpose(0, 3, 1, 2, 4)
    raise RuntimeError(f"Unsupported head tensor shape: {tensor.shape}")


def xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
    xy = xywh[:, :2]
    wh = xywh[:, 2:4]
    top_left = xy - wh / 2
    bottom_right = xy + wh / 2
    return torch.cat((top_left, bottom_right), dim=1)


def clip_boxes_xyxy(boxes: torch.Tensor, width: float, height: float) -> torch.Tensor:
    boxes[:, 0].clamp_(0, width - 1)
    boxes[:, 1].clamp_(0, height - 1)
    boxes[:, 2].clamp_(0, width - 1)
    boxes[:, 3].clamp_(0, height - 1)
    return boxes


def postprocess_combined(
    pred: np.ndarray,
    metas: List[Dict[str, float]],
    num_classes: int,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
) -> List[Dict[str, Any]]:
    pred_t = torch.from_numpy(tensor_to_no_reshape(pred, num_classes)).float()
    obj = pred_t[..., 4:5].sigmoid()
    cls = pred_t[..., 5:].sigmoid()
    boxes = pred_t[..., :4]

    detections: List[Dict[str, Any]] = []
    for batch_index, meta in enumerate(metas):
        cls_scores = obj[batch_index] * cls[batch_index]
        box_indices, cls_indices = torch.nonzero(cls_scores > conf_thres, as_tuple=True)
        if box_indices.numel() == 0:
            continue
        scores = cls_scores[box_indices, cls_indices]
        xyxy = xywh_to_xyxy(boxes[batch_index][box_indices].clone())
        keep = batched_nms(xyxy, scores, cls_indices, iou_thres)[:max_det]
        xyxy = xyxy[keep]
        scores = scores[keep]
        cls_indices = cls_indices[keep]

        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - meta["pad_x"]) / meta["scale"]
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - meta["pad_y"]) / meta["scale"]
        xyxy = clip_boxes_xyxy(xyxy, meta["img_w"], meta["img_h"])

        for box, score, cls_idx in zip(xyxy.tolist(), scores.tolist(), cls_indices.tolist()):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            if w <= 1.0 or h <= 1.0:
                continue
            coco_cls = COCO80_TO_91[int(cls_idx)]
            detections.append({
                "image_id": int(meta["image_id"]),
                "category_id": coco_cls,
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(score),
            })
    return detections


def postprocess_yolo_heads(
    head_tensors: Sequence[np.ndarray],
    metas: List[Dict[str, float]],
    img_size: int,
    anchors: List[Tuple[float, float]],
    masks: List[Tuple[int, ...]],
    num_classes: int,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
) -> List[Dict[str, Any]]:
    no = num_classes + 5
    prepared = [prepare_head_tensor(t, num_classes) for t in head_tensors]
    prepared.sort(key=lambda x: x.shape[2])
    if len(prepared) != len(masks):
        raise RuntimeError(f"Expected {len(masks)} YOLO heads, got {len(prepared)}")

    batch_outputs: List[torch.Tensor] = []
    for head, mask in zip(prepared, masks):
        pred = torch.from_numpy(head).float()
        b, na, gh, gw, _ = pred.shape
        if na != len(mask):
            raise RuntimeError(f"Head anchor count mismatch: head has {na}, mask has {len(mask)}")
        stride = img_size / gh
        device = pred.device
        yy, xx = torch.meshgrid(torch.arange(gh, device=device), torch.arange(gw, device=device), indexing="ij")
        grid = torch.stack((xx, yy), dim=-1).view(1, 1, gh, gw, 2).float()
        anchor_tensor = torch.tensor([anchors[idx] for idx in mask], device=device).view(1, na, 1, 1, 2).float()

        xy = (pred[..., 0:2].sigmoid() + grid) * stride
        wh = pred[..., 2:4].exp() * anchor_tensor
        obj = pred[..., 4:5].sigmoid()
        cls = pred[..., 5:].sigmoid()
        batch_outputs.append(torch.cat((xy, wh, obj, cls), dim=-1).view(b, -1, no))

    combined = torch.cat(batch_outputs, dim=1)
    detections: List[Dict[str, Any]] = []
    for batch_index, meta in enumerate(metas):
        cls_scores = combined[batch_index, :, 4:5] * combined[batch_index, :, 5:]
        box_indices, cls_indices = torch.nonzero(cls_scores > conf_thres, as_tuple=True)
        if box_indices.numel() == 0:
            continue
        scores = cls_scores[box_indices, cls_indices]
        xywh = combined[batch_index, box_indices, :4].clone()
        xyxy = xywh_to_xyxy(xywh)
        keep = batched_nms(xyxy, scores, cls_indices, iou_thres)[:max_det]
        xyxy = xyxy[keep]
        scores = scores[keep]
        cls_indices = cls_indices[keep]

        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - meta["pad_x"]) / meta["scale"]
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - meta["pad_y"]) / meta["scale"]
        xyxy = clip_boxes_xyxy(xyxy, meta["img_w"], meta["img_h"])

        for box, score, cls_idx in zip(xyxy.tolist(), scores.tolist(), cls_indices.tolist()):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            if w <= 1.0 or h <= 1.0:
                continue
            coco_cls = COCO80_TO_91[int(cls_idx)]
            detections.append({
                "image_id": int(meta["image_id"]),
                "category_id": coco_cls,
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(score),
            })
    return detections


def decode_predictions(
    output_dict: Dict[str, np.ndarray],
    preferred_output: Optional[str],
    metas: List[Dict[str, float]],
    img_size: int,
    anchors: List[Tuple[float, float]],
    masks: List[Tuple[int, ...]],
    num_classes: int,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    items = candidate_outputs(output_dict, preferred_output)
    probe = {key: list(np.asarray(value).shape) for key, value in items}

    combined_candidates = [(key, np.asarray(value)) for key, value in items if np.asarray(value).ndim == 3]
    for key, value in combined_candidates:
        try:
            detections = postprocess_combined(value, metas, num_classes, conf_thres, iou_thres, max_det)
            return detections, {"mode": "combined", "selected_outputs": [key], "output_shapes": probe}
        except Exception:
            continue

    head_candidates = [(key, np.asarray(value)) for key, value in items if np.asarray(value).ndim == 4]
    if len(head_candidates) >= len(masks):
        try:
            selected = head_candidates[: len(masks)]
            detections = postprocess_yolo_heads(
                [value for _, value in selected],
                metas,
                img_size,
                anchors,
                masks,
                num_classes,
                conf_thres,
                iou_thres,
                max_det,
            )
            return detections, {
                "mode": "yolo_heads",
                "selected_outputs": [key for key, _ in selected],
                "output_shapes": probe,
            }
        except Exception as error:
            raise RuntimeError(f"Failed to decode YOLO heads. Output shapes: {probe}. Error: {error}") from error

    raise RuntimeError(f"Could not decode outputs. Available shapes: {probe}")


def compute_coco_metrics(ann_file: Path, predictions_path: Path, metrics_out: Path, metrics_text: Path, limit: int) -> Dict[str, Any]:
    import contextlib
    import io
    from pycocotools.cocoeval import COCOeval

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_text.parent.mkdir(parents=True, exist_ok=True)
    coco = COCO(ann_file.as_posix())
    pred_rows = json.loads(predictions_path.read_text(encoding="utf-8"))
    if not pred_rows:
        metrics = {
            "ann_file": ann_file.as_posix(),
            "predictions": predictions_path.as_posix(),
            "num_images": limit if limit > 0 else len(coco.getImgIds()),
            "metrics": {
                "AP@[.50:.95]": 0.0,
                "AP@0.50": 0.0,
                "AP@0.75": 0.0,
                "AP_small": 0.0,
                "AP_medium": 0.0,
                "AP_large": 0.0,
                "AR@1": 0.0,
                "AR@10": 0.0,
                "AR@100": 0.0,
                "AR_small": 0.0,
                "AR_medium": 0.0,
                "AR_large": 0.0,
            },
        }
        metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        metrics_text.write_text("Predictions are empty; metrics set to 0.0.\n", encoding="utf-8")
        return metrics

    dt = coco.loadRes(predictions_path.as_posix())
    ev = COCOeval(coco, dt, "bbox")
    if limit > 0:
        ev.params.imgIds = coco.getImgIds()[:limit]
    ev.evaluate()
    ev.accumulate()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ev.summarize()
    metrics_text.write_text(buffer.getvalue(), encoding="utf-8")
    keys = [
        "AP@[.50:.95]",
        "AP@0.50",
        "AP@0.75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR@1",
        "AR@10",
        "AR@100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    values = [float(x) for x in ev.stats.tolist()]
    metrics = {
        "ann_file": ann_file.as_posix(),
        "predictions": predictions_path.as_posix(),
        "num_images": len(ev.params.imgIds) if ev.params.imgIds else len(coco.getImgIds()),
        "metrics": {key: value for key, value in zip(keys, values)},
    }
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    args = parse_args()
    if not args.program_path.exists():
        raise FileNotFoundError(f"TPU program not found: {args.program_path}")

    try:
        import pytpu as tpu  # type: ignore
    except Exception as error:
        raise RuntimeError("Missing dependency: pytpu") from error

    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    anchors = parse_anchors(args.anchors)
    masks = parse_masks(args.masks)
    batch_size = infer_batch_size(args.program_path, args.batch_size)

    coco = COCO(args.ann_file.as_posix())
    img_ids = coco.getImgIds()
    if args.limit > 0:
        img_ids = img_ids[: args.limit]

    devices = tpu.Device.list_devices()
    if not devices:
        raise RuntimeError("TPU device not found")
    device_id = args.device or devices[0]
    if device_id not in devices:
        raise RuntimeError(f"Requested device {device_id} not available. available={devices}")

    results: List[Dict[str, Any]] = []
    infer_time = 0.0
    measured_images = 0
    decode_probe: Optional[Dict[str, Any]] = None
    resolved_input_name: Optional[str] = None
    probe_output_keys: Optional[List[str]] = None

    with tpu.Device.open(device_id) as tpu_device:
        with tpu_device.load(args.program_path.as_posix()) as tpu_program:
            with tpu_program.inference() as inference:
                probe_img_info = coco.loadImgs(img_ids[0])[0]
                probe_x_single, _ = preprocess_image(args.img_dir / probe_img_info["file_name"], args.img_size, args.input_layout, args.input_range)
                probe_x = make_batch([probe_x_single], batch_size)
                runtime_hints = collect_runtime_input_hints(inference, tpu_program)
                resolved_input_name, probe_out = resolve_runtime_input_name(inference, args.input_tensor_name, probe_x, runtime_hints)
                probe_output_keys = list(probe_out.keys())

                num_batches = (len(img_ids) + batch_size - 1) // batch_size
                processed_images = 0
                for batch_idx in range(num_batches):
                    start = batch_idx * batch_size
                    end = min((batch_idx + 1) * batch_size, len(img_ids))
                    batch_ids = img_ids[start:end]

                    batch_tensors: List[np.ndarray] = []
                    metas: List[Dict[str, float]] = []
                    for img_id in batch_ids:
                        info = coco.loadImgs(img_id)[0]
                        x, meta = preprocess_image(args.img_dir / info["file_name"], args.img_size, args.input_layout, args.input_range)
                        batch_tensors.append(x)
                        meta["image_id"] = int(img_id)
                        metas.append(meta)

                    if not batch_tensors:
                        continue

                    x_batch = make_batch(batch_tensors, batch_size)
                    t0 = time.perf_counter()
                    output_dict = run_once(inference, resolved_input_name, x_batch)
                    t1 = time.perf_counter()
                    batch_results, decode_probe = decode_predictions(
                        output_dict,
                        args.output_tensor_name,
                        metas,
                        args.img_size,
                        anchors,
                        masks,
                        args.num_classes,
                        args.conf_thres,
                        args.iou_thres,
                        args.max_det,
                    )
                    results.extend(batch_results)

                    processed_images += len(metas)
                    if processed_images > args.warmup_images:
                        measured_in_batch = len(metas)
                        if processed_images - len(metas) < args.warmup_images:
                            measured_in_batch = processed_images - args.warmup_images
                        infer_time += t1 - t0
                        measured_images += measured_in_batch

    args.predictions_out.write_text(json.dumps(results), encoding="utf-8")
    metrics = compute_coco_metrics(args.ann_file, args.predictions_out, args.metrics_out, args.metrics_text, args.limit)
    summary = {
        "pipeline": "tiny_yolo3_vendor_direct_tpu",
        "program_path": args.program_path.as_posix(),
        "dataset_img_dir": args.img_dir.as_posix(),
        "ann_file": args.ann_file.as_posix(),
        "device": device_id,
        "batch_size": batch_size,
        "requested_batch_size": args.batch_size,
        "input_layout": args.input_layout,
        "input_range": args.input_range,
        "img_size": args.img_size,
        "effective_images": len(img_ids),
        "warmup_images": args.warmup_images,
        "detections": len(results),
        "measured_inference_sec": infer_time,
        "throughput_img_per_sec": measured_images / max(infer_time, 1e-9),
        "predictions_file": args.predictions_out.as_posix(),
        "metrics_json": args.metrics_out.as_posix(),
        "metrics_text": args.metrics_text.as_posix(),
        "metrics": metrics.get("metrics", {}),
        "resolved_input_name": resolved_input_name,
        "probe_output_keys": probe_output_keys,
        "decode_probe": decode_probe,
    }
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved tiny_yolo3 accuracy summary: {args.summary_out}")
    print(f"Saved tiny_yolo3 metrics: {args.metrics_out}")


if __name__ == "__main__":
    main()
