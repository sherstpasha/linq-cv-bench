import argparse
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

os.environ.setdefault("YOLO_AUTOINSTALL", "False")

from coco_utils import COCO80_TO_91, letterbox

if TYPE_CHECKING:
    from pycocotools.coco import COCO  # pragma: no cover

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv5 H1 TPU inference on COCO and save predictions")
    parser.add_argument("--program-path", type=Path, default=REPO_ROOT / "experiments/detection/yolov5su_h1.tpu")
    parser.add_argument("--img-dir", type=Path, default=REPO_ROOT / "data/evaluation/MSCOCO2017/val2017")
    parser.add_argument("--ann-file", type=Path, default=REPO_ROOT / "data/evaluation/MSCOCO2017/annotations/instances_val2017.json")
    parser.add_argument("--predictions-out", type=Path, default=REPO_ROOT / "experiments/detection/predictions_h1_tpu.json")
    parser.add_argument("--timing-out", type=Path, default=REPO_ROOT / "experiments/detection/inference_timing_h1_tpu.json")
    parser.add_argument("--scales-json", type=Path, default=REPO_ROOT / "experiments/detection/scales.json")
    parser.add_argument("--input-tensor-name", type=str, default=None, help="Override input tensor name")
    parser.add_argument("--output-tensor-name", type=str, default=None, help="Override output tensor name")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.65)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup-images", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None, help="TPU device path like /dev/tpu0 (default: first available)")
    parser.add_argument("--debug-probe", action="store_true", help="Print detailed probe output tensor stats")
    return parser.parse_args()


def key_candidates(name: Optional[str]) -> List[str]:
    if not name:
        return []
    out = [name]
    if name.endswith(":0"):
        out.append(name[:-2])
    else:
        out.append(f"{name}:0")
    return out


def load_io_names(scales_json: Path, input_override: Optional[str], output_override: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if input_override or output_override:
        return input_override, output_override
    if not scales_json.exists():
        return None, None

    try:
        data = json.loads(scales_json.read_text(encoding="utf-8"))
    except Exception:
        return None, None

    input_name = None
    output_name = None
    for name, info in data.items():
        dtype = str(info.get("data_type", "")).lower()
        if input_name is None and dtype in ("int8", "uint8", "float32"):
            input_name = name
        if output_name is None and dtype in ("float32", "float16"):
            output_name = name
    return input_name, output_name


def preprocess(path: Path, img_size: int) -> tuple[np.ndarray, tuple[int, int], float, int, int]:
    img = cv2.imread(path.as_posix())
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img_h, img_w = img.shape[:2]
    img_lb, scale, pad_x, pad_y = letterbox(img, img_size)
    x = np.expand_dims(img_lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0, axis=0)
    return x, (img_h, img_w), scale, pad_x, pad_y


def make_batch(tensors: List[np.ndarray], batch_size: int) -> np.ndarray:
    if not tensors:
        raise RuntimeError("Empty batch")
    x = np.concatenate(tensors, axis=0)
    if x.shape[0] == batch_size:
        return x
    if x.shape[0] > batch_size:
        return x[:batch_size]
    pad_count = batch_size - x.shape[0]
    pad = np.repeat(x[-1:], repeats=pad_count, axis=0)
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
    probes = [inference, tpu_program]
    accessors = [
        "input_names",
        "inputs",
        "get_input_names",
        "get_inputs",
        "tensor_descriptions",
        "get_tensor_descriptions",
    ]
    for obj in probes:
        for name in accessors:
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
    for x in hints:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def resolve_runtime_input_name(
    inference: object,
    preferred_input: Optional[str],
    probe_x: np.ndarray,
    runtime_hints: List[str],
) -> tuple[str, Dict[str, np.ndarray]]:
    candidates = key_candidates(preferred_input)
    candidates += runtime_hints
    tried = set()
    errors: Dict[str, str] = {}
    for name in candidates:
        if name in tried:
            continue
        tried.add(name)
        try:
            out = run_once(inference, name, probe_x)
            return name, out
        except Exception as e:
            errors[name] = str(e)
            continue

    error_preview = {k: errors[k] for k in sorted(errors)[:4]}
    raise RuntimeError(
        "Could not resolve input tensor name. "
        f"Tried: {sorted(tried)}. "
        f"Sample errors: {error_preview}. "
        "Pass --input-tensor-name explicitly."
    )


def pick_output(output_dict: Dict[str, np.ndarray], preferred_name: Optional[str]) -> np.ndarray:
    for k in key_candidates(preferred_name):
        if k in output_dict:
            return output_dict[k]
    if len(output_dict) == 1:
        return next(iter(output_dict.values()))
    for v in output_dict.values():
        if isinstance(v, np.ndarray) and v.ndim >= 3:
            return v
    raise RuntimeError(f"Cannot resolve output tensor. Keys: {list(output_dict.keys())}")


def print_probe_stats(output_dict: Dict[str, np.ndarray]) -> None:
    print("Probe output stats:")
    for k, v in output_dict.items():
        arr = np.asarray(v)
        if arr.size == 0:
            print(f"  - {k}: shape={arr.shape}, dtype={arr.dtype}, empty")
            continue
        arr_f = arr.astype(np.float32, copy=False)
        print(
            f"  - {k}: shape={arr.shape}, dtype={arr.dtype}, "
            f"min={float(np.min(arr_f)):.6f}, max={float(np.max(arr_f)):.6f}, mean={float(np.mean(arr_f)):.6f}"
        )


def normalize_prediction_shape(pred: np.ndarray) -> np.ndarray:
    if pred.ndim != 3:
        raise ValueError(f"Unexpected prediction rank: {pred.ndim}, shape={pred.shape}")
    if pred.shape[1] in (84, 85):
        return pred
    if pred.shape[2] in (84, 85):
        return pred.transpose(0, 2, 1)
    raise ValueError(f"Unexpected prediction shape: {pred.shape}")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise RuntimeError("--batch-size must be > 0")
    if not args.program_path.exists():
        raise FileNotFoundError(f"TPU program not found: {args.program_path}")

    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.timing_out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from pycocotools.coco import COCO
        from ultralytics.utils.nms import non_max_suppression
    except Exception as e:
        raise RuntimeError("Missing dependencies: torch, pycocotools, ultralytics") from e

    coco = COCO(args.ann_file.as_posix())
    img_ids = coco.getImgIds()
    if args.limit > 0:
        img_ids = img_ids[: args.limit]

    preferred_input, preferred_output = load_io_names(args.scales_json, args.input_tensor_name, args.output_tensor_name)

    try:
        import pytpu as tpu  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: pytpu") from e

    devices = tpu.Device.list_devices()
    if not devices:
        raise RuntimeError("TPU device not found (Device.list_devices() is empty)")
    device_id = args.device or devices[0]
    if device_id not in devices:
        raise RuntimeError(f"Requested device {device_id} not available. available={devices}")

    print(f"Using TPU device: {device_id}")
    print(f"Program: {args.program_path}")

    results: List[Dict[str, Any]] = []
    infer_time = 0.0
    measured_images = 0

    with tpu.Device.open(device_id) as tpu_device:
        with tpu_device.load(args.program_path.as_posix()) as tpu_program:
            with tpu_program.inference() as inference:
                probe_path = args.img_dir / coco.loadImgs(img_ids[0])[0]["file_name"]
                probe_x_single, _, _, _, _ = preprocess(probe_path, args.img_size)
                probe_x = make_batch([probe_x_single], args.batch_size)
                runtime_hints = collect_runtime_input_hints(inference, tpu_program)
                print(f"Runtime input hints: {runtime_hints}")

                runtime_input_name, probe_out = resolve_runtime_input_name(inference, preferred_input, probe_x, runtime_hints)
                _ = pick_output(probe_out, preferred_output)
                print(f"Resolved runtime input tensor: {runtime_input_name}")
                if preferred_output:
                    print(f"Preferred output tensor: {preferred_output}")
                print(f"Probe output keys: {list(probe_out.keys())}")
                if args.debug_probe:
                    print_probe_stats(probe_out)

                num_batches = (len(img_ids) + args.batch_size - 1) // args.batch_size
                processed_images = 0

                for batch_idx in tqdm(range(num_batches), desc="TPU Inference"):
                    start = batch_idx * args.batch_size
                    end = min((batch_idx + 1) * args.batch_size, len(img_ids))
                    batch_ids = img_ids[start:end]

                    batch_tensors: List[np.ndarray] = []
                    metas: List[tuple[int, int, int, float, int, int]] = []
                    # (img_id, img_h, img_w, scale, pad_x, pad_y)
                    for img_id in batch_ids:
                        info = coco.loadImgs(img_id)[0]
                        x, (img_h, img_w), scale, pad_x, pad_y = preprocess(args.img_dir / info["file_name"], args.img_size)
                        batch_tensors.append(x)
                        metas.append((int(img_id), img_h, img_w, scale, pad_x, pad_y))

                    x_batch = make_batch(batch_tensors, args.batch_size)

                    t0 = time.perf_counter()
                    out_dict = run_once(inference, runtime_input_name, x_batch)
                    t1 = time.perf_counter()

                    pred = normalize_prediction_shape(np.asarray(pick_output(out_dict, preferred_output)))
                    if pred.shape[0] < len(batch_ids):
                        raise RuntimeError(
                            f"Output batch smaller than input batch: output={pred.shape}, valid={len(batch_ids)}"
                        )
                    valid_pred = pred[: len(batch_ids)]

                    dets = non_max_suppression(
                        torch.from_numpy(valid_pred),
                        conf_thres=args.conf_thres,
                        iou_thres=args.iou_thres,
                        max_det=args.max_det,
                    )
                    for det_idx, det in enumerate(dets):
                        if det is None or len(det) == 0:
                            continue
                        img_id, img_h, img_w, scale, pad_x, pad_y = metas[det_idx]
                        for x1, y1, x2, y2, conf, cls in det.cpu().numpy():
                            cls_idx = int(cls)
                            if cls_idx < 0 or cls_idx >= len(COCO80_TO_91):
                                continue
                            x1 = (x1 - pad_x) / scale
                            y1 = (y1 - pad_y) / scale
                            x2 = (x2 - pad_x) / scale
                            y2 = (y2 - pad_y) / scale
                            x1 = max(0.0, min(float(x1), img_w - 1.0))
                            y1 = max(0.0, min(float(y1), img_h - 1.0))
                            x2 = max(0.0, min(float(x2), img_w - 1.0))
                            y2 = max(0.0, min(float(y2), img_h - 1.0))
                            w, h = x2 - x1, y2 - y1
                            if w <= 1.0 or h <= 1.0:
                                continue
                            results.append(
                                {
                                    "image_id": img_id,
                                    "category_id": COCO80_TO_91[cls_idx],
                                    "bbox": [x1, y1, w, h],
                                    "score": float(conf),
                                }
                            )

                    processed_images += len(batch_ids)
                    if processed_images > args.warmup_images:
                        measured_in_batch = len(batch_ids)
                        if processed_images - len(batch_ids) < args.warmup_images:
                            measured_in_batch = processed_images - args.warmup_images
                        infer_time += (t1 - t0)
                        measured_images += measured_in_batch

    with args.predictions_out.open("w", encoding="utf-8") as f:
        json.dump(results, f)

    timing = {
        "backend": "TPU",
        "device": str(device_id),
        "program_file": args.program_path.as_posix(),
        "batch_size": args.batch_size,
        "images": len(img_ids),
        "warmup_images": args.warmup_images,
        "detections": len(results),
        "measured_inference_sec": infer_time,
        "throughput_img_per_sec": measured_images / max(infer_time, 1e-9),
        "predictions_file": args.predictions_out.as_posix(),
        "resolved_input_tensor": runtime_input_name,
        "preferred_output_tensor": preferred_output,
    }
    with args.timing_out.open("w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
