import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pycocotools.coco import COCO


REPO_ROOT = Path(__file__).resolve().parents[2]
BATCH_RE = re.compile(r"_b(\d+)\.tpu$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RetinaNet H1 TPU inference on COCO and save predictions")
    parser.add_argument(
        "--program-path",
        type=Path,
        default=REPO_ROOT / "artifacts/detection/retinanet/retinanet_resnet50_fpn_b8.tpu",
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
        default=REPO_ROOT / "experiments/detection/retinanet/predictions_tpu.json",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=REPO_ROOT / "experiments/detection/retinanet/tpu_summary.json",
    )
    parser.add_argument(
        "--build-summary",
        type=Path,
        default=REPO_ROOT / "artifacts/detection/retinanet/build_summary.json",
    )
    parser.add_argument("--input-tensor-name", type=str, default=None)
    parser.add_argument("--output-tensor-name", type=str, default=None)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--warmup-images", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def preprocess(path: Path, width: int, height: int) -> tuple[np.ndarray, tuple[int, int]]:
    img = cv2.imread(path.as_posix())
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img_h, img_w = img.shape[:2]
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    x = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(x, axis=0), (img_h, img_w)


def key_candidates(name: Optional[str]) -> List[str]:
    if not name:
        return []
    out = [name]
    if name.endswith(":0"):
        out.append(name[:-2])
    else:
        out.append(f"{name}:0")
    return out


def load_io_names(build_summary: Path, input_override: Optional[str], output_override: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if input_override or output_override:
        return input_override, output_override
    if not build_summary.exists():
        return None, None
    try:
        data = json.loads(build_summary.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    return data.get("mapped_input_name"), data.get("mapped_output_name")


def infer_batch_size(program_path: Path, build_summary_path: Path, requested_batch_size: int) -> int:
    if requested_batch_size > 0:
        return requested_batch_size
    match = BATCH_RE.search(program_path.name)
    if match:
        return int(match.group(1))
    if build_summary_path.exists():
        try:
            data = json.loads(build_summary_path.read_text(encoding="utf-8"))
            for batch_size, path in data.get("compiled_programs", {}).items():
                if Path(path) == program_path:
                    return int(batch_size)
        except Exception:
            pass
    return 1


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
    hints.extend(["images", "images:0", "input", "input:0", "Placeholder", "Placeholder:0"])

    uniq: List[str] = []
    seen = set()
    for item in hints:
        if item and item not in seen:
            seen.add(item)
            uniq.append(item)
    return uniq


def resolve_runtime_input_name(inference: object, preferred_input: Optional[str], probe_x: np.ndarray, runtime_hints: List[str]) -> tuple[str, Dict[str, np.ndarray]]:
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
    raise RuntimeError(f"Could not resolve input tensor name. Tried: {sorted(tried)}. Sample errors: {dict(list(errors.items())[:4])}")


def pick_output(output_dict: Dict[str, np.ndarray], preferred_name: Optional[str]) -> np.ndarray:
    for key in key_candidates(preferred_name):
        if key in output_dict:
            return output_dict[key]
    if len(output_dict) == 1:
        return next(iter(output_dict.values()))
    for value in output_dict.values():
        if isinstance(value, np.ndarray) and value.ndim >= 2:
            return value
    raise RuntimeError(f"Cannot resolve output tensor. Keys: {list(output_dict.keys())}")


def normalize_detections_shape(dets: np.ndarray) -> np.ndarray:
    arr = np.asarray(dets)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)
    if arr.ndim != 3 or arr.shape[-1] != 6:
        raise RuntimeError(f"Unexpected detections shape: {arr.shape}")
    return arr


def main() -> None:
    args = parse_args()
    if not args.program_path.exists():
        raise FileNotFoundError(f"TPU program not found: {args.program_path}")

    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)

    coco = COCO(args.ann_file.as_posix())
    valid_coco_ids = set(coco.getCatIds())
    img_ids = coco.getImgIds()
    if args.limit > 0:
        img_ids = img_ids[: args.limit]

    preferred_input, preferred_output = load_io_names(args.build_summary, args.input_tensor_name, args.output_tensor_name)
    batch_size = infer_batch_size(args.program_path, args.build_summary, args.batch_size)

    import pytpu as tpu  # type: ignore

    devices = tpu.Device.list_devices()
    if not devices:
        raise RuntimeError("TPU device not found")
    device_id = args.device or devices[0]

    results: List[Dict[str, Any]] = []
    infer_time = 0.0
    measured_images = 0

    with tpu.Device.open(device_id) as tpu_device:
        with tpu_device.load(args.program_path.as_posix()) as tpu_program:
            with tpu_program.inference() as inference:
                probe_path = args.img_dir / coco.loadImgs(img_ids[0])[0]["file_name"]
                probe_x_single, _ = preprocess(probe_path, width=args.width, height=args.height)
                probe_x = make_batch([probe_x_single], batch_size)
                runtime_hints = collect_runtime_input_hints(inference, tpu_program)

                runtime_input_name, probe_out = resolve_runtime_input_name(inference, preferred_input, probe_x, runtime_hints)
                _ = pick_output(probe_out, preferred_output)

                num_batches = (len(img_ids) + batch_size - 1) // batch_size
                processed_images = 0

                for batch_idx in range(num_batches):
                    start = batch_idx * batch_size
                    end = min((batch_idx + 1) * batch_size, len(img_ids))
                    batch_ids = img_ids[start:end]

                    tensors: List[np.ndarray] = []
                    metas: List[Tuple[int, int, int]] = []
                    for img_id in batch_ids:
                        info = coco.loadImgs(img_id)[0]
                        x, (orig_h, orig_w) = preprocess(args.img_dir / info["file_name"], width=args.width, height=args.height)
                        tensors.append(x)
                        metas.append((int(img_id), orig_h, orig_w))

                    x_batch = make_batch(tensors, batch_size)

                    t0 = time.perf_counter()
                    out_dict = run_once(inference, runtime_input_name, x_batch)
                    t1 = time.perf_counter()

                    dets = normalize_detections_shape(np.asarray(pick_output(out_dict, preferred_output)))[: len(metas)]
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
                            width = x2 - x1
                            height = y2 - y1
                            if width <= 1.0 or height <= 1.0:
                                continue
                            results.append({"image_id": img_id, "category_id": cat_id, "bbox": [x1, y1, width, height], "score": score})

                    processed_images += len(metas)
                    if processed_images > args.warmup_images:
                        measured_in_batch = len(metas)
                        if processed_images - len(metas) < args.warmup_images:
                            measured_in_batch = processed_images - args.warmup_images
                        infer_time += (t1 - t0)
                        measured_images += measured_in_batch

    args.predictions_out.write_text(json.dumps(results), encoding="utf-8")
    summary = {
        "pipeline": "direct_tpu_detection",
        "program_path": args.program_path.as_posix(),
        "ann_file": args.ann_file.as_posix(),
        "img_dir": args.img_dir.as_posix(),
        "device": str(device_id),
        "batch_size": batch_size,
        "requested_batch_size": args.batch_size,
        "images": len(img_ids),
        "warmup_images": args.warmup_images,
        "detections": len(results),
        "measured_inference_sec": infer_time,
        "throughput_img_per_sec": measured_images / max(infer_time, 1e-9),
        "predictions_file": args.predictions_out.as_posix(),
        "resolved_input_tensor": runtime_input_name,
        "preferred_output_tensor": preferred_output,
        "input_size": [args.height, args.width],
    }
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
