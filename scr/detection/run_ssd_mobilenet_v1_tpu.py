import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from pycocotools.coco import COCO


REPO_ROOT = Path(__file__).resolve().parents[2]
BATCH_RE = re.compile(r"_b(\d+)\.tpu$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run direct TPU inference for SSD-MobileNetV1 on a COCO fragment"
    )
    parser.add_argument(
        "--program-path",
        type=Path,
        default=REPO_ROOT / "artifacts/detection/ssd_mobilenet_v1/ssd_mobilenet_v1_b1.tpu",
    )
    parser.add_argument(
        "--build-summary",
        type=Path,
        default=REPO_ROOT / "artifacts/detection/ssd_mobilenet_v1/build_summary.json",
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
        default=REPO_ROOT / "experiments/detection/ssd_mobilenet_v1/predictions.json",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=REPO_ROOT / "experiments/detection/ssd_mobilenet_v1/summary.json",
    )
    parser.add_argument("--batch-size", type=int, default=0, help="0 means infer from .tpu filename/build summary")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_program_batch_size(program_path: Path, build_summary: Dict[str, Any], requested_batch_size: int) -> int:
    if requested_batch_size > 0:
        return requested_batch_size

    match = BATCH_RE.search(program_path.name)
    if match:
        return int(match.group(1))

    compiled_programs = build_summary.get("compiled_programs", {})
    for batch_size, path in compiled_programs.items():
        if Path(path) == program_path:
            return int(batch_size)

    return 1


def resolve_runtime_metadata(build_summary: Dict[str, Any]) -> Dict[str, Any]:
    image_size = int(build_summary.get("image_size", 300))
    input_layout = str(build_summary.get("input_layout", "nhwc"))
    input_dtype = str(build_summary.get("input_dtype", "uint8"))
    preferred_outputs = build_summary.get("mapped_output_names") or build_summary.get("onnx_output_names") or []
    preferred_input = build_summary.get("mapped_input_name") or build_summary.get("onnx_input_name")
    return {
        "image_size": image_size,
        "input_layout": input_layout,
        "input_dtype": input_dtype,
        "preferred_input": preferred_input,
        "preferred_outputs": preferred_outputs,
    }


def preprocess_image(image_path: Path, image_size: int, input_layout: str, input_dtype: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        orig_size = image.size
        image = image.resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.uint8)

    if input_dtype.startswith("float32"):
        arr = arr.astype(np.float32)
    elif not input_dtype.startswith("uint8"):
        raise RuntimeError(f"Unsupported runtime input dtype: {input_dtype}")

    if input_layout == "nchw":
        arr = np.transpose(arr, (2, 0, 1))
    elif input_layout != "nhwc":
        raise RuntimeError(f"Unsupported runtime input layout: {input_layout}")

    return arr, (int(orig_size[1]), int(orig_size[0]))


def make_batch(tensors: List[np.ndarray], batch_size: int) -> np.ndarray:
    if not tensors:
        raise RuntimeError("Empty batch")
    x = np.stack(tensors, axis=0)
    if x.shape[0] == batch_size:
        return x
    if x.shape[0] > batch_size:
        return x[:batch_size]
    pad = np.repeat(x[-1:], repeats=batch_size - x.shape[0], axis=0)
    return np.concatenate([x, pad], axis=0)


def key_candidates(name: Optional[str]) -> List[str]:
    if not name:
        return []
    out = [name]
    if name.endswith(":0"):
        out.append(name[:-2])
    else:
        out.append(f"{name}:0")
    return out


def _as_name_list(obj: Any) -> List[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [str(key) for key in obj.keys()]
    if isinstance(obj, (list, tuple, set)):
        return [str(item) for item in obj]
    return []


def collect_runtime_input_hints(inference: object, tpu_program: object) -> List[str]:
    hints: List[str] = []
    for obj in (inference, tpu_program):
        for accessor in (
            "input_names",
            "inputs",
            "get_input_names",
            "get_inputs",
            "tensor_descriptions",
            "get_tensor_descriptions",
        ):
            if not hasattr(obj, accessor):
                continue
            attr = getattr(obj, accessor)
            try:
                value = attr() if callable(attr) else attr
            except Exception:
                continue
            hints.extend(_as_name_list(value))
    hints.extend(["image_tensor", "image_tensor:0", "input", "input:0", "Placeholder", "Placeholder:0"])

    unique_hints: List[str] = []
    seen = set()
    for item in hints:
        if item and item not in seen:
            seen.add(item)
            unique_hints.append(item)
    return unique_hints


def run_once(inference: object, input_name: str, x: np.ndarray) -> Dict[str, np.ndarray]:
    if hasattr(inference, "run"):
        return inference.run({input_name: x})  # type: ignore[attr-defined]
    return inference.sync({input_name: x})  # type: ignore[attr-defined]


def resolve_runtime_input_name(
    inference: object,
    preferred_input_name: Optional[str],
    probe_x: np.ndarray,
    runtime_hints: List[str],
) -> Tuple[str, Dict[str, np.ndarray]]:
    candidates = key_candidates(preferred_input_name) + runtime_hints
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
    raise RuntimeError(
        "Could not resolve input tensor name. "
        f"Tried: {sorted(tried)}. "
        f"Sample errors: {dict(list(errors.items())[:4])}"
    )


def pick_named_output(
    output_dict: Dict[str, np.ndarray],
    preferred_name: Optional[str],
    keywords: List[str],
    rank_hint: Optional[int] = None,
    last_dim_hint: Optional[int] = None,
) -> Tuple[np.ndarray, str]:
    for key in key_candidates(preferred_name):
        if key in output_dict:
            return np.asarray(output_dict[key]), key

    for key, value in output_dict.items():
        key_lower = key.lower()
        if all(keyword in key_lower for keyword in keywords):
            return np.asarray(value), key

    for key, value in output_dict.items():
        arr = np.asarray(value)
        if rank_hint is not None and arr.ndim != rank_hint:
            continue
        if last_dim_hint is not None and (arr.ndim == 0 or arr.shape[-1] != last_dim_hint):
            continue
        return arr, key

    raise RuntimeError(f"Could not resolve output. Available keys: {list(output_dict.keys())}")


def normalize_boxes(boxes: np.ndarray) -> np.ndarray:
    arr = np.asarray(boxes)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)
    if arr.ndim != 3 or arr.shape[-1] != 4:
        raise RuntimeError(f"Unexpected boxes shape: {arr.shape}")
    return arr


def normalize_matrix(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=0)
    if arr.ndim != 2:
        raise RuntimeError(f"Unexpected matrix shape: {arr.shape}")
    return arr


def normalize_counts(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    return arr


def main() -> None:
    args = parse_args()
    if not args.program_path.exists():
        raise FileNotFoundError(f"TPU program not found: {args.program_path}")
    if not args.build_summary.exists():
        raise FileNotFoundError(f"Build summary not found: {args.build_summary}")
    if args.limit < 0:
        raise RuntimeError("--limit must be >= 0")
    if args.score_threshold < 0.0:
        raise RuntimeError("--score-threshold must be >= 0")

    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)

    build_summary = load_json(args.build_summary)
    runtime_metadata = resolve_runtime_metadata(build_summary)
    batch_size = infer_program_batch_size(args.program_path, build_summary, args.batch_size)
    coco = COCO(args.ann_file.as_posix())
    img_ids = coco.getImgIds()
    if args.limit > 0:
        img_ids = img_ids[: args.limit]

    import pytpu as tpu  # type: ignore

    devices = tpu.Device.list_devices()
    if not devices:
        raise RuntimeError("TPU device not found")
    device_id = args.device or devices[0]

    preferred_outputs = runtime_metadata["preferred_outputs"]
    preferred_output_map = {
        "boxes": next((name for name in preferred_outputs if "boxes" in name.lower()), None),
        "classes": next((name for name in preferred_outputs if "classes" in name.lower()), None),
        "scores": next((name for name in preferred_outputs if "scores" in name.lower()), None),
        "num_detections": next((name for name in preferred_outputs if "num" in name.lower()), None),
    }

    results: List[Dict[str, Any]] = []
    infer_time = 0.0
    measured_images = 0
    resolved_input_name = None
    resolved_output_keys: Dict[str, str] = {}

    with tpu.Device.open(device_id) as tpu_device:
        with tpu_device.load(args.program_path.as_posix()) as tpu_program:
            with tpu_program.inference() as inference:
                probe_info = coco.loadImgs(img_ids[0])[0]
                probe_tensor, _ = preprocess_image(
                    args.img_dir / probe_info["file_name"],
                    image_size=runtime_metadata["image_size"],
                    input_layout=runtime_metadata["input_layout"],
                    input_dtype=runtime_metadata["input_dtype"],
                )
                probe_batch = make_batch([probe_tensor], batch_size)
                runtime_hints = collect_runtime_input_hints(inference, tpu_program)
                resolved_input_name, probe_out = resolve_runtime_input_name(
                    inference=inference,
                    preferred_input_name=runtime_metadata["preferred_input"],
                    probe_x=probe_batch,
                    runtime_hints=runtime_hints,
                )

                _, resolved_output_keys["boxes"] = pick_named_output(
                    probe_out,
                    preferred_output_map["boxes"],
                    keywords=["boxes"],
                    rank_hint=3,
                    last_dim_hint=4,
                )
                _, resolved_output_keys["classes"] = pick_named_output(
                    probe_out,
                    preferred_output_map["classes"],
                    keywords=["classes"],
                    rank_hint=2,
                )
                _, resolved_output_keys["scores"] = pick_named_output(
                    probe_out,
                    preferred_output_map["scores"],
                    keywords=["scores"],
                    rank_hint=2,
                )
                _, resolved_output_keys["num_detections"] = pick_named_output(
                    probe_out,
                    preferred_output_map["num_detections"],
                    keywords=["num", "detection"],
                )

                num_batches = (len(img_ids) + batch_size - 1) // batch_size
                processed_images = 0
                for batch_index in range(num_batches):
                    start = batch_index * batch_size
                    end = min((batch_index + 1) * batch_size, len(img_ids))
                    batch_ids = img_ids[start:end]

                    tensors: List[np.ndarray] = []
                    metas: List[Tuple[int, int, int]] = []
                    for image_id in batch_ids:
                        info = coco.loadImgs(image_id)[0]
                        tensor, (orig_h, orig_w) = preprocess_image(
                            args.img_dir / info["file_name"],
                            image_size=runtime_metadata["image_size"],
                            input_layout=runtime_metadata["input_layout"],
                            input_dtype=runtime_metadata["input_dtype"],
                        )
                        tensors.append(tensor)
                        metas.append((int(image_id), orig_h, orig_w))

                    batch = make_batch(tensors, batch_size)
                    t0 = time.perf_counter()
                    output_dict = run_once(inference, resolved_input_name, batch)
                    t1 = time.perf_counter()

                    boxes = normalize_boxes(np.asarray(output_dict[resolved_output_keys["boxes"]]))[: len(metas)]
                    classes = normalize_matrix(np.asarray(output_dict[resolved_output_keys["classes"]]))[: len(metas)]
                    scores = normalize_matrix(np.asarray(output_dict[resolved_output_keys["scores"]]))[: len(metas)]
                    counts = normalize_counts(np.asarray(output_dict[resolved_output_keys["num_detections"]]))[: len(metas)]

                    for row_index, (image_id, orig_h, orig_w) in enumerate(metas):
                        det_count = min(
                            int(round(float(counts[row_index]))),
                            boxes.shape[1],
                            classes.shape[1],
                            scores.shape[1],
                        )
                        for det_index in range(det_count):
                            score = float(scores[row_index, det_index])
                            if score < args.score_threshold:
                                continue
                            category_id = int(round(float(classes[row_index, det_index])))
                            if category_id <= 0:
                                continue
                            ymin, xmin, ymax, xmax = [float(v) for v in boxes[row_index, det_index]]
                            x1 = max(0.0, min(xmin * orig_w, orig_w - 1.0))
                            y1 = max(0.0, min(ymin * orig_h, orig_h - 1.0))
                            x2 = max(0.0, min(xmax * orig_w, orig_w - 1.0))
                            y2 = max(0.0, min(ymax * orig_h, orig_h - 1.0))
                            width = x2 - x1
                            height = y2 - y1
                            if width <= 1.0 or height <= 1.0:
                                continue
                            results.append(
                                {
                                    "image_id": image_id,
                                    "category_id": category_id,
                                    "bbox": [x1, y1, width, height],
                                    "score": score,
                                }
                            )

                    processed_images += len(metas)
                    if processed_images > args.warmup_batches * batch_size:
                        measured_in_batch = len(metas)
                        if processed_images - len(metas) < args.warmup_batches * batch_size:
                            measured_in_batch = processed_images - args.warmup_batches * batch_size
                        infer_time += (t1 - t0)
                        measured_images += measured_in_batch

    args.predictions_out.write_text(json.dumps(results), encoding="utf-8")
    summary = {
        "pipeline": "direct_tpu_detection",
        "program_path": args.program_path.as_posix(),
        "build_summary": args.build_summary.as_posix(),
        "runtime_metadata": runtime_metadata,
        "img_dir": args.img_dir.as_posix(),
        "ann_file": args.ann_file.as_posix(),
        "effective_images": len(img_ids),
        "device": str(device_id),
        "batch_size": batch_size,
        "requested_batch_size": args.batch_size,
        "warmup_batches": args.warmup_batches,
        "measured_inference_sec": infer_time,
        "throughput_img_per_sec": measured_images / max(infer_time, 1e-9),
        "predictions_file": args.predictions_out.as_posix(),
        "detections": len(results),
        "score_threshold": args.score_threshold,
        "resolved_input_name": resolved_input_name,
        "resolved_output_keys": resolved_output_keys,
    }
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
