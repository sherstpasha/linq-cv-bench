import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BATCH_RE = re.compile(r"_b(\d+)\.tpu$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run direct TPU accuracy for ResNet-50 on the evaluation split"
    )
    parser.add_argument(
        "--program-path",
        type=Path,
        default=REPO_ROOT / "artifacts/classification/resnet50_b1.tpu",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "data/evaluation/imagenet",
    )
    parser.add_argument("--val-map", type=Path, default=None)
    parser.add_argument(
        "--build-summary",
        type=Path,
        default=REPO_ROOT / "artifacts/classification/build_summary.json",
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=REPO_ROOT / "experiments/classification/accuracy/predictions.jsonl",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=REPO_ROOT / "experiments/classification/accuracy/summary.json",
    )
    parser.add_argument("--input-tensor-name", type=str, default=None)
    parser.add_argument("--output-tensor-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=0, help="0 means infer from .tpu filename/build summary")
    parser.add_argument("--samples", type=int, default=0, help="0 means all rows from val_map.txt")
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--device", type=str, default=None, help="Explicit TPU device, e.g. /dev/tpu0")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_val_rows(path: Path) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image_name, label = line.split()
            rows.append((image_name, int(label)))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


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


def resolve_runtime_metadata(build_summary: Dict[str, Any]) -> Dict[str, str]:
    export_metadata = build_summary.get("model_export_metadata") or {}
    input_layout = export_metadata.get("input_layout") or build_summary.get("input_layout") or "nchw"
    input_value_range = (
        export_metadata.get("input_value_range")
        or build_summary.get("runtime_input_value_range")
        or build_summary.get("input_value_range")
        or "normalized"
    )
    return {
        "input_layout": str(input_layout),
        "input_value_range": str(input_value_range),
    }


def preprocess_image(image_path: Path, runtime_metadata: Dict[str, str]) -> np.ndarray:
    input_layout = runtime_metadata["input_layout"]
    input_value_range = runtime_metadata["input_value_range"]

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


def make_batch(tensors: List[np.ndarray], batch_size: int) -> np.ndarray:
    if not tensors:
        raise RuntimeError("Empty batch")
    x = np.stack(tensors, axis=0).astype(np.float32)
    if x.shape[0] == batch_size:
        return x
    if x.shape[0] > batch_size:
        return x[:batch_size]
    pad_count = batch_size - x.shape[0]
    pad = np.repeat(x[-1:], repeats=pad_count, axis=0)
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

    hints.extend(["input.1", "input", "input:0", "images", "Placeholder", "Placeholder:0"])
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


def pick_output(output_dict: Dict[str, np.ndarray], preferred_name: Optional[str]) -> np.ndarray:
    for key in key_candidates(preferred_name):
        if key in output_dict:
            return np.asarray(output_dict[key])
    if len(output_dict) == 1:
        return np.asarray(next(iter(output_dict.values())))
    for _, value in output_dict.items():
        array = np.asarray(value)
        if array.ndim >= 2:
            return array
        if array.ndim == 1 and array.size >= 1000:
            return array
    raise RuntimeError(f"Cannot resolve output tensor. Keys: {list(output_dict.keys())}")


def top5_indices(logits: np.ndarray) -> np.ndarray:
    top5 = np.argpartition(logits, -5, axis=1)[:, -5:]
    top5_scores = np.take_along_axis(logits, top5, axis=1)
    order = np.argsort(-top5_scores, axis=1)
    return np.take_along_axis(top5, order, axis=1)


def main() -> None:
    args = parse_args()
    if not args.program_path.exists():
        raise FileNotFoundError(f"TPU program not found: {args.program_path}")
    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {args.dataset_dir}")
    if not args.build_summary.exists():
        raise FileNotFoundError(f"Build summary not found: {args.build_summary}")

    val_map = args.val_map or (args.dataset_dir / "val_map.txt")
    if not val_map.exists():
        raise FileNotFoundError(f"val_map.txt not found: {val_map}")

    try:
        import pytpu as tpu  # type: ignore
    except Exception as error:
        raise RuntimeError("Missing dependency: pytpu") from error

    build_summary = load_json(args.build_summary)
    runtime_metadata = resolve_runtime_metadata(build_summary)
    rows = load_val_rows(val_map)
    if args.samples > 0:
        rows = rows[: min(args.samples, len(rows))]

    program_batch_size = infer_program_batch_size(args.program_path, build_summary, args.batch_size)
    preferred_input_name = (
        args.input_tensor_name or build_summary.get("mapped_input_name") or build_summary.get("onnx_input_name")
    )
    preferred_output_name = (
        args.output_tensor_name or build_summary.get("selected_output_node") or build_summary.get("onnx_output_name")
    )

    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)

    devices = tpu.Device.list_devices()
    if not devices:
        raise RuntimeError("TPU device not found (Device.list_devices() is empty)")
    if args.device is not None:
        if args.device not in devices:
            raise RuntimeError(f"Requested device {args.device} is not available. Available devices: {devices}")
        device_id = args.device
    else:
        device_id = devices[0]

    infer_time = 0.0
    measured_images = 0
    measured_batches = 0
    good_top1 = 0
    good_top5 = 0
    resolved_input_name = None
    probe_output_keys: List[str] = []

    with tpu.Device.open(device_id) as tpu_device:
        with tpu_device.load(args.program_path.as_posix()) as tpu_program:
            with tpu_program.inference() as inference:
                first_tensor = preprocess_image(args.dataset_dir / rows[0][0], runtime_metadata)
                probe_x = make_batch([first_tensor], program_batch_size)
                runtime_hints = collect_runtime_input_hints(inference, tpu_program)
                resolved_input_name, probe_out = resolve_runtime_input_name(
                    inference,
                    preferred_input_name=preferred_input_name,
                    probe_x=probe_x,
                    runtime_hints=runtime_hints,
                )
                probe_output_keys = list(probe_out.keys())
                _ = pick_output(probe_out, preferred_output_name)

                with args.predictions_out.open("w", encoding="utf-8") as out_file:
                    for batch_idx, start in enumerate(range(0, len(rows), program_batch_size)):
                        batch_rows = rows[start : start + program_batch_size]
                        tensors = [
                            preprocess_image(args.dataset_dir / image_name, runtime_metadata)
                            for image_name, _ in batch_rows
                        ]
                        x = make_batch(tensors, program_batch_size)

                        t0 = time.perf_counter()
                        out_dict = run_once(inference, resolved_input_name, x)
                        t1 = time.perf_counter()

                        if batch_idx >= args.warmup_batches:
                            infer_time += t1 - t0
                            measured_images += len(batch_rows)
                            measured_batches += 1

                        logits = pick_output(out_dict, preferred_output_name)
                        logits = np.asarray(logits)
                        if logits.ndim == 1:
                            logits = np.expand_dims(logits, axis=0)
                        valid_logits = logits[: len(batch_rows)]
                        top5 = top5_indices(valid_logits)
                        top1 = top5[:, 0]

                        for (image_name, raw_label), pred1, pred5 in zip(batch_rows, top1, top5):
                            label = raw_label - 1
                            pred1 = int(pred1)
                            pred5_list = [int(value) for value in pred5.tolist()]
                            if pred1 == label:
                                good_top1 += 1
                            if label in pred5_list:
                                good_top5 += 1
                            out_file.write(json.dumps({"image": image_name, "top5": pred5_list}) + "\n")

    total = len(rows)
    summary = {
        "pipeline": "direct_tpu_accuracy",
        "program_path": args.program_path.as_posix(),
        "build_summary": args.build_summary.as_posix(),
        "runtime_metadata": runtime_metadata,
        "dataset_dir": args.dataset_dir.as_posix(),
        "val_map": val_map.as_posix(),
        "effective_samples": total,
        "device": str(device_id),
        "batch_size": program_batch_size,
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
        "preferred_input_name": preferred_input_name,
        "preferred_output_name": preferred_output_name,
        "resolved_input_name": resolved_input_name,
        "probe_output_keys": probe_output_keys,
    }
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved accuracy summary: {args.summary_out}")


if __name__ == "__main__":
    main()
