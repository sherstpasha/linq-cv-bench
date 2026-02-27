import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Sample:
    image_id: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FCN-ResNet50 TPU inference (.tpu program) and save predicted masks")
    parser.add_argument("--program-path", type=Path, default=REPO_ROOT / "experiments/segmentation/fcn_resnet50_h1.tpu")
    parser.add_argument("--voc-root", type=Path, default=REPO_ROOT / "data/evaluation/VOCdevkit/VOC2012")
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--predictions-dir", type=Path, default=REPO_ROOT / "experiments/segmentation/predictions_h1_tpu")
    parser.add_argument("--timing-out", type=Path, default=REPO_ROOT / "experiments/segmentation/inference_timing_h1_tpu.json")
    parser.add_argument("--scales-json", type=Path, default=REPO_ROOT / "experiments/segmentation/scales.json")
    parser.add_argument("--input-tensor-name", type=str, default=None, help="Override input tensor name")
    parser.add_argument("--output-tensor-name", type=str, default=None, help="Override output tensor name")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup-images", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--height", type=int, default=520)
    parser.add_argument("--width", type=int, default=520)
    parser.add_argument("--num-classes", type=int, default=21)
    parser.add_argument("--device", type=str, default=None, help="TPU device path, e.g. /dev/tpu3")
    return parser.parse_args()


def load_ids(split_file: Path, limit: int) -> List[str]:
    ids = [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    return ids[:limit] if limit > 0 else ids


def build_samples(voc_root: Path, split_file: Path, limit: int) -> List[Sample]:
    jpeg_dir = voc_root / "JPEGImages"
    image_ids = load_ids(split_file, limit)
    samples: List[Sample] = []
    for image_id in image_ids:
        path = jpeg_dir / f"{image_id}.jpg"
        if path.exists():
            samples.append(Sample(image_id=image_id, path=path))
    if not samples:
        raise RuntimeError(f"No images found for split: {split_file}")
    return samples


def preprocess(image: Image.Image, width: int, height: int) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((width, height), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr.astype(np.float32), axis=0)


def load_io_names(scales_json: Path, input_override: Optional[str], output_override: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
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
        if input_name is None and dtype in ("int8", "uint8"):
            input_name = name
        if output_name is None and dtype in ("float32", "float16"):
            output_name = name
    return input_name, output_name


def key_candidates(name: Optional[str]) -> List[str]:
    if not name:
        return []
    out = [name]
    if name.endswith(":0"):
        out.append(name[:-2])
    else:
        out.append(f"{name}:0")
    return out


def pick_output(output_dict: Dict[str, np.ndarray], preferred_name: Optional[str]) -> np.ndarray:
    for k in key_candidates(preferred_name):
        if k in output_dict:
            return output_dict[k]
    if len(output_dict) == 1:
        return next(iter(output_dict.values()))
    for _, v in output_dict.items():
        if isinstance(v, np.ndarray) and v.ndim >= 3:
            return v
    raise RuntimeError(f"Cannot resolve output tensor. Keys: {list(output_dict.keys())}")


def run_once(inference: object, input_name: str, x: np.ndarray) -> Dict[str, np.ndarray]:
    if hasattr(inference, "run"):
        return inference.run({input_name: x})  # type: ignore[attr-defined]
    return inference.sync({input_name: x})  # type: ignore[attr-defined]


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


def _as_name_list(obj: Any) -> List[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [str(k) for k in obj.keys()]
    if isinstance(obj, (list, tuple, set)):
        return [str(x) for x in obj]
    return []


def _collect_runtime_input_hints(inference: object, tpu_program: object) -> List[str]:
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
    hints.extend(["input", "input:0", "images", "images:0", "Placeholder", "Placeholder:0"])

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
) -> Tuple[str, Dict[str, np.ndarray]]:
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
        "Pass --input-tensor-name explicitly if your program uses a custom input name."
    )


def to_logits_shape(logits: np.ndarray, expected_batch: int) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.ndim == 3:
        logits = np.expand_dims(logits, axis=0)
    if logits.ndim != 4:
        raise RuntimeError(f"Unexpected segmentation output shape: {logits.shape}")

    if logits.shape[0] < expected_batch:
        raise RuntimeError(f"Output batch smaller than expected: output={logits.shape}, expected={expected_batch}")
    return logits


def logits_to_mask(sample_logits: np.ndarray, num_classes: int) -> np.ndarray:
    if sample_logits.ndim != 3:
        raise RuntimeError(f"Unexpected per-sample logits shape: {sample_logits.shape}")

    # Prefer exact class-channel match for robust NCHW/NHWC handling.
    if sample_logits.shape[0] == num_classes:
        class_axis = 0
    elif sample_logits.shape[-1] == num_classes:
        class_axis = 2
    else:
        # Fallback: classes are usually the smallest dimension.
        class_axis = int(np.argmin(sample_logits.shape))
    return np.argmax(sample_logits, axis=class_axis).astype(np.uint8)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise RuntimeError("--batch-size must be > 0")
    if not args.program_path.exists():
        raise FileNotFoundError(f"TPU program not found: {args.program_path}")

    split_file = args.split_file or (args.voc_root / "ImageSets/Segmentation/val.txt")
    samples = build_samples(args.voc_root, split_file, args.limit)

    args.predictions_dir.mkdir(parents=True, exist_ok=True)
    args.timing_out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import pytpu as tpu  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: pytpu") from e

    preferred_input, preferred_output = load_io_names(args.scales_json, args.input_tensor_name, args.output_tensor_name)

    devices = tpu.Device.list_devices()
    if not devices:
        raise RuntimeError("TPU device not found (Device.list_devices() is empty)")
    device_id = args.device or devices[0]
    print(f"Using TPU device: {device_id}")
    print(f"Program: {args.program_path}")

    infer_time = 0.0
    measured_images = 0
    total_images = 0

    with tpu.Device.open(device_id) as tpu_device:
        with tpu_device.load(args.program_path.as_posix()) as tpu_program:
            with tpu_program.inference() as inference:
                with Image.open(samples[0].path) as image:
                    probe_x_single = preprocess(image, width=args.width, height=args.height)
                probe_x = make_batch([probe_x_single], args.batch_size)

                runtime_hints = _collect_runtime_input_hints(inference, tpu_program)
                print(f"Runtime input hints: {runtime_hints}")
                runtime_input_name, probe_out = resolve_runtime_input_name(inference, preferred_input, probe_x, runtime_hints)
                _ = pick_output(probe_out, preferred_output)
                print(f"Resolved runtime input tensor: {runtime_input_name}")
                if preferred_output:
                    print(f"Preferred output tensor: {preferred_output}")
                print(f"Probe output keys: {list(probe_out.keys())}")

                num_batches = (len(samples) + args.batch_size - 1) // args.batch_size
                processed_images = 0
                for batch_idx in tqdm(range(num_batches), desc="TPU Inference"):
                    start = batch_idx * args.batch_size
                    end = min((batch_idx + 1) * args.batch_size, len(samples))
                    batch_samples = samples[start:end]

                    tensors: List[np.ndarray] = []
                    original_sizes: List[Tuple[int, int]] = []
                    for sample in batch_samples:
                        with Image.open(sample.path) as image:
                            original_sizes.append((image.height, image.width))
                            tensors.append(preprocess(image, width=args.width, height=args.height))
                    x = make_batch(tensors, args.batch_size)

                    t0 = time.perf_counter()
                    out_dict = run_once(inference, runtime_input_name, x)
                    t1 = time.perf_counter()

                    logits = pick_output(out_dict, preferred_output)
                    logits = to_logits_shape(logits, expected_batch=len(batch_samples))
                    valid_logits = logits[: len(batch_samples)]

                    for sample, sample_logits, orig_hw in zip(batch_samples, valid_logits, original_sizes):
                        pred = logits_to_mask(sample_logits, num_classes=args.num_classes)
                        pred_img = Image.fromarray(pred)
                        if pred.shape != orig_hw:
                            pred_img = pred_img.resize((orig_hw[1], orig_hw[0]), resample=Image.NEAREST)
                        pred_img.save(args.predictions_dir / f"{sample.image_id}.png")
                        total_images += 1

                    processed_images += len(batch_samples)
                    if processed_images > args.warmup_images:
                        measured_in_batch = len(batch_samples)
                        if processed_images - len(batch_samples) < args.warmup_images:
                            measured_in_batch = processed_images - args.warmup_images
                        infer_time += (t1 - t0)
                        measured_images += measured_in_batch

    timing = {
        "backend": "TPU",
        "device": str(device_id),
        "program_file": args.program_path.as_posix(),
        "batch_size": args.batch_size,
        "images": total_images,
        "warmup_images": args.warmup_images,
        "measured_inference_sec": infer_time,
        "throughput_img_per_sec": measured_images / max(infer_time, 1e-9),
        "predictions_dir": args.predictions_dir.as_posix(),
        "split_file": split_file.as_posix(),
        "resolved_input_tensor": runtime_input_name,
        "preferred_output_tensor": preferred_output,
    }
    with args.timing_out.open("w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
