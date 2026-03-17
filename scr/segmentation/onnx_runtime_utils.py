import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional_export_metadata(model_path: Path) -> Dict[str, Any]:
    metadata_path = model_path.with_suffix(".json")
    if not metadata_path.exists():
        return {}
    return load_json(metadata_path)


def ort_input_dtype_to_numpy(type_name: str) -> np.dtype[np.generic]:
    type_name = type_name.lower()
    if "float16" in type_name:
        return np.dtype(np.float16)
    if "float" in type_name:
        return np.dtype(np.float32)
    if "uint8" in type_name:
        return np.dtype(np.uint8)
    if "int64" in type_name:
        return np.dtype(np.int64)
    if "int32" in type_name:
        return np.dtype(np.int32)
    raise RuntimeError(f"Unsupported ONNX input type: {type_name}")


def resolve_provider(provider: str) -> str:
    available = ort.get_available_providers()
    provider = provider.lower()
    if provider == "auto":
        if "CUDAExecutionProvider" in available:
            return "CUDAExecutionProvider"
        return "CPUExecutionProvider"
    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                f"CUDAExecutionProvider is not available. available_providers={available}"
            )
        return "CUDAExecutionProvider"
    if provider == "cpu":
        if "CPUExecutionProvider" not in available:
            raise RuntimeError(
                f"CPUExecutionProvider is not available. available_providers={available}"
            )
        return "CPUExecutionProvider"
    raise RuntimeError(f"Unsupported provider: {provider}")


def preload_cuda_runtime_if_needed(resolved_provider: str) -> None:
    if resolved_provider != "CUDAExecutionProvider":
        return
    try:
        ort.preload_dlls(directory="")
        return
    except Exception:
        pass
    try:
        import torch  # type: ignore

        _ = torch.cuda.is_available()
    except Exception:
        return


def create_session(model_path: Path, provider: str) -> Tuple[ort.InferenceSession, str]:
    resolved_provider = resolve_provider(provider)
    preload_cuda_runtime_if_needed(resolved_provider)
    providers = [resolved_provider]
    if resolved_provider != "CPUExecutionProvider":
        providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(model_path.as_posix(), providers=providers)
    return session, resolved_provider


def merge_input_shape(session_shape: Sequence[Any], export_shape: Sequence[Any]) -> List[Any]:
    out: List[Any] = []
    length = max(len(session_shape), len(export_shape))
    for idx in range(length):
        session_dim = session_shape[idx] if idx < len(session_shape) else None
        export_dim = export_shape[idx] if idx < len(export_shape) else None
        if isinstance(session_dim, int) and session_dim > 0:
            out.append(session_dim)
        elif isinstance(export_dim, int) and export_dim > 0:
            out.append(export_dim)
        else:
            out.append(session_dim if session_dim is not None else export_dim)
    return out


def infer_runtime_contract(session: ort.InferenceSession, export_metadata: Dict[str, Any]) -> Dict[str, Any]:
    inputs = session.get_inputs()
    if not inputs:
        raise RuntimeError("ONNX model has no inputs")
    input_info = inputs[0]
    input_dtype = ort_input_dtype_to_numpy(input_info.type)
    input_shape = merge_input_shape(list(input_info.shape), list(export_metadata.get("input_shape") or []))
    input_layout = str(export_metadata.get("input_layout") or "nchw")
    input_value_range = str(export_metadata.get("input_value_range") or "normalized")
    output_name = export_metadata.get("output_name")
    return {
        "input_name": input_info.name,
        "input_dtype": input_dtype.name,
        "input_shape": input_shape,
        "input_layout": input_layout,
        "input_value_range": input_value_range,
        "output_name": output_name,
    }


def infer_static_batch_size(runtime_contract: Dict[str, Any], requested_batch_size: int) -> int:
    input_shape = runtime_contract.get("input_shape") or []
    if input_shape:
        first_dim = input_shape[0]
        if isinstance(first_dim, int) and first_dim > 0:
            return first_dim
    return requested_batch_size


def resolve_spatial_size(runtime_contract: Dict[str, Any], default_height: int, default_width: int) -> Tuple[int, int]:
    input_shape = runtime_contract.get("input_shape") or []
    input_layout = runtime_contract.get("input_layout") or "nchw"
    if input_layout == "nchw" and len(input_shape) >= 4:
        height = input_shape[2]
        width = input_shape[3]
        if isinstance(height, int) and height > 0 and isinstance(width, int) and width > 0:
            return height, width
    if input_layout == "nhwc" and len(input_shape) >= 4:
        height = input_shape[1]
        width = input_shape[2]
        if isinstance(height, int) and height > 0 and isinstance(width, int) and width > 0:
            return height, width
    return default_height, default_width


def load_ids(split_file: Path, limit: int) -> List[str]:
    ids = [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    return ids[:limit] if limit > 0 else ids


def build_samples(voc_root: Path, split_file: Path, limit: int) -> List[Tuple[str, Path, Path]]:
    image_ids = load_ids(split_file, limit)
    samples: List[Tuple[str, Path, Path]] = []
    for image_id in image_ids:
        image_path = voc_root / "JPEGImages" / f"{image_id}.jpg"
        gt_path = voc_root / "SegmentationClass" / f"{image_id}.png"
        if image_path.exists() and gt_path.exists():
            samples.append((image_id, image_path, gt_path))
    if not samples:
        raise RuntimeError(f"No VOC samples found for split: {split_file}")
    return samples


def preprocess_image(
    image_path: Path,
    runtime_contract: Dict[str, Any],
    height: int,
    width: int,
) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((width, height), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32)

    input_layout = runtime_contract["input_layout"]
    input_dtype = np.dtype(runtime_contract["input_dtype"])
    input_value_range = runtime_contract["input_value_range"]

    if input_value_range == "normalized":
        arr = arr / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    elif input_value_range == "unit_float":
        arr = arr / 255.0
    elif input_value_range == "uint8":
        pass
    else:
        raise RuntimeError(f"Unsupported input_value_range: {input_value_range}")

    if input_layout == "nchw":
        arr = np.transpose(arr, (2, 0, 1))
    elif input_layout != "nhwc":
        raise RuntimeError(f"Unsupported input_layout: {input_layout}")

    return arr.astype(input_dtype, copy=False)


def make_batch(tensors: Sequence[np.ndarray], batch_size: int) -> np.ndarray:
    if not tensors:
        raise RuntimeError("Empty batch")
    x = np.stack(tensors, axis=0)
    if x.shape[0] == batch_size:
        return x
    if x.shape[0] > batch_size:
        return x[:batch_size]
    pad = np.repeat(x[-1:], repeats=batch_size - x.shape[0], axis=0)
    return np.concatenate([x, pad], axis=0)


def pick_output(
    session: ort.InferenceSession,
    output_dict: Dict[str, np.ndarray],
    preferred_name: Optional[str],
) -> np.ndarray:
    candidates: List[str] = []
    if preferred_name:
        candidates.append(preferred_name)
    for output in session.get_outputs():
        candidates.append(output.name)

    seen = set()
    for key in candidates:
        if not key or key in seen:
            continue
        seen.add(key)
        if key in output_dict:
            return np.asarray(output_dict[key])

    if len(output_dict) == 1:
        return np.asarray(next(iter(output_dict.values())))

    for value in output_dict.values():
        array = np.asarray(value)
        if array.ndim >= 4:
            return array

    raise RuntimeError(f"Cannot resolve output tensor. Keys: {list(output_dict.keys())}")


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
    if sample_logits.shape[0] == num_classes:
        class_axis = 0
    elif sample_logits.shape[-1] == num_classes:
        class_axis = 2
    else:
        class_axis = int(np.argmin(sample_logits.shape))
    return np.argmax(sample_logits, axis=class_axis).astype(np.uint8)
