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
        # Prefer CUDA/cuDNN from pip-installed NVIDIA site packages to avoid
        # mixing them with arbitrary system-wide CUDA installs.
        ort.preload_dlls(directory="")
        return
    except Exception:
        pass
    try:
        import torch  # type: ignore

        _ = torch.cuda.is_available()
    except Exception:
        # Best-effort preload only. If CUDA libs are available system-wide,
        # ONNX Runtime can still initialize without PyTorch.
        return


def create_session(model_path: Path, provider: str) -> Tuple[ort.InferenceSession, str]:
    resolved_provider = resolve_provider(provider)
    preload_cuda_runtime_if_needed(resolved_provider)
    providers = [resolved_provider]
    if resolved_provider != "CPUExecutionProvider":
        providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(model_path.as_posix(), providers=providers)
    return session, resolved_provider


def infer_runtime_contract(
    session: ort.InferenceSession,
    export_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    inputs = session.get_inputs()
    if not inputs:
        raise RuntimeError("ONNX model has no inputs")
    input_info = inputs[0]
    input_dtype = ort_input_dtype_to_numpy(input_info.type)
    input_shape = list(input_info.shape)
    input_layout = str(export_metadata.get("input_layout") or "nhwc")
    input_value_range = str(export_metadata.get("input_value_range") or "uint8")
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


def preprocess_image(
    image_path: Path,
    runtime_contract: Dict[str, Any],
) -> np.ndarray:
    with Image.open(image_path) as image:
        image = resize_center_crop(image)
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


def pick_logits(
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

    raise RuntimeError(f"Cannot resolve output tensor. Keys: {list(output_dict.keys())}")


def top5_indices(logits: np.ndarray) -> np.ndarray:
    top5 = np.argpartition(logits, -5, axis=1)[:, -5:]
    top5_scores = np.take_along_axis(logits, top5, axis=1)
    order = np.argsort(-top5_scores, axis=1)
    return np.take_along_axis(top5, order, axis=1)
