import contextlib
import io
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from coco_utils import COCO80_TO_91, letterbox


METRIC_KEYS = [
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


def _infer_layout_from_shape(shape: Sequence[Any]) -> str:
    if len(shape) != 4:
        raise RuntimeError(f"Expected rank-4 image input, got shape={shape}")
    if isinstance(shape[1], int) and shape[1] in (1, 3):
        return "nchw"
    if isinstance(shape[-1], int) and shape[-1] in (1, 3):
        return "nhwc"
    return "nhwc"


def _infer_image_size(shape: Sequence[Any], layout: str, fallback: int) -> int:
    if len(shape) != 4:
        return fallback
    if layout == "nchw":
        h = shape[2]
        w = shape[3]
    else:
        h = shape[1]
        w = shape[2]
    if isinstance(h, int) and isinstance(w, int) and h > 0 and h == w:
        return h
    return fallback


def infer_tiny_yolo3_contract(
    session: ort.InferenceSession,
    export_metadata: Dict[str, Any],
    box_order: str,
) -> Dict[str, Any]:
    inputs = list(session.get_inputs())
    outputs = list(session.get_outputs())
    image_input = next((item for item in inputs if len(item.shape) == 4), None)
    image_shape_input = next((item for item in inputs if len(item.shape) == 2), None)
    if image_input is None:
        raise RuntimeError(
            "Could not infer tiny_yolo3 image input. "
            f"inputs={[(item.name, list(item.shape), item.type) for item in inputs]}"
        )

    image_layout = str(export_metadata.get("input_layout") or _infer_layout_from_shape(image_input.shape))
    image_value_range = str(export_metadata.get("input_value_range") or "unit_float")
    image_size = int(export_metadata.get("image_size") or _infer_image_size(image_input.shape, image_layout, 416))
    static_batch_size = export_metadata.get("static_batch_size")
    mode = str(export_metadata.get("model_variant") or "")

    if not mode:
        if image_shape_input is not None and len(outputs) >= 3:
            mode = "modelzoo_nms"
        elif len(outputs) >= 2 and all(len(output.shape) == 4 for output in outputs[:2]):
            mode = "yolo_heads"
        else:
            raise RuntimeError(
                "Could not infer tiny_yolo3 model variant. "
                f"inputs={[(item.name, list(item.shape), item.type) for item in inputs]} "
                f"outputs={[(item.name, list(item.shape), item.type) for item in outputs]}"
            )

    contract = {
        "mode": mode,
        "image_input_name": image_input.name,
        "image_input_dtype": ort_input_dtype_to_numpy(image_input.type).name,
        "image_input_shape": list(image_input.shape),
        "image_input_layout": image_layout,
        "image_input_value_range": image_value_range,
        "image_color_order": str(export_metadata.get("input_color_order") or "bgr"),
        "image_shape_input_name": image_shape_input.name if image_shape_input is not None else None,
        "image_shape_input_dtype": (
            ort_input_dtype_to_numpy(image_shape_input.type).name if image_shape_input is not None else None
        ),
        "image_shape_input_shape": list(image_shape_input.shape) if image_shape_input is not None else None,
        "output_names": [item.name for item in outputs],
        "box_order": str(export_metadata.get("box_order") or box_order),
        "image_size": image_size,
        "static_batch_size": static_batch_size,
        "num_classes": int(export_metadata.get("num_classes") or 80),
        "anchors": export_metadata.get("anchors"),
        "masks": export_metadata.get("masks"),
        "preferred_output_name": export_metadata.get("output_name"),
    }
    return contract


def infer_static_batch_size(runtime_contract: Dict[str, Any], requested_batch_size: int) -> int:
    static_batch_size = runtime_contract.get("static_batch_size")
    if isinstance(static_batch_size, int) and static_batch_size > 0:
        return static_batch_size
    input_shape = runtime_contract.get("image_input_shape") or []
    if input_shape:
        first_dim = input_shape[0]
        if isinstance(first_dim, int) and first_dim > 0:
            return first_dim
    return requested_batch_size


def image_size_from_contract(runtime_contract: Dict[str, Any]) -> int:
    image_size = runtime_contract.get("image_size")
    if isinstance(image_size, int) and image_size > 0:
        return image_size
    input_shape = runtime_contract["image_input_shape"]
    layout = runtime_contract["image_input_layout"]
    return _infer_image_size(input_shape, layout, 416)


def preprocess_image(
    image_path: Path,
    runtime_contract: Dict[str, Any],
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float]]:
    img = cv2.imread(image_path.as_posix())
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    img_h, img_w = img.shape[:2]
    img_size = image_size_from_contract(runtime_contract)
    img_lb, scale, pad_x, pad_y = letterbox(img, img_size)
    arr = img_lb.astype(np.float32)

    if runtime_contract["image_value_range"] == "unit_float":
        arr = arr / 255.0
    elif runtime_contract["image_value_range"] != "uint8":
        raise RuntimeError(f"Unsupported input value range: {runtime_contract['image_value_range']}")

    if runtime_contract["image_input_layout"] == "nchw":
        arr = np.transpose(arr, (2, 0, 1))
    elif runtime_contract["image_input_layout"] != "nhwc":
        raise RuntimeError(f"Unsupported input layout: {runtime_contract['image_input_layout']}")

    arr = arr.astype(np.dtype(runtime_contract["image_input_dtype"]), copy=False)

    image_shape = None
    if runtime_contract.get("image_shape_input_name"):
        image_shape = np.array(
            [img_h, img_w],
            dtype=np.dtype(runtime_contract["image_shape_input_dtype"]),
        )

    meta = {
        "img_h": float(img_h),
        "img_w": float(img_w),
        "scale": float(scale),
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
    }
    return arr, image_shape, meta


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


def create_input_feed(
    runtime_contract: Dict[str, Any],
    image_batch: np.ndarray,
    image_shape_batch: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    feed = {runtime_contract["image_input_name"]: image_batch}
    image_shape_input_name = runtime_contract.get("image_shape_input_name")
    if image_shape_input_name:
        if image_shape_batch is None:
            raise RuntimeError("image_shape input is required for this model")
        feed[str(image_shape_input_name)] = image_shape_batch
    return feed


def pick_output_map(session: ort.InferenceSession, values: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    outputs = session.get_outputs()
    return {output.name: np.asarray(value) for output, value in zip(outputs, values)}


def _normalize_boxes(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes)
    if boxes.ndim == 2:
        boxes = np.expand_dims(boxes, axis=0)
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise RuntimeError(f"Unexpected boxes shape: {boxes.shape}")
    return boxes


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores)
    if scores.ndim == 2:
        scores = np.expand_dims(scores, axis=0)
    if scores.ndim != 3:
        raise RuntimeError(f"Unexpected scores shape: {scores.shape}")
    return scores


def _normalize_indices(indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices)
    if indices.ndim == 3:
        indices = indices.reshape(-1, indices.shape[-1])
    if indices.ndim != 2 or indices.shape[-1] != 3:
        raise RuntimeError(f"Unexpected indices shape: {indices.shape}")
    return indices


def decode_tiny_yolo3_nms_outputs(
    output_map: Dict[str, np.ndarray],
    runtime_contract: Dict[str, Any],
    metas: List[Dict[str, float]],
    score_threshold: float,
    max_det: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    output_names = runtime_contract["output_names"]
    boxes = _normalize_boxes(output_map[output_names[0]])
    scores = _normalize_scores(output_map[output_names[1]])
    indices = _normalize_indices(output_map[output_names[2]])
    box_order = runtime_contract["box_order"]

    detections: List[Dict[str, Any]] = []
    per_image_count = {batch_index: 0 for batch_index in range(len(metas))}
    for row in indices.tolist():
        batch_index, class_index, box_index = [int(x) for x in row]
        if batch_index < 0 or class_index < 0 or box_index < 0:
            continue
        if batch_index >= len(metas):
            continue
        if class_index >= scores.shape[1] or box_index >= boxes.shape[1]:
            continue
        if per_image_count[batch_index] >= max_det:
            continue

        score = float(scores[batch_index, class_index, box_index])
        if score < score_threshold:
            continue

        box = boxes[batch_index, box_index].astype(np.float32)
        if box_order == "yxyx":
            y1, x1, y2, x2 = [float(x) for x in box.tolist()]
        elif box_order == "xyxy":
            x1, y1, x2, y2 = [float(x) for x in box.tolist()]
        else:
            raise RuntimeError(f"Unsupported box_order: {box_order}")

        meta = metas[batch_index]
        x1 = float(np.clip(x1, 0.0, meta["img_w"] - 1.0))
        y1 = float(np.clip(y1, 0.0, meta["img_h"] - 1.0))
        x2 = float(np.clip(x2, 0.0, meta["img_w"] - 1.0))
        y2 = float(np.clip(y2, 0.0, meta["img_h"] - 1.0))
        width = x2 - x1
        height = y2 - y1
        if width <= 1.0 or height <= 1.0:
            continue

        detections.append(
            {
                "image_id": int(meta["image_id"]),
                "category_id": COCO80_TO_91[class_index],
                "bbox": [x1, y1, width, height],
                "score": score,
            }
        )
        per_image_count[batch_index] += 1

    decode_probe = {
        "mode": runtime_contract["mode"],
        "output_names": output_names,
        "output_shapes": {key: list(np.asarray(value).shape) for key, value in output_map.items()},
        "box_order": box_order,
    }
    return detections, decode_probe


def compute_coco_metrics(
    ann_file: Path,
    predictions_path: Path,
    metrics_out: Path,
    metrics_text: Path,
    limit: int,
) -> Dict[str, Any]:
    from pycocotools.coco import COCO
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
            "metrics": {key: 0.0 for key in METRIC_KEYS},
            "note": "Predictions are empty; metrics were set to 0.0.",
        }
        metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        metrics_text.write_text("Predictions are empty; metrics were set to 0.0.\n", encoding="utf-8")
        return metrics

    dt = coco.loadRes(predictions_path.as_posix())
    ev = COCOeval(coco, dt, "bbox")
    if limit > 0:
        ev.params.imgIds = coco.getImgIds()[: limit]
    ev.evaluate()
    ev.accumulate()
    summary_buffer = io.StringIO()
    with contextlib.redirect_stdout(summary_buffer):
        ev.summarize()
    metrics_text.write_text(summary_buffer.getvalue(), encoding="utf-8")

    vals = [float(x) for x in ev.stats.tolist()]
    result = {
        "ann_file": ann_file.as_posix(),
        "predictions": predictions_path.as_posix(),
        "num_images": len(ev.params.imgIds) if ev.params.imgIds else len(coco.getImgIds()),
        "metrics": {key: value for key, value in zip(METRIC_KEYS, vals)},
    }
    metrics_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, output_path.open("wb") as file:
        file.write(response.read())
