import argparse
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H1 quantization for RetinaNet ONNX")
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "experiments/detection_model/retinanet_resnet50_fpn.onnx")
    parser.add_argument("--calibration-dir", type=Path, default=REPO_ROOT / "data/calibration/MSCOCO2017/val2017")
    parser.add_argument("--output-qm", type=Path, default=REPO_ROOT / "experiments/detection_model/retinanet_resnet50_fpn_h1_quantized.qm")
    parser.add_argument("--input-tensor-name", type=str, default=None)
    parser.add_argument("--output-tensor-name", type=str, default=None)
    parser.add_argument("--num-calibration-images", type=int, default=0)
    parser.add_argument("--calibration-chunk-size", type=int, default=64)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--percentile", type=float, default=100.0)
    parser.add_argument("--batch-axis", type=int, default=0)
    parser.add_argument("--save-quantized-graph-pb", type=Path, default=None)
    return parser.parse_args()


def list_images(root: Path) -> List[Path]:
    images: List[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(p)
    return images


def preprocess(path: Path, width: int, height: int) -> np.ndarray:
    img = cv2.imread(path.as_posix())
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    x = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return x.astype(np.float32)


def build_calibration_tensor_memmap(
    calibration_dir: Path,
    num_images: int,
    chunk_size: int,
    width: int,
    height: int,
    tmp_dir: Path,
) -> Tuple[np.memmap, Path]:
    images = list_images(calibration_dir)
    if not images:
        raise RuntimeError(f"No images found in calibration dir: {calibration_dir}")
    if num_images > 0:
        images = images[: min(num_images, len(images))]

    n = len(images)
    if chunk_size <= 0:
        raise RuntimeError("--calibration-chunk-size must be > 0")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="calibration_frcnn_", suffix=".dat", dir=tmp_dir.as_posix(), delete=False) as f:
        memmap_path = Path(f.name)

    batch = np.memmap(memmap_path.as_posix(), dtype=np.float32, mode="w+", shape=(n, 3, height, width))
    total_chunks = math.ceil(n / chunk_size)
    for chunk_idx in range(total_chunks):
        start = chunk_idx * chunk_size
        end = min(n, (chunk_idx + 1) * chunk_size)
        print(f"Preprocess calibration chunk {chunk_idx + 1}/{total_chunks} ({start}:{end})")
        tensors: List[np.ndarray] = []
        for img_path in images[start:end]:
            tensors.append(preprocess(img_path, width=width, height=height))
        batch[start:end] = np.stack(tensors, axis=0).astype(np.float32)
        batch.flush()

    return batch, memmap_path


def force_static_onnx_input_shape(onnx_model: Any, input_name: str, height: int, width: int) -> None:
    for value_info in onnx_model.graph.input:
        if value_info.name != input_name:
            continue
        dims = value_info.type.tensor_type.shape.dim
        if len(dims) != 4:
            raise RuntimeError(f"Expected 4D input tensor, got {len(dims)} dims for '{input_name}'")
        dims[0].dim_value = 1
        dims[1].dim_value = 3
        dims[2].dim_value = int(height)
        dims[3].dim_value = int(width)
        return
    raise KeyError(f"Input tensor '{input_name}' not found in ONNX graph")


def load_converted_graph(onnx_model: Any) -> Tuple[Any, Dict[str, str]]:
    try:
        from tpu_framework import onnx_to_tf  # type: ignore
    except Exception:
        from onnx_direct import onnx_to_tf  # type: ignore

    converted = None
    errors: List[str] = []
    for kwargs in (
        {"onnx_model": onnx_model, "try_simplify": True},
        {"onnx_model": onnx_model, "try_simplify": False},
        {"onnx_model": onnx_model},
    ):
        try:
            converted = onnx_to_tf(**kwargs)
            break
        except Exception as e:
            errors.append(f"{kwargs}: {type(e).__name__}: {e}")
    if converted is None:
        raise RuntimeError("Failed to convert ONNX to TF graph: " + " | ".join(errors))

    if isinstance(converted, tuple):
        tf_graph = converted[0]
        mapping = converted[1] if len(converted) > 1 and isinstance(converted[1], dict) else {}
        return tf_graph, mapping
    return converted, {}


def to_graph_def(graph_like: Any) -> Any:
    import tensorflow as tf  # type: ignore

    if isinstance(graph_like, tf.Graph):
        return graph_like.as_graph_def()
    if isinstance(graph_like, tf.compat.v1.GraphDef):
        return graph_like
    raise TypeError(f"Unsupported graph type: {type(graph_like)}")


def map_tensor_name(name: str, mapping: Dict[str, str]) -> str:
    return mapping.get(name, name)


def tensor_name_to_node_name(name: str) -> str:
    return name.split(":", 1)[0]


def unique(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not args.calibration_dir.exists():
        raise FileNotFoundError(f"Calibration dir not found: {args.calibration_dir}")

    import onnx
    from tpu_framework import RegularModel  # type: ignore

    print(f"Loading ONNX: {args.model_path}")
    onnx_model = onnx.load(args.model_path.as_posix())

    onnx_input_default = onnx_model.graph.input[0].name
    onnx_output_default = onnx_model.graph.output[0].name
    onnx_input_name = args.input_tensor_name or onnx_input_default
    onnx_output_name = args.output_tensor_name or onnx_output_default

    force_static_onnx_input_shape(onnx_model, input_name=onnx_input_name, height=args.height, width=args.width)

    converted_graph, mapping = load_converted_graph(onnx_model)
    graph_def = to_graph_def(converted_graph)

    mapped_input = map_tensor_name(onnx_input_name, mapping)
    mapped_output = map_tensor_name(onnx_output_name, mapping)

    calib_tensor, memmap_path = build_calibration_tensor_memmap(
        calibration_dir=args.calibration_dir,
        num_images=args.num_calibration_images,
        chunk_size=args.calibration_chunk_size,
        width=args.width,
        height=args.height,
        tmp_dir=args.output_qm.parent,
    )
    calibration_dict = {mapped_input: calib_tensor}
    print(f"Calibration tensor: {mapped_input} shape={tuple(calib_tensor.shape)}")
    print(f"ONNX input/output: {onnx_input_name} -> {onnx_output_name}")
    print(f"Mapped input/output: {mapped_input} -> {mapped_output}")

    try:
        input_shapes = {mapped_input: (1, 3, args.height, args.width)}
        output_candidates = unique([mapped_output, tensor_name_to_node_name(mapped_output), onnx_output_name, f"{onnx_output_name}:0"])

        regular_model = None
        selected_output = None
        last_error: Exception | None = None

        for output_candidate in output_candidates:
            model_kwargs = {
                "original_graph_def": graph_def,
                "input_shapes": input_shapes,
                "output_nodes": [output_candidate],
            }
            if mapping:
                model_kwargs["anchors_mapping"] = mapping
            try:
                regular_model = RegularModel(**model_kwargs)
                selected_output = output_candidate
                break
            except Exception as e:
                last_error = e

        if regular_model is None:
            raise RuntimeError(f"Could not initialize RegularModel with outputs: {output_candidates}") from last_error

        print(f"Output node (RegularModel): {selected_output}")

        try:
            thresholds = regular_model.calibrate(calibration_data=calibration_dict, percentile=args.percentile)
        except TypeError:
            thresholds = regular_model.calibrate(calibration_data=calibration_dict)
        quantized_model = regular_model.quantize(thresholds)

        args.output_qm.parent.mkdir(parents=True, exist_ok=True)
        quantized_model.save(file_dir=args.output_qm.parent.as_posix(), file_name=args.output_qm.name)
        print(f"Saved quantized model: {args.output_qm}")

        if args.save_quantized_graph_pb is not None:
            quant_graph = quantized_model.as_graph(batch_size=1, batch_axis=args.batch_axis)
            quant_graph_def = to_graph_def(quant_graph)
            args.save_quantized_graph_pb.parent.mkdir(parents=True, exist_ok=True)
            with args.save_quantized_graph_pb.open("wb") as f:
                f.write(quant_graph_def.SerializeToString())
            print(f"Saved quantized graph: {args.save_quantized_graph_pb}")
    finally:
        try:
            memmap_path.unlink(missing_ok=True)
            print(f"Removed temporary calibration tensor: {memmap_path}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
