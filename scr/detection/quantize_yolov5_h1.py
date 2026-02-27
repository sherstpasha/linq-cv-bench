import argparse
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from coco_utils import letterbox

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H1 quantization for YOLOv5 ONNX using calibration images")
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "experiments/detection/yolov5su.onnx")
    parser.add_argument("--calibration-dir", type=Path, default=REPO_ROOT / "data/calibration/MSCOCO2017/val2017")
    parser.add_argument("--output-qm", type=Path, default=REPO_ROOT / "experiments/detection/yolov5su_h1_quantized.qm")
    parser.add_argument("--input-tensor-name", type=str, default=None, help="ONNX input tensor id (default: first ONNX input)")
    parser.add_argument("--output-tensor-name", type=str, default=None, help="ONNX output tensor id (default: first ONNX output)")
    parser.add_argument("--num-calibration-images", type=int, default=0, help="How many calibration images to use (0=all)")
    parser.add_argument("--calibration-chunk-size", type=int, default=128, help="Images per preprocessing chunk")
    parser.add_argument("--img-size", type=int, default=640, help="Letterbox target size")
    parser.add_argument("--percentile", type=float, default=100.0)
    parser.add_argument("--batch-axis", type=int, default=0)
    parser.add_argument("--save-quantized-graph-pb", type=Path, default=None, help="Optional .pb output from quantized model")
    return parser.parse_args()


def list_images(root: Path) -> List[Path]:
    images: List[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(p)
    return images


def preprocess_yolo(path: Path, img_size: int) -> np.ndarray:
    img = cv2.imread(path.as_posix())
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img_lb, _, _, _ = letterbox(img, img_size)
    x = img_lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return x.astype(np.float32)


def build_calibration_tensor_memmap(
    calibration_dir: Path,
    num_images: int,
    chunk_size: int,
    img_size: int,
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
    with tempfile.NamedTemporaryFile(prefix="calibration_yolo_", suffix=".dat", dir=tmp_dir.as_posix(), delete=False) as f:
        memmap_path = Path(f.name)

    batch = np.memmap(memmap_path.as_posix(), dtype=np.float32, mode="w+", shape=(n, 3, img_size, img_size))
    total_chunks = math.ceil(n / chunk_size)
    for chunk_idx in range(total_chunks):
        start = chunk_idx * chunk_size
        end = min(n, (chunk_idx + 1) * chunk_size)
        print(f"Preprocess calibration chunk {chunk_idx + 1}/{total_chunks} ({start}:{end})")
        tensors: List[np.ndarray] = []
        for img_path in images[start:end]:
            tensors.append(preprocess_yolo(img_path, img_size))
        batch[start:end] = np.stack(tensors, axis=0).astype(np.float32)
        batch.flush()

    return batch, memmap_path


def load_converted_graph(onnx_model: Any) -> Tuple[Any, Dict[str, str]]:
    try:
        from tpu_framework import onnx_to_tf  # type: ignore
    except Exception:
        from onnx_direct import onnx_to_tf  # type: ignore

    try:
        converted = onnx_to_tf(onnx_model=onnx_model, try_simplify=True)
    except TypeError:
        converted = onnx_to_tf(onnx_model)

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


def graph_diagnostics(graph_def: Any) -> Tuple[List[str], List[str], List[str]]:
    import tensorflow as tf  # type: ignore

    node_names = [n.name for n in graph_def.node]
    print(f"Converted GraphDef nodes: {len(node_names)}")

    graph = tf.Graph()
    with graph.as_default():
        tf.graph_util.import_graph_def(graph_def, name="")

    placeholders: List[str] = []
    leaves: List[str] = []
    for op in graph.get_operations():
        if op.type == "Placeholder":
            for out in op.outputs:
                placeholders.append(out.name)
                print(f"Placeholder: {out.name} shape={out.shape.as_list()}")
        for out in op.outputs:
            if len(out.consumers()) == 0 and out.dtype.name.startswith("float"):
                leaves.append(out.name)

    if leaves:
        print("Leaf float tensors (up to 10):")
        for name in leaves[:10]:
            print(f"  - {name}")

    interesting = [n for n in node_names if any(k in n.lower() for k in ("concat", "detect", "output", "model_24", "sigmoid"))]
    if interesting:
        print("Interesting node names (tail 40):")
        for n in interesting[-40:]:
            print(f"  - {n}")

    return node_names, placeholders, leaves


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not args.calibration_dir.exists():
        raise FileNotFoundError(f"Calibration dir not found: {args.calibration_dir}")

    try:
        import onnx
        from tpu_framework import RegularModel  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependencies: onnx and tpu_framework") from e

    print(f"Loading ONNX: {args.model_path}")
    onnx_model = onnx.load(args.model_path.as_posix())
    if not onnx_model.graph.input:
        raise RuntimeError("ONNX model has no inputs")
    if not onnx_model.graph.output:
        raise RuntimeError("ONNX model has no outputs")

    onnx_input_default = onnx_model.graph.input[0].name
    onnx_output_default = onnx_model.graph.output[0].name
    onnx_input_name = args.input_tensor_name or onnx_input_default
    onnx_output_name = args.output_tensor_name or onnx_output_default

    converted_graph, mapping = load_converted_graph(onnx_model)
    graph_def = to_graph_def(converted_graph)
    _, _, leaf_tensors = graph_diagnostics(graph_def)

    mapped_input = map_tensor_name(onnx_input_name, mapping)
    mapped_output = map_tensor_name(onnx_output_name, mapping)
    print(f"ONNX input/output: {onnx_input_name} -> {onnx_output_name}")
    print(f"Mapped input/output: {mapped_input} -> {mapped_output}")

    calib_tensor, memmap_path = build_calibration_tensor_memmap(
        calibration_dir=args.calibration_dir,
        num_images=args.num_calibration_images,
        chunk_size=args.calibration_chunk_size,
        img_size=args.img_size,
        tmp_dir=args.output_qm.parent,
    )
    calibration_dict = {mapped_input: calib_tensor}
    print(f"Calibration tensor: {mapped_input} shape={tuple(calib_tensor.shape)}")

    try:
        input_shapes = {mapped_input: (1, 3, args.img_size, args.img_size)}
        output_candidates = unique(
            [
                mapped_output,
                tensor_name_to_node_name(mapped_output),
                onnx_output_name,
                f"{onnx_output_name}:0",
            ]
            + leaf_tensors[:8]
        )

        regular_model = None
        selected_output = None
        errors: Dict[str, str] = {}

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
                errors[output_candidate] = str(e)

        if regular_model is None:
            print("Failed output candidates:")
            for k, v in errors.items():
                print(f"  - {k}: {v}")
            raise RuntimeError(f"Could not initialize RegularModel with outputs: {output_candidates}")

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
