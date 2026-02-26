import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H1 quantization for ResNet50 ONNX using calibration subset")
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "experiments/classification/resnet50.onnx")
    parser.add_argument("--calibration-dir", type=Path, default=REPO_ROOT / "data/calibration/imagenet")
    parser.add_argument("--output-qm", type=Path, default=REPO_ROOT / "experiments/classification/resnet50_h1_quantized.qm")
    parser.add_argument("--input-tensor-name", type=str, default="input:0")
    parser.add_argument("--output-tensor-name", type=str, default="fc_Gemm_3/add:0")
    parser.add_argument("--num-calibration-images", type=int, default=0, help="How many calibration images to use (0=all)")
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


def preprocess_resnet50(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    width, height = image.size
    scale = 256.0 / min(width, height)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    image = image.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))

    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return arr.astype(np.float32)


def build_calibration_dict(calibration_dir: Path, input_tensor_name: str, num_images: int) -> Dict[str, np.ndarray]:
    images = list_images(calibration_dir)
    if not images:
        raise RuntimeError(f"No images found in calibration dir: {calibration_dir}")
    if num_images > 0:
        images = images[: min(num_images, len(images))]

    tensors: List[np.ndarray] = []
    for img_path in images:
        with Image.open(img_path) as image:
            tensors.append(preprocess_resnet50(image))
    batch = np.stack(tensors, axis=0).astype(np.float32)  # NCHW
    return {input_tensor_name: batch}


def load_converted_graph(onnx_model: Any) -> Tuple[Any, Dict[str, str]]:
    # Preferred import path from framework package.
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


def resolve_output_node_name_for_regular_model(graph_def: Any, requested_name: str) -> str:
    node_names = {n.name for n in graph_def.node}
    if requested_name in node_names:
        return requested_name
    if requested_name.endswith(":0"):
        op_name = requested_name[:-2]
        if op_name in node_names:
            return op_name
    else:
        with_suffix = f"{requested_name}:0"
        if with_suffix in node_names:
            return with_suffix

    # Fallback: choose last likely classification head.
    candidates = [n.name for n in graph_def.node if "gemm" in n.name.lower() or "fc" in n.name.lower() or "logits" in n.name.lower()]
    if candidates:
        return candidates[-1]

    raise KeyError(f"Output node '{requested_name}' not found in converted graph")


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
    converted_graph, mapping = load_converted_graph(onnx_model)
    graph_def = to_graph_def(converted_graph)

    mapped_input = map_tensor_name(args.input_tensor_name, mapping)
    mapped_output = map_tensor_name(args.output_tensor_name, mapping)
    regular_output_node = resolve_output_node_name_for_regular_model(graph_def, mapped_output)

    calibration_dict = build_calibration_dict(
        calibration_dir=args.calibration_dir,
        input_tensor_name=mapped_input,
        num_images=args.num_calibration_images,
    )
    calib_tensor = calibration_dict[mapped_input]
    print(f"Calibration tensor: {mapped_input} shape={tuple(calib_tensor.shape)}")
    print(f"Output tensor (mapped): {mapped_output}")
    print(f"Output node (RegularModel): {regular_output_node}")

    input_shapes = {mapped_input: (1, 3, 224, 224)}
    model_kwargs = {
        "original_graph_def": graph_def,
        "input_shapes": input_shapes,
        "output_nodes": [regular_output_node],
    }
    if mapping:
        model_kwargs["anchors_mapping"] = mapping

    regular_model = RegularModel(**model_kwargs)

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


if __name__ == "__main__":
    main()
