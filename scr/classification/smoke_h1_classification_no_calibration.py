import argparse
import json
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test for H1 stack: ONNX -> onnx_to_tf -> 1 TensorFlow inference (no calibration)"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT / "experiments/classification/resnet50.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Path to input image (default: first image from data/evaluation/imagenet)",
    )
    parser.add_argument("--input-tensor-name", type=str, default="input:1")
    parser.add_argument("--output-tensor-name", type=str, default="logits:0")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON file")
    return parser.parse_args()


def _as_graph_def(converted_model: Any, tf_module: Any) -> Any:
    if isinstance(converted_model, tuple):
        converted_model = converted_model[0]

    if isinstance(converted_model, tf_module.Graph):
        return converted_model.as_graph_def()
    if isinstance(converted_model, tf_module.compat.v1.GraphDef):
        return converted_model

    raise TypeError(f"Unsupported converted model type: {type(converted_model)}")


def preprocess(image: Image.Image) -> np.ndarray:
    # Manual H1 example preprocessing for ResNet-like classification (NHWC).
    image = image.convert("RGB")
    width, height = image.size
    min_dim = int(min(height, width) * 0.85)
    crop_top = (height - min_dim) // 2
    crop_left = (width - min_dim) // 2
    image = image.crop((crop_left, crop_top, crop_left + min_dim, crop_top + min_dim))
    image = image.resize((224, 224))
    tensor = np.asarray(image).astype(np.float32)
    tensor = tensor - np.array([123.68, 116.779, 103.939], dtype=np.float32)
    return np.expand_dims(tensor, axis=0)


def _ensure_tensor_name(name: str) -> str:
    return name if ":" in name else f"{name}:0"


def _resolve_tensor_name(graph: Any, requested_name: str) -> str:
    name = _ensure_tensor_name(requested_name)
    try:
        graph.get_tensor_by_name(name)
        return name
    except Exception:
        pass

    if ":" in name:
        op_name, idx_str = name.rsplit(":", 1)
        try:
            requested_idx = int(idx_str)
        except ValueError:
            requested_idx = 0
        op = graph.get_operation_by_name(op_name)
        if op.outputs:
            if requested_idx < len(op.outputs):
                return f"{op_name}:{requested_idx}"
            return f"{op_name}:0"
    raise KeyError(f"Tensor not found: {requested_name}")


def find_first_image(data_root: Path) -> Path:
    for path in sorted(data_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return path
    raise FileNotFoundError(f"No image files found under: {data_root}")


def run_inference(graph_def: Any, input_tensor_name: str, output_tensor_name: str, input_tensor: np.ndarray) -> np.ndarray:
    import tensorflow as tf  # imported lazily to keep script import light

    graph = tf.Graph()
    with graph.as_default():
        tf.graph_util.import_graph_def(graph_def, name="")

    input_name = _resolve_tensor_name(graph, input_tensor_name)
    output_name = _resolve_tensor_name(graph, output_tensor_name)
    if input_name != _ensure_tensor_name(input_tensor_name):
        print(f"Resolved input tensor: {input_tensor_name} -> {input_name}")
    if output_name != _ensure_tensor_name(output_tensor_name):
        print(f"Resolved output tensor: {output_tensor_name} -> {output_name}")
    input_tensor_ref = graph.get_tensor_by_name(input_name)
    output_tensor_ref = graph.get_tensor_by_name(output_name)

    with tf.compat.v1.Session(graph=graph) as sess:
        return sess.run(output_tensor_ref, feed_dict={input_tensor_ref: input_tensor})


def main() -> None:
    args = parse_args()
    if args.image_path is None:
        args.image_path = find_first_image(REPO_ROOT / "data/evaluation/imagenet")

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not args.image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    try:
        import onnx
        import tensorflow as tf
        from onnx_direct import onnx_to_tf
    except Exception as e:
        raise RuntimeError(
            "Missing dependencies for H1 smoke test. Required: onnx, tensorflow, onnx_direct (from H1 stack)."
        ) from e

    onnx_model = onnx.load(args.model_path.as_posix())
    converted = onnx_to_tf(onnx_model)
    graph_def = _as_graph_def(converted, tf)

    with Image.open(args.image_path) as image:
        input_tensor = preprocess(image)

    output = run_inference(
        graph_def=graph_def,
        input_tensor_name=args.input_tensor_name,
        output_tensor_name=args.output_tensor_name,
        input_tensor=input_tensor,
    )

    logits = np.asarray(output).reshape(-1)
    top5 = np.argsort(logits)[-5:][::-1] + 1  # 1-based class ids as in manual example
    result = {
        "model_path": args.model_path.as_posix(),
        "image_path": args.image_path.as_posix(),
        "input_tensor_name": args.input_tensor_name,
        "output_tensor_name": args.output_tensor_name,
        "output_shape": list(np.asarray(output).shape),
        "predicted_class_top1": int(top5[0]),
        "predicted_class_top5": [int(x) for x in top5.tolist()],
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
