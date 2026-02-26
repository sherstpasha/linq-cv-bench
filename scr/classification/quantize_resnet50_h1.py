import argparse
from pathlib import Path
from typing import Any, Dict, List

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
    parser.add_argument("--num-calibration-images", type=int, default=512, help="How many calibration images to use")
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


def resolve_quantize_api() -> Any:
    try:
        import dnn_quant  # type: ignore

        return dnn_quant
    except Exception:
        from tpu_framework import dnn_quant  # type: ignore

        return dnn_quant


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not args.calibration_dir.exists():
        raise FileNotFoundError(f"Calibration dir not found: {args.calibration_dir}")

    try:
        import onnx
        from onnx_direct import onnx_to_tf  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependencies: onnx, onnx_direct") from e

    dnn_quant = resolve_quantize_api()

    print(f"Loading ONNX: {args.model_path}")
    onnx_model = onnx.load(args.model_path.as_posix())
    converted = onnx_to_tf(onnx_model)
    if isinstance(converted, tuple):
        converted = converted[0]

    calibration_dict = build_calibration_dict(
        calibration_dir=args.calibration_dir,
        input_tensor_name=args.input_tensor_name,
        num_images=args.num_calibration_images,
    )
    calib_tensor = calibration_dict[args.input_tensor_name]
    print(f"Calibration tensor: {args.input_tensor_name} shape={tuple(calib_tensor.shape)}")

    quantized_model = dnn_quant.QuantizedModel.quantize(
        original_graph_def=converted,
        calibration_dict=calibration_dict,
        input_shapes={args.input_tensor_name: (1, 3, 224, 224)},
        output_nodes=[args.output_tensor_name],
        percentile=args.percentile,
        batch_axis=args.batch_axis,
    )

    args.output_qm.parent.mkdir(parents=True, exist_ok=True)
    quantized_model.save(args.output_qm.as_posix())
    print(f"Saved quantized model: {args.output_qm}")

    if args.save_quantized_graph_pb is not None:
        graph_def = quantized_model.as_graph(batch_size=1, batch_axis=args.batch_axis)
        args.save_quantized_graph_pb.parent.mkdir(parents=True, exist_ok=True)
        with args.save_quantized_graph_pb.open("wb") as f:
            f.write(graph_def.SerializeToString())
        print(f"Saved quantized graph: {args.save_quantized_graph_pb}")


if __name__ == "__main__":
    main()
