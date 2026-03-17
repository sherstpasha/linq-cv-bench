import argparse
import json
from pathlib import Path

from onnx_runtime_utils import download_file


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_URL = "https://huggingface.co/onnxmodelzoo/tiny-yolov3-11/resolve/main/tiny-yolov3-11.onnx"
DEFAULT_MODEL_PAGE = "https://huggingface.co/onnxmodelzoo/tiny-yolov3-11"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Tiny YOLOv3 ONNX model from ONNX Model Zoo")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "models/detection/tiny-yolov3-11.onnx")
    parser.add_argument("--url", type=str, default=DEFAULT_MODEL_URL)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        print(f"Model already exists: {args.output}")
    else:
        download_file(args.url, args.output)
        print(f"Downloaded tiny_yolo3 ONNX model to: {args.output}")

    metadata = {
        "model_name": "tiny-yolov3-11",
        "source_url": args.url,
        "source_page": DEFAULT_MODEL_PAGE,
        "input_layout": "nchw",
        "input_color_order": "bgr",
        "input_value_range": "unit_float",
        "preprocess": "letterbox",
        "input_shape": [1, 3, 416, 416],
        "image_shape_input": [1, 2],
        "box_order": "yxyx",
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved tiny_yolo3 ONNX metadata: {metadata_path}")


if __name__ == "__main__":
    main()
