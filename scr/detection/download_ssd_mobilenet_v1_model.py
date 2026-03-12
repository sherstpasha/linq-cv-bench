import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_URL = (
    "https://github.com/onnx/models/raw/main/"
    "validated/vision/object_detection_segmentation/"
    "ssd-mobilenetv1/model/ssd_mobilenet_v1_10.onnx"
)
MODEL_SHA256 = "1fbcf47654165f2e0b5f1bdf3f123b9e9e1128cd6463717767b76ab4b5246f9a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SSD-MobileNetV1 ONNX model from the official ONNX Model Zoo"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "models/detection/ssd_mobilenet_v1_10.onnx",
    )
    parser.add_argument("--url", type=str, default=MODEL_URL)
    parser.add_argument("--sha256", type=str, default=MODEL_SHA256)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response, output.open("wb") as file:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            file.write(chunk)


def main() -> None:
    args = parse_args()
    metadata_path = args.output.with_suffix(".json")

    if args.output.exists() and not args.force:
        actual_sha = sha256sum(args.output)
        if actual_sha != args.sha256:
            raise RuntimeError(
                f"Existing model has unexpected sha256: {actual_sha}. "
                "Use --force to replace it."
            )
    else:
        download(args.url, args.output)
        actual_sha = sha256sum(args.output)
        if actual_sha != args.sha256:
            raise RuntimeError(
                f"Downloaded model sha256 mismatch: expected {args.sha256}, got {actual_sha}"
            )

    metadata = {
        "source": "onnx_model_zoo",
        "url": args.url,
        "sha256": args.sha256,
        "output": args.output.as_posix(),
        "input_name": "image_tensor:0",
        "output_names": [
            "detection_boxes:0",
            "detection_classes:0",
            "detection_scores:0",
            "num_detections:0",
        ],
        "input_layout": "nhwc",
        "input_dtype": "uint8",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Model is available at: {args.output}")
    print(f"Saved model metadata: {metadata_path}")


if __name__ == "__main__":
    main()
