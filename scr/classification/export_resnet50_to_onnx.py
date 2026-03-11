import argparse
import json
from pathlib import Path

import torch
from torchvision.models import ResNet50_Weights, resnet50


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export torchvision ResNet-50 to ONNX for IVA H1 classification pipeline"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "models/classification/resnet50.onnx",
        help="Output ONNX path",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version; default keeps compatibility with vendor converters",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise RuntimeError("--batch-size must be > 0")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    weights = None if args.no_pretrained else ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.eval()
    dummy_input = torch.randn(args.batch_size, 3, 224, 224, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            args.output.as_posix(),
            dynamo=False,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
        )

    metadata = {
        "output": args.output.as_posix(),
        "opset": args.opset,
        "batch_size": args.batch_size,
        "pretrained": not args.no_pretrained,
        "weights": weights.__class__.__name__ + "." + weights.name if weights is not None else None,
        "input_name": "input",
        "output_name": "logits",
        "input_shape": [args.batch_size, 3, 224, 224],
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Exported ONNX model to: {args.output}")
    print(f"Saved export metadata: {metadata_path}")


if __name__ == "__main__":
    main()
