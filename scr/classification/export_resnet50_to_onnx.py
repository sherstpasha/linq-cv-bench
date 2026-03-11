import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


class ResNet50ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", IMAGENET_MEAN.clone())
        self.register_buffer("std", IMAGENET_STD.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = x / 255.0
        x = (x - self.mean) / self.std
        return self.model(x)


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
    wrapped_model = ResNet50ExportWrapper(model=model.eval()).eval()
    input_shape = (args.batch_size, 224, 224, 3)
    dummy_input = torch.rand(*input_shape, dtype=torch.float32) * 255.0

    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
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
        "input_shape": list(input_shape),
        "input_layout": "nhwc",
        "input_value_range": "uint8",
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Exported ONNX model to: {args.output}")
    print(f"Saved export metadata: {metadata_path}")


if __name__ == "__main__":
    main()
