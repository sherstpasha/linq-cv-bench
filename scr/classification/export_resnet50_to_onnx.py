import argparse
from pathlib import Path

import torch
from torchvision.models import ResNet50_Weights, resnet50

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export torchvision ResNet-50 to ONNX")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "experiments/classification/resnet50.onnx")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = resnet50(weights=None if args.no_pretrained else ResNet50_Weights.IMAGENET1K_V2)
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
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        )

    print(f"Exported ONNX model to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
