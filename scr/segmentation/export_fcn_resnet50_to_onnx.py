import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50

REPO_ROOT = Path(__file__).resolve().parents[2]


class FCNExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export torchvision FCN-ResNet50 to ONNX")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "experiments/segmentation/fcn_resnet50.onnx")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--height", type=int, default=520)
    parser.add_argument("--width", type=int, default=520)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise RuntimeError("--batch-size must be > 0")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    weights = None if args.no_pretrained else FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model = fcn_resnet50(weights=weights)
    wrapped = FCNExportWrapper(model.eval())
    input_shape = (args.batch_size, 3, args.height, args.width)
    dummy_input = torch.randn(*input_shape, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            dummy_input,
            args.output.as_posix(),
            dynamo=False,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"}, "logits": {0: "batch", 2: "height", 3: "width"}},
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
        "input_layout": "nchw",
        "input_value_range": "normalized",
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Exported ONNX model to: {args.output.resolve()}")
    print(f"Saved export metadata: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
