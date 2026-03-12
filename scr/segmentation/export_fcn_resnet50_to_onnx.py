import argparse
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
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = fcn_resnet50(weights=None if args.no_pretrained else FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
    wrapped = FCNExportWrapper(model.eval())
    dummy_input = torch.randn(1, 3, args.height, args.width, dtype=torch.float32)

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
    print(f"Exported ONNX model to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
