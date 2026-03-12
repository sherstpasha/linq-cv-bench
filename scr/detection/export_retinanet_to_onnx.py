import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights, retinanet_resnet50_fpn


REPO_ROOT = Path(__file__).resolve().parents[2]


class RetinaNetExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, max_det: int, score_threshold: float) -> None:
        super().__init__()
        self.model = model
        self.max_det = int(max_det)
        self.score_threshold = float(score_threshold)

    def _pad_detections(self, detections: torch.Tensor) -> torch.Tensor:
        # Avoid Python control flow during export. Concatenate with a fixed zero
        # buffer and slice back to max_det so the graph keeps a real input.
        pad = detections.new_zeros((self.max_det, 6))
        return torch.cat([detections, pad], dim=0)[: self.max_det]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image = x[0]
        output = self.model([image])[0]
        boxes = output["boxes"]
        scores = output["scores"]
        labels = output["labels"].to(dtype=boxes.dtype)
        if self.score_threshold > 0:
            keep = scores >= self.score_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
        detections = torch.cat([boxes, scores.unsqueeze(1), labels.unsqueeze(1)], dim=1)
        return self._pad_detections(detections).unsqueeze(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export torchvision RetinaNet to ONNX")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "models/detection/retinanet_resnet50_fpn.onnx",
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--score-thres", type=float, default=0.0)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.batch_size != 1:
        print(f"Requested export batch-size={args.batch_size}, using batch-size=1.")
    export_batch_size = 1

    weights = None if args.no_pretrained else RetinaNet_ResNet50_FPN_Weights.COCO_V1
    model = retinanet_resnet50_fpn(weights=weights, box_detections_per_img=args.max_det)
    wrapped = RetinaNetExportWrapper(model.eval(), max_det=args.max_det, score_threshold=args.score_thres)

    dummy_input = torch.rand(export_batch_size, 3, args.height, args.width, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            dummy_input,
            args.output.as_posix(),
            dynamo=False,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=False,
            input_names=["images"],
            output_names=["detections"],
            dynamic_axes={
                "images": {0: "batch"},
                "detections": {0: "batch"},
            },
        )

    metadata = {
        "output": args.output.as_posix(),
        "opset": args.opset,
        "batch_size": export_batch_size,
        "pretrained": not args.no_pretrained,
        "weights": weights.__class__.__name__ + "." + weights.name if weights is not None else None,
        "input_name": "images",
        "output_name": "detections",
        "input_shape": [export_batch_size, 3, args.height, args.width],
        "input_layout": "nchw",
        "input_value_range": "unit_float",
        "max_det": args.max_det,
        "score_threshold": args.score_thres,
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Exported ONNX model to: {args.output.resolve()}")
    print(f"Saved export metadata: {metadata_path}")


if __name__ == "__main__":
    main()
