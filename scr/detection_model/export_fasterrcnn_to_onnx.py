import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

REPO_ROOT = Path(__file__).resolve().parents[2]


class FasterRCNNExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, max_det: int, score_threshold: float) -> None:
        super().__init__()
        self.model = model
        self.max_det = int(max_det)
        self.score_threshold = float(score_threshold)

    def _pad_detections(self, det: torch.Tensor) -> torch.Tensor:
        n = det.shape[0]
        if n >= self.max_det:
            return det[: self.max_det]
        pad = torch.zeros((self.max_det - n, 6), dtype=det.dtype, device=det.device)
        return torch.cat([det, pad], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ONNX export path for torchvision detection is far more stable with batch=1.
        # We export [1, max_det, 6] tensor: [x1, y1, x2, y2, score, coco_label_id].
        img = x[0]
        out = self.model([img])[0]
        boxes = out["boxes"]
        scores = out["scores"]
        labels = out["labels"].to(dtype=boxes.dtype)
        if self.score_threshold > 0:
            keep = scores >= self.score_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
        det = torch.cat([boxes, scores.unsqueeze(1), labels.unsqueeze(1)], dim=1) if boxes.numel() > 0 else boxes.new_zeros((0, 6))
        return self._pad_detections(det).unsqueeze(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export torchvision FasterRCNN to ONNX")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "experiments/detection_model/fasterrcnn_resnet50_fpn.onnx")
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
        print(f"Requested export batch-size={args.batch_size}, forcing batch-size=1 for stable FasterRCNN ONNX export.")
    export_batch_size = 1

    weights = None if args.no_pretrained else FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights, box_detections_per_img=args.max_det)
    wrapped = FasterRCNNExportWrapper(model.eval(), max_det=args.max_det, score_threshold=args.score_thres)

    dummy_input = torch.rand(export_batch_size, 3, args.height, args.width, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            dummy_input,
            args.output.as_posix(),
            dynamo=False,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["detections"],
            dynamic_axes={
                "images": {0: "batch"},
                "detections": {0: "batch"},
            },
        )
    print(f"Exported ONNX model to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
