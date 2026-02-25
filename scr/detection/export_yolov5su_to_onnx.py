import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLOv5su to ONNX")
    parser.add_argument("--weights", type=Path, default=REPO_ROOT / "yolov5su.pt")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "experiments/detection/yolov5su.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--dynamic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights.as_posix())
    exported_path = model.export(format="onnx", opset=args.opset, dynamic=args.dynamic, imgsz=args.imgsz)
    exported = Path(exported_path)
    if exported.resolve() != args.output.resolve():
        args.output.write_bytes(exported.read_bytes())
    print(f"Exported ONNX model to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
