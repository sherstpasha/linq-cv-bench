import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("YOLO_AUTOINSTALL", "False")

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILTIN_WEIGHTS = {"yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLOv8s from Ultralytics to ONNX")
    parser.add_argument("--weights", type=str, default="yolov8s.pt")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "models/detection/yolov8s.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--simplify", action="store_true")
    return parser.parse_args()


def resolve_weights_source(weights: str) -> str:
    candidate = Path(weights).expanduser()
    if candidate.exists():
        return candidate.as_posix()
    if candidate.name in BUILTIN_WEIGHTS:
        return candidate.name
    return weights


def load_onnx_metadata(onnx_path: Path) -> Dict[str, Any]:
    import onnx

    model = onnx.load(onnx_path.as_posix())
    input_name = model.graph.input[0].name if model.graph.input else None
    output_name = model.graph.output[0].name if model.graph.output else None
    input_shape = None
    if model.graph.input:
        dims = model.graph.input[0].type.tensor_type.shape.dim
        input_shape = [dim.dim_value if dim.HasField("dim_value") else None for dim in dims]
    return {
        "input_name": input_name,
        "output_name": output_name,
        "input_shape": input_shape,
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except Exception as error:
        raise RuntimeError("Missing dependency: ultralytics") from error

    weights_source = resolve_weights_source(args.weights)
    model = YOLO(weights_source)
    export_path = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            batch=args.batch_size,
            dynamic=args.dynamic,
            simplify=args.simplify,
            nms=False,
        )
    )

    if export_path.resolve() != args.output.resolve():
        args.output.write_bytes(export_path.read_bytes())

    onnx_meta = load_onnx_metadata(args.output)
    metadata = {
        "output": args.output.as_posix(),
        "weights": args.weights,
        "weights_source": weights_source,
        "opset": args.opset,
        "batch_size": args.batch_size,
        "imgsz": args.imgsz,
        "dynamic": args.dynamic,
        "simplify": args.simplify,
        **onnx_meta,
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Exported ONNX model to: {args.output}")
    print(f"Saved export metadata: {metadata_path}")


if __name__ == "__main__":
    main()
