import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_MODELZOO_URL = "https://huggingface.co/onnxmodelzoo/tiny-yolov3-11/resolve/main/tiny-yolov3-11.onnx"
DEFAULT_STRICT_CFG_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
DEFAULT_STRICT_WEIGHTS_URL = "https://data.pjreddie.com/files/yolov3-tiny.weights"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tiny YOLOv3 ONNX accuracy and performance on CPU or CUDA")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--export-python", type=Path, default=None)
    parser.add_argument("--model-source", choices=["modelzoo", "strict"], default="modelzoo")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--download-url", type=str, default=DEFAULT_MODELZOO_URL)
    parser.add_argument("--cfg-url", type=str, default=DEFAULT_STRICT_CFG_URL)
    parser.add_argument("--weights-url", type=str, default=DEFAULT_STRICT_WEIGHTS_URL)
    parser.add_argument("--cfg-path", type=Path, default=REPO_ROOT / "models/detection/yolov3-tiny.cfg")
    parser.add_argument("--weights-path", type=Path, default=REPO_ROOT / "models/detection/yolov3-tiny.weights")
    parser.add_argument("--img-dir", type=Path, default=REPO_ROOT / "data/evaluation/MSCOCO2017/val2017")
    parser.add_argument(
        "--ann-file",
        type=Path,
        default=REPO_ROOT / "data/evaluation/MSCOCO2017/annotations/instances_val2017.json",
    )
    parser.add_argument("--provider", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--accuracy-batch-size", type=int, default=1)
    parser.add_argument("--accuracy-limit", type=int, default=5000)
    parser.add_argument("--accuracy-warmup-images", type=int, default=10)
    parser.add_argument("--performance-batch-size", type=int, default=1)
    parser.add_argument("--performance-samples", type=int, default=500)
    parser.add_argument("--performance-warmup-images", type=int, default=10)
    parser.add_argument("--score-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--box-order", choices=["yxyx", "xyxy"], default="yxyx")
    parser.add_argument("--reexport-model", action="store_true")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=REPO_ROOT / "experiments/detection/tiny_yolo3_onnx",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def run(cmd: List[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {"error": f"missing file: {path.as_posix()}"}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_model_path(args: argparse.Namespace) -> Path:
    if args.model_path is not None:
        return args.model_path
    if args.model_source == "strict":
        return REPO_ROOT / "models/detection/tiny_yolo3_strict.onnx"
    return REPO_ROOT / "models/detection/tiny-yolov3-11.onnx"


def main() -> None:
    args = parse_args()
    py = args.python.as_posix()
    export_py = (args.export_python or args.python).as_posix()
    model_path = resolve_model_path(args)
    args.experiments_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or (args.experiments_dir / "results_summary.json")

    model_prepare_cmd: Optional[List[str]] = None
    if args.reexport_model or not model_path.exists():
        if args.model_source == "strict":
            model_prepare_cmd = [
                export_py,
                (THIS_DIR / "export_tiny_yolo3_strict_to_onnx.py").as_posix(),
                "--python",
                export_py,
                "--output",
                model_path.as_posix(),
                "--cfg-path",
                args.cfg_path.as_posix(),
                "--weights-path",
                args.weights_path.as_posix(),
                "--cfg-url",
                args.cfg_url,
                "--weights-url",
                args.weights_url,
            ]
            if args.reexport_model:
                model_prepare_cmd.append("--force-download")
        else:
            model_prepare_cmd = [
                py,
                (THIS_DIR / "download_tiny_yolo3_onnx.py").as_posix(),
                "--output",
                model_path.as_posix(),
                "--url",
                args.download_url,
            ]
            if args.reexport_model:
                model_prepare_cmd.append("--force")
        run(model_prepare_cmd)

    accuracy_summary = args.experiments_dir / "accuracy" / "summary.json"
    performance_summary = args.experiments_dir / "performance" / "b1" / "summary.json"

    accuracy_cmd = [
        py,
        (THIS_DIR / "run_tiny_yolo3_onnx_accuracy.py").as_posix(),
        "--model-path",
        model_path.as_posix(),
        "--img-dir",
        args.img_dir.as_posix(),
        "--ann-file",
        args.ann_file.as_posix(),
        "--provider",
        args.provider,
        "--batch-size",
        str(args.accuracy_batch_size),
        "--limit",
        str(args.accuracy_limit),
        "--warmup-images",
        str(args.accuracy_warmup_images),
        "--score-thres",
        str(args.score_thres),
        "--iou-thres",
        str(args.iou_thres),
        "--max-det",
        str(args.max_det),
        "--box-order",
        args.box_order,
        "--predictions-out",
        (args.experiments_dir / "accuracy" / "predictions.json").as_posix(),
        "--summary-out",
        accuracy_summary.as_posix(),
        "--metrics-out",
        (args.experiments_dir / "accuracy" / "metrics.json").as_posix(),
        "--metrics-text",
        (args.experiments_dir / "accuracy" / "metrics.txt").as_posix(),
    ]
    run(accuracy_cmd)

    performance_cmd = [
        py,
        (THIS_DIR / "run_tiny_yolo3_onnx_performance.py").as_posix(),
        "--model-path",
        model_path.as_posix(),
        "--img-dir",
        args.img_dir.as_posix(),
        "--ann-file",
        args.ann_file.as_posix(),
        "--provider",
        args.provider,
        "--batch-size",
        str(args.performance_batch_size),
        "--samples",
        str(args.performance_samples),
        "--warmup-images",
        str(args.performance_warmup_images),
        "--box-order",
        args.box_order,
        "--summary-out",
        performance_summary.as_posix(),
    ]
    run(performance_cmd)

    summary = {
        "pipeline": "tiny_yolo3_onnx",
        "model_source": args.model_source,
        "model_path": model_path.as_posix(),
        "img_dir": args.img_dir.as_posix(),
        "ann_file": args.ann_file.as_posix(),
        "provider": args.provider,
        "experiments_dir": args.experiments_dir.as_posix(),
        "model_prepare": {
            "command": model_prepare_cmd,
            "executed": bool(model_prepare_cmd),
        },
        "accuracy": {
            "command": accuracy_cmd,
            "summary": load_json(accuracy_summary),
        },
        "performance": {
            "command": performance_cmd,
            "summary": load_json(performance_summary),
        },
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved tiny_yolo3 ONNX summary: {output_json}")


if __name__ == "__main__":
    main()
