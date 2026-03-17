import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_URL = "https://huggingface.co/onnxmodelzoo/tiny-yolov3-11/resolve/main/tiny-yolov3-11.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tiny YOLOv3 ONNX accuracy and performance on CPU or CUDA")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "models/detection/tiny-yolov3-11.onnx")
    parser.add_argument("--download-url", type=str, default=DEFAULT_MODEL_URL)
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
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--box-order", choices=["yxyx", "xyxy"], default="yxyx")
    parser.add_argument("--redownload-model", action="store_true")
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


def main() -> None:
    args = parse_args()
    py = args.python.as_posix()
    args.experiments_dir.mkdir(parents=True, exist_ok=True)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or (args.experiments_dir / "results_summary.json")

    download_cmd = None
    if args.redownload_model or not args.model_path.exists():
        download_cmd = [
            py,
            (THIS_DIR / "download_tiny_yolo3_onnx.py").as_posix(),
            "--output",
            args.model_path.as_posix(),
            "--url",
            args.download_url,
        ]
        if args.redownload_model:
            download_cmd.append("--force")
        run(download_cmd)

    accuracy_summary = args.experiments_dir / "accuracy" / "summary.json"
    performance_summary = args.experiments_dir / "performance" / "b1" / "summary.json"

    accuracy_cmd = [
        py,
        (THIS_DIR / "run_tiny_yolo3_onnx_accuracy.py").as_posix(),
        "--model-path",
        args.model_path.as_posix(),
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
        args.model_path.as_posix(),
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
        "model_path": args.model_path.as_posix(),
        "download_url": args.download_url,
        "img_dir": args.img_dir.as_posix(),
        "ann_file": args.ann_file.as_posix(),
        "provider": args.provider,
        "experiments_dir": args.experiments_dir.as_posix(),
        "model_download": {
            "command": download_cmd,
            "executed": bool(download_cmd),
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
