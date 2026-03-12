import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export classic YOLOv5s to ONNX and run a COCO reference fragment")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=REPO_ROOT / "third_party/yolov5",
    )
    parser.add_argument("--repo-ref", type=str, default="v7.0")
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO_ROOT / "models/detection/yolov5s.pt",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT / "models/detection/yolov5s.onnx",
    )
    parser.add_argument(
        "--img-dir",
        type=Path,
        default=REPO_ROOT / "data/evaluation/MSCOCO2017/val2017",
    )
    parser.add_argument(
        "--ann-file",
        type=Path,
        default=REPO_ROOT / "data/evaluation/MSCOCO2017/annotations/instances_val2017.json",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=REPO_ROOT / "experiments/detection_onnx_reference",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--reexport-model", action="store_true")
    parser.add_argument("--providers", type=str, default=None)
    parser.add_argument("--clone-if-missing", action="store_true")
    parser.add_argument("--install-requirements", action="store_true")
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

    output_json = args.output_json or (args.experiments_dir / "results_summary.json")
    export_cmd: Optional[List[str]] = None
    if args.reexport_model or not args.model_path.exists():
        export_cmd = [
            py,
            (THIS_DIR / "export_yolov5s_to_onnx.py").as_posix(),
            "--repo-dir",
            args.repo_dir.as_posix(),
            "--repo-ref",
            args.repo_ref,
            "--weights",
            args.weights.as_posix(),
            "--output",
            args.model_path.as_posix(),
            "--imgsz",
            str(args.imgsz),
            "--opset",
            str(args.opset),
            "--batch-size",
            "1",
            "--clone-if-missing",
        ]
        if args.install_requirements:
            export_cmd.append("--install-requirements")
        run(export_cmd)

    infer_cmd = [
        py,
        (THIS_DIR / "run_yolov5_onnx_reference.py").as_posix(),
        "--model-path",
        args.model_path.as_posix(),
        "--img-dir",
        args.img_dir.as_posix(),
        "--ann-file",
        args.ann_file.as_posix(),
        "--predictions-out",
        (args.experiments_dir / "predictions.json").as_posix(),
        "--summary-out",
        (args.experiments_dir / "inference_summary.json").as_posix(),
        "--img-size",
        str(args.imgsz),
        "--batch-size",
        str(args.batch_size),
        "--limit",
        str(args.limit),
    ]
    if args.providers:
        infer_cmd += ["--providers", args.providers]
    run(infer_cmd)

    metrics_cmd = [
        py,
        (THIS_DIR / "metrics.py").as_posix(),
        "--ann-file",
        args.ann_file.as_posix(),
        "--predictions",
        (args.experiments_dir / "predictions.json").as_posix(),
        "--output-json",
        (args.experiments_dir / "metrics.json").as_posix(),
        "--limit",
        str(args.limit),
    ]
    run(metrics_cmd)

    summary = {
        "pipeline": "yolov5s_onnx_reference",
        "model_path": args.model_path.as_posix(),
        "weights": args.weights.as_posix(),
        "repo_dir": args.repo_dir.as_posix(),
        "repo_ref": args.repo_ref,
        "img_dir": args.img_dir.as_posix(),
        "ann_file": args.ann_file.as_posix(),
        "experiments_dir": args.experiments_dir.as_posix(),
        "export": {
            "command": export_cmd,
            "executed": export_cmd is not None,
            "metadata": load_json(args.model_path.with_suffix(".json")),
        },
        "inference": {
            "command": infer_cmd,
            "summary": load_json(args.experiments_dir / "inference_summary.json"),
        },
        "metrics": {
            "command": metrics_cmd,
            "summary": load_json(args.experiments_dir / "metrics.json"),
        },
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved detection summary: {output_json}")


if __name__ == "__main__":
    main()
