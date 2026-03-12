import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent


def run(cmd: List[str], stdout_path: Path, stderr_path: Path) -> Dict[str, str]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running: {' '.join(cmd)}")
    print(f"  stdout: {stdout_path}")
    print(f"  stderr: {stderr_path}")
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        result = subprocess.run(cmd, stdout=stdout_file, stderr=stderr_file)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}. "
            f"stdout: {stdout_path} stderr: {stderr_path}"
        )
    return {
        "stdout": stdout_path.as_posix(),
        "stderr": stderr_path.as_posix(),
    }


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {"error": f"missing file: {path.as_posix()}"}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full RetinaNet pipeline: export, ONNX reference, build TPU, direct TPU, COCO metrics"
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT / "models/detection/retinanet_resnet50_fpn.onnx",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=REPO_ROOT / "data/calibration/MSCOCO2017/val2017",
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
        "--artifacts-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/detection/retinanet",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=REPO_ROOT / "experiments/detection/retinanet",
    )
    parser.add_argument("--model-name", type=str, default="retinanet_resnet50_fpn")
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--score-thres", type=float, default=0.001)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--compile-preset", type=str, default="O1", choices=["O1", "O5", "DEFAULT"])
    parser.add_argument("--num-calibration-images", type=int, default=500)
    parser.add_argument("--export-opset", type=int, default=18)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--reexport-model", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--skip-tpu", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    py = args.python.as_posix()

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.experiments_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.experiments_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or (args.experiments_dir / "results_summary.json")
    build_summary_path = args.artifacts_dir / "build_summary.json"
    export_logs: Optional[Dict[str, str]] = None
    onnx_infer_logs: Optional[Dict[str, str]] = None
    onnx_metrics_logs: Optional[Dict[str, str]] = None
    build_logs: Optional[Dict[str, str]] = None
    tpu_infer_logs: Optional[Dict[str, str]] = None
    tpu_metrics_logs: Optional[Dict[str, str]] = None

    export_cmd: Optional[List[str]] = None
    if args.reexport_model or not args.model_path.exists():
        export_cmd = [
            py,
            (THIS_DIR / "export_retinanet_to_onnx.py").as_posix(),
            "--output",
            args.model_path.as_posix(),
            "--opset",
            str(args.export_opset),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--batch-size",
            "1",
            "--max-det",
            str(args.max_det),
            "--score-thres",
            str(args.score_thres),
        ]
        if args.no_pretrained:
            export_cmd.append("--no-pretrained")
        export_logs = run(
            export_cmd,
            stdout_path=logs_dir / "export.stdout.log",
            stderr_path=logs_dir / "export.stderr.log",
        )

    onnx_infer_cmd: Optional[List[str]] = None
    onnx_metrics_cmd: Optional[List[str]] = None
    if not args.skip_onnx:
        onnx_predictions = args.experiments_dir / "predictions_onnx.json"
        onnx_summary = args.experiments_dir / "onnx_summary.json"
        onnx_metrics = args.experiments_dir / "metrics_onnx.json"
        onnx_infer_cmd = [
            py,
            (THIS_DIR / "run_retinanet_onnx.py").as_posix(),
            "--model-path",
            args.model_path.as_posix(),
            "--img-dir",
            args.img_dir.as_posix(),
            "--ann-file",
            args.ann_file.as_posix(),
            "--predictions-out",
            onnx_predictions.as_posix(),
            "--summary-out",
            onnx_summary.as_posix(),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--conf-thres",
            str(args.conf_thres),
            "--limit",
            str(args.limit),
            "--batch-size",
            str(args.batch_size),
        ]
        onnx_infer_logs = run(
            onnx_infer_cmd,
            stdout_path=logs_dir / "onnx_infer.stdout.log",
            stderr_path=logs_dir / "onnx_infer.stderr.log",
        )

        onnx_metrics_cmd = [
            py,
            (THIS_DIR / "metrics.py").as_posix(),
            "--ann-file",
            args.ann_file.as_posix(),
            "--predictions",
            onnx_predictions.as_posix(),
            "--output-json",
            onnx_metrics.as_posix(),
            "--limit",
            str(args.limit),
        ]
        onnx_metrics_logs = run(
            onnx_metrics_cmd,
            stdout_path=logs_dir / "onnx_metrics.stdout.log",
            stderr_path=logs_dir / "onnx_metrics.stderr.log",
        )

    build_cmd: Optional[List[str]] = None
    if not args.skip_build:
        build_cmd = [
            py,
            (THIS_DIR / "build_retinanet_program.py").as_posix(),
            "--model-path",
            args.model_path.as_posix(),
            "--calibration-dir",
            args.calibration_dir.as_posix(),
            "--artifacts-dir",
            args.artifacts_dir.as_posix(),
            "--model-name",
            args.model_name,
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--compile-preset",
            args.compile_preset,
            "--num-calibration-images",
            str(args.num_calibration_images),
            "--batch-sizes",
            "1",
            str(args.batch_size),
        ]
        build_logs = run(
            build_cmd,
            stdout_path=logs_dir / "build.stdout.log",
            stderr_path=logs_dir / "build.stderr.log",
        )

    tpu_infer_cmd: Optional[List[str]] = None
    tpu_metrics_cmd: Optional[List[str]] = None
    if not args.skip_tpu:
        tpu_predictions = args.experiments_dir / "predictions_tpu.json"
        tpu_summary = args.experiments_dir / "tpu_summary.json"
        tpu_metrics = args.experiments_dir / "metrics_tpu.json"
        tpu_program = args.artifacts_dir / f"{args.model_name}_b{args.batch_size}.tpu"

        tpu_infer_cmd = [
            py,
            (THIS_DIR / "run_retinanet_tpu.py").as_posix(),
            "--program-path",
            tpu_program.as_posix(),
            "--build-summary",
            build_summary_path.as_posix(),
            "--img-dir",
            args.img_dir.as_posix(),
            "--ann-file",
            args.ann_file.as_posix(),
            "--predictions-out",
            tpu_predictions.as_posix(),
            "--summary-out",
            tpu_summary.as_posix(),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--conf-thres",
            str(args.conf_thres),
            "--limit",
            str(args.limit),
            "--batch-size",
            str(args.batch_size),
        ]
        tpu_infer_logs = run(
            tpu_infer_cmd,
            stdout_path=logs_dir / "tpu_infer.stdout.log",
            stderr_path=logs_dir / "tpu_infer.stderr.log",
        )

        tpu_metrics_cmd = [
            py,
            (THIS_DIR / "metrics.py").as_posix(),
            "--ann-file",
            args.ann_file.as_posix(),
            "--predictions",
            tpu_predictions.as_posix(),
            "--output-json",
            tpu_metrics.as_posix(),
            "--limit",
            str(args.limit),
        ]
        tpu_metrics_logs = run(
            tpu_metrics_cmd,
            stdout_path=logs_dir / "tpu_metrics.stdout.log",
            stderr_path=logs_dir / "tpu_metrics.stderr.log",
        )

    summary = {
        "pipeline": "retinanet_detection",
        "model_name": args.model_name,
        "model_path": args.model_path.as_posix(),
        "calibration_dir": args.calibration_dir.as_posix(),
        "img_dir": args.img_dir.as_posix(),
        "ann_file": args.ann_file.as_posix(),
        "artifacts_dir": args.artifacts_dir.as_posix(),
        "experiments_dir": args.experiments_dir.as_posix(),
        "export": {
            "command": export_cmd,
            "executed": export_cmd is not None,
            "logs": export_logs,
            "metadata": load_json(args.model_path.with_suffix(".json")),
        },
        "onnx": {
            "inference_command": onnx_infer_cmd,
            "metrics_command": onnx_metrics_cmd,
            "inference_logs": onnx_infer_logs,
            "metrics_logs": onnx_metrics_logs,
            "summary": load_json(args.experiments_dir / "onnx_summary.json") if not args.skip_onnx else {"skipped": True},
            "metrics": load_json(args.experiments_dir / "metrics_onnx.json") if not args.skip_onnx else {"skipped": True},
        },
        "build": {
            "command": build_cmd,
            "logs": build_logs,
            "summary": load_json(build_summary_path) if not args.skip_build else {"skipped": True},
        },
        "tpu": {
            "inference_command": tpu_infer_cmd,
            "metrics_command": tpu_metrics_cmd,
            "inference_logs": tpu_infer_logs,
            "metrics_logs": tpu_metrics_logs,
            "summary": load_json(args.experiments_dir / "tpu_summary.json") if not args.skip_tpu else {"skipped": True},
            "metrics": load_json(args.experiments_dir / "metrics_tpu.json") if not args.skip_tpu else {"skipped": True},
        },
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved detection summary: {output_json}")


if __name__ == "__main__":
    main()
