import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_PROGRAM_PATH = Path("linq_files/tpu_programs/tiny_yolo3_b8_o5_128x128_asic.tpu")
DEFAULT_INPUT_TENSOR_NAME = "input_1:0"
DEFAULT_IMG_SIZE = 416
DEFAULT_INPUT_LAYOUT = "nhwc"
DEFAULT_INPUT_RANGE = "unit_float"
DEFAULT_CONF_THRES = 0.001
DEFAULT_IOU_THRES = 0.45
DEFAULT_MAX_DET = 300
DEFAULT_BATCH_SIZE = 8
DEFAULT_LIMIT = 5000


def tail_text(path: Path, line_count: int = 80) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_bytes().decode("utf-8", errors="ignore")
    except Exception:
        return ""
    return "\n".join(text.splitlines()[-line_count:])


def run(cmd: List[str], stdout_path: Path, stderr_path: Path) -> Dict[str, str]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running: {' '.join(cmd)}")
    print(f"  stdout: {stdout_path}")
    print(f"  stderr: {stderr_path}")
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        result = subprocess.run(cmd, stdout=stdout_file, stderr=stderr_file)
    if result.returncode != 0:
        message = f"Command failed with exit code {result.returncode}. stdout: {stdout_path} stderr: {stderr_path}"
        stderr_tail = tail_text(stderr_path)
        stdout_tail = tail_text(stdout_path, 30)
        if stderr_tail:
            message += f"\n--- stderr tail ---\n{stderr_tail}"
        if stdout_tail:
            message += f"\n--- stdout tail ---\n{stdout_tail}"
        raise RuntimeError(message)
    return {"stdout": stdout_path.as_posix(), "stderr": stderr_path.as_posix()}


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {"error": f"missing file: {path.as_posix()}"}
    return json.loads(path.read_text(encoding="utf-8"))


def load_previous_stage(output_json: Path, stage_name: str) -> Dict:
    if not output_json.exists():
        return {}
    try:
        data = json.loads(output_json.read_text(encoding="utf-8"))
    except Exception:
        return {}
    stage = data.get(stage_name)
    return stage if isinstance(stage, dict) else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vendor tiny_yolo3: direct TPU quality on COCO + MLPerf performance")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--mlperf-binary", type=str, default="mlperf")
    parser.add_argument(
        "--program-path",
        type=Path,
        default=DEFAULT_PROGRAM_PATH,
    )
    parser.add_argument("--img-dir", type=Path, default=REPO_ROOT / "data/evaluation/MSCOCO2017/val2017")
    parser.add_argument(
        "--ann-file",
        type=Path,
        default=REPO_ROOT / "data/evaluation/MSCOCO2017/annotations/instances_val2017.json",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=REPO_ROOT / "experiments/detection/tiny_yolo3_vendor",
    )
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--input-layout", choices=["nchw", "nhwc"], default=DEFAULT_INPUT_LAYOUT)
    parser.add_argument("--input-range", choices=["unit_float", "uint8"], default=DEFAULT_INPUT_RANGE)
    parser.add_argument("--conf-thres", type=float, default=DEFAULT_CONF_THRES)
    parser.add_argument("--iou-thres", type=float, default=DEFAULT_IOU_THRES)
    parser.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--qps", type=int, default=400)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--input-tensor-name", type=str, default=DEFAULT_INPUT_TENSOR_NAME)
    parser.add_argument("--output-tensor-name", type=str, default=None)
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--skip-performance", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    py = args.python.as_posix()
    args.experiments_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.experiments_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.experiments_dir / "results_summary.json"
    previous_accuracy = load_previous_stage(output_json, "accuracy")
    previous_performance = load_previous_stage(output_json, "performance")
    accuracy_summary_path = args.experiments_dir / "accuracy_summary.json"
    performance_summary_path = args.experiments_dir / "performance" / f"b{args.batch_size}" / "summary.json"

    accuracy_cmd = None
    performance_cmd = None
    accuracy_logs = None
    performance_logs = None

    if not args.skip_accuracy:
        accuracy_cmd = [
            py,
            (THIS_DIR / "run_tiny_yolo3_accuracy.py").as_posix(),
            "--program-path",
            args.program_path.as_posix(),
            "--img-dir",
            args.img_dir.as_posix(),
            "--ann-file",
            args.ann_file.as_posix(),
            "--predictions-out",
            (args.experiments_dir / "predictions.json").as_posix(),
            "--summary-out",
            (args.experiments_dir / "accuracy_summary.json").as_posix(),
            "--metrics-out",
            (args.experiments_dir / "metrics.json").as_posix(),
            "--metrics-text",
            (args.experiments_dir / "metrics.txt").as_posix(),
            "--img-size",
            str(args.img_size),
            "--input-layout",
            args.input_layout,
            "--input-range",
            args.input_range,
            "--conf-thres",
            str(args.conf_thres),
            "--iou-thres",
            str(args.iou_thres),
            "--max-det",
            str(args.max_det),
            "--limit",
            str(args.limit),
            "--batch-size",
            str(args.batch_size),
        ]
        if args.input_tensor_name:
            accuracy_cmd += ["--input-tensor-name", args.input_tensor_name]
        if args.output_tensor_name:
            accuracy_cmd += ["--output-tensor-name", args.output_tensor_name]
        accuracy_logs = run(accuracy_cmd, logs_dir / "accuracy.stdout.log", logs_dir / "accuracy.stderr.log")

    if not args.skip_performance:
        performance_cmd = [
            py,
            (THIS_DIR / "run_tiny_yolo3_performance.py").as_posix(),
            "--mlperf-binary",
            args.mlperf_binary,
            "--program-path",
            args.program_path.as_posix(),
            "--batch-size",
            str(args.batch_size),
            "--qps",
            str(args.qps),
            "--runs",
            str(args.runs),
            "--output-dir",
            (args.experiments_dir / "performance").as_posix(),
        ]
        performance_logs = run(performance_cmd, logs_dir / "performance.stdout.log", logs_dir / "performance.stderr.log")

    summary = {
        "pipeline": "tiny_yolo3_vendor_detection",
        "python": py,
        "mlperf_binary": args.mlperf_binary,
        "program_path": args.program_path.as_posix(),
        "img_dir": args.img_dir.as_posix(),
        "ann_file": args.ann_file.as_posix(),
        "experiments_dir": args.experiments_dir.as_posix(),
        "accuracy": {
            "command": accuracy_cmd if accuracy_cmd is not None else previous_accuracy.get("command"),
            "logs": accuracy_logs if accuracy_logs is not None else previous_accuracy.get("logs"),
            "summary": (
                load_json(accuracy_summary_path)
                if accuracy_summary_path.exists()
                else previous_accuracy.get("summary", {"skipped": True})
            ),
        },
        "performance": {
            "command": performance_cmd if performance_cmd is not None else previous_performance.get("command"),
            "logs": performance_logs if performance_logs is not None else previous_performance.get("logs"),
            "summary": (
                load_json(performance_summary_path)
                if performance_summary_path.exists()
                else previous_performance.get("summary", {"skipped": True})
            ),
        },
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved tiny_yolo3 vendor results: {output_json}")


if __name__ == "__main__":
    main()
