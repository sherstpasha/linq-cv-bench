import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
ACCURACY_RE = re.compile(r"accuracy=([0-9.]+)%\s*,\s*good=(\d+)\s*,\s*total=(\d+)")
PREDICTION_FORMAT_RE = re.compile(r"prediction_format=(\S+)")
LABEL_SHIFT_RE = re.compile(r"label_shift=([-0-9]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MLPerf accuracy for ResNet-50 on IVA H1"
    )
    parser.add_argument("--mlperf-binary", type=str, default="mlperf")
    parser.add_argument(
        "--accuracy-script",
        type=Path,
        default=THIS_DIR / "evaluate_resnet50_accuracy.py",
        help="Path to accuracy evaluator script",
    )
    parser.add_argument(
        "--program-path",
        type=Path,
        default=REPO_ROOT / "artifacts/classification/resnet50_b1.tpu",
        help="Path to batch-1 TPU program",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "data/evaluation/imagenet",
        help="Evaluation dataset directory containing val_map.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "experiments/classification/accuracy",
        help="Directory for logs and JSON summary",
    )
    parser.add_argument("--dataset-type", type=str, default="resnet50")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Number of evaluation samples to use from the start of val_map.txt; 0 means all rows",
    )
    parser.add_argument("--dtype", type=str, default="auto")
    return parser.parse_args()


def count_samples(val_map_path: Path) -> int:
    count = 0
    with val_map_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):
                count += 1
    return count


def run_and_capture(cmd: list[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> subprocess.CompletedProcess:
    process = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    stdout_path.write_text(process.stdout, encoding="utf-8")
    stderr_path.write_text(process.stderr, encoding="utf-8")
    return process


def main() -> None:
    args = parse_args()
    val_map_path = args.dataset_dir / "val_map.txt"
    if not args.program_path.exists():
        raise FileNotFoundError(f"TPU program not found: {args.program_path}")
    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {args.dataset_dir}")
    if not val_map_path.exists():
        raise FileNotFoundError(f"val_map.txt not found: {val_map_path}")
    if not args.accuracy_script.exists():
        raise FileNotFoundError(f"Accuracy evaluator not found: {args.accuracy_script}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    effective_samples = args.samples if args.samples > 0 else count_samples(val_map_path)

    mlperf_cmd = [
        args.mlperf_binary,
        "-s",
        "offline",
        "-m",
        "accuracy",
        "-o",
        str(args.offset),
        "-p",
        args.program_path.as_posix(),
        "-t",
        args.dataset_type,
        "-d",
        args.dataset_dir.as_posix(),
        "-n",
        str(effective_samples),
    ]
    mlperf_process = run_and_capture(
        cmd=mlperf_cmd,
        cwd=args.output_dir,
        stdout_path=args.output_dir / "mlperf_stdout.txt",
        stderr_path=args.output_dir / "mlperf_stderr.txt",
    )
    if mlperf_process.returncode != 0:
        raise RuntimeError(
            f"MLPerf accuracy command failed with code {mlperf_process.returncode}. "
            f"See {args.output_dir / 'mlperf_stderr.txt'}"
        )

    accuracy_log_path = args.output_dir / "mlperf_log_accuracy.json"
    if not accuracy_log_path.exists():
        raise FileNotFoundError(f"Expected MLPerf output was not created: {accuracy_log_path}")

    accuracy_cmd = [
        sys.executable,
        args.accuracy_script.as_posix(),
        "--imagenet-val-file",
        val_map_path.as_posix(),
        "--mlperf-accuracy-file",
        accuracy_log_path.as_posix(),
        "--dtype",
        args.dtype,
    ]
    accuracy_process = run_and_capture(
        cmd=accuracy_cmd,
        cwd=args.output_dir,
        stdout_path=args.output_dir / "accuracy_stdout.txt",
        stderr_path=args.output_dir / "accuracy_stderr.txt",
    )
    if accuracy_process.returncode != 0:
        raise RuntimeError(
            f"Accuracy evaluator failed with code {accuracy_process.returncode}. "
            f"See {args.output_dir / 'accuracy_stderr.txt'}"
        )

    combined_output = f"{accuracy_process.stdout}\n{accuracy_process.stderr}"
    accuracy_match = ACCURACY_RE.search(combined_output)
    if not accuracy_match:
        raise RuntimeError(
            "Could not parse accuracy evaluator output. "
            f"See {args.output_dir / 'accuracy_stdout.txt'}"
        )
    prediction_format_match = PREDICTION_FORMAT_RE.search(combined_output)
    label_shift_match = LABEL_SHIFT_RE.search(combined_output)

    summary = {
        "program_path": args.program_path.as_posix(),
        "dataset_dir": args.dataset_dir.as_posix(),
        "val_map": val_map_path.as_posix(),
        "dataset_type": args.dataset_type,
        "offset": args.offset,
        "requested_samples": args.samples,
        "effective_samples": effective_samples,
        "prediction_format": prediction_format_match.group(1) if prediction_format_match else None,
        "label_shift": int(label_shift_match.group(1)) if label_shift_match else None,
        "accuracy_percent": float(accuracy_match.group(1)),
        "good": int(accuracy_match.group(2)),
        "total": int(accuracy_match.group(3)),
        "mlperf_accuracy_log": accuracy_log_path.as_posix(),
        "mlperf_command": mlperf_cmd,
        "accuracy_command": accuracy_cmd,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved accuracy summary: {summary_path}")


if __name__ == "__main__":
    main()
