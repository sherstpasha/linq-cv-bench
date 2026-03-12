import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RE = re.compile(r"Result is\s*:\s*([A-Z]+)")
SAMPLES_RE = re.compile(r"Samples per second\s*:\s*([0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vendor tiny_yolo3 performance via MLPerf")
    parser.add_argument("--mlperf-binary", type=str, default="mlperf")
    parser.add_argument(
        "--program-path",
        type=Path,
        default=Path("linq_files/tpu_programs/tiny_yolo3_b8_o5_128x128_asic.tpu"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--qps", type=int, default=400)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "experiments/detection/tiny_yolo3_vendor/performance",
    )
    return parser.parse_args()


def parse_mlperf_summary(summary_path: Path) -> tuple[Optional[str], Optional[float]]:
    if not summary_path.exists():
        return None, None
    text = summary_path.read_text(encoding="utf-8", errors="replace")
    result_match = RESULT_RE.search(text)
    samples_match = SAMPLES_RE.search(text)
    result = result_match.group(1) if result_match else None
    samples_per_second = float(samples_match.group(1)) if samples_match else None
    return result, samples_per_second


def main() -> None:
    args = parse_args()
    if not args.program_path.exists():
        raise FileNotFoundError(f"TPU program not found: {args.program_path}")
    if args.runs <= 0:
        raise RuntimeError("--runs must be > 0")

    batch_dir = args.output_dir / f"b{args.batch_size}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    runs: List[dict] = []
    valid_samples: List[float] = []
    for run_index in range(1, args.runs + 1):
        run_dir = batch_dir / f"run_{run_index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.mlperf_binary,
            "-s",
            "offline",
            "-m",
            "performance",
            "-p",
            args.program_path.as_posix(),
            "-q",
            str(args.qps),
        ]
        process = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True)
        (run_dir / "mlperf_stdout.txt").write_text(process.stdout, encoding="utf-8")
        (run_dir / "mlperf_stderr.txt").write_text(process.stderr, encoding="utf-8")

        summary_file = run_dir / "mlperf_log_summary.txt"
        result, samples_per_second = parse_mlperf_summary(summary_file)
        is_valid = result == "VALID" and samples_per_second is not None
        if is_valid:
            valid_samples.append(samples_per_second)

        run_summary = {
            "run_index": run_index,
            "command": cmd,
            "returncode": process.returncode,
            "summary_file": summary_file.as_posix(),
            "result": result,
            "samples_per_second": samples_per_second,
            "is_valid": is_valid,
        }
        (run_dir / "summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
        runs.append(run_summary)

    all_valid = len(valid_samples) == args.runs
    summary = {
        "pipeline": "tiny_yolo3_vendor_mlperf_performance",
        "program_path": args.program_path.as_posix(),
        "batch_size": args.batch_size,
        "effective_qps": args.qps,
        "runs_requested": args.runs,
        "all_valid": all_valid,
        "valid_run_count": len(valid_samples),
        "mean_samples_per_second": (sum(valid_samples) / len(valid_samples)) if all_valid else None,
        "runs": runs,
    }
    summary_path = batch_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved tiny_yolo3 performance summary: {summary_path}")


if __name__ == "__main__":
    main()
