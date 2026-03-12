import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vendor resnet50_mlperf TPU programs strictly in the PMI configuration"
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--mlperf-binary", type=str, required=True)
    parser.add_argument(
        "--program-b1",
        type=Path,
        required=True,
        help="Path to vendor resnet50_mlperf batch-1 .tpu program",
    )
    parser.add_argument(
        "--program-b8",
        type=Path,
        required=True,
        help="Path to vendor resnet50_mlperf batch-8 .tpu program",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "data/evaluation/imagenet",
    )
    parser.add_argument(
        "--accuracy-script",
        type=Path,
        default=THIS_DIR / "accuracy-imagenet.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "experiments/classification_vendor_mlperf",
    )
    parser.add_argument("--accuracy-samples", type=int, default=1000)
    parser.add_argument("--performance-runs", type=int, default=3)
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.program_b1.exists():
        raise FileNotFoundError(f"Vendor batch-1 program not found: {args.program_b1}")
    if not args.program_b8.exists():
        raise FileNotFoundError(f"Vendor batch-8 program not found: {args.program_b8}")
    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {args.dataset_dir}")
    if not args.accuracy_script.exists():
        raise FileNotFoundError(f"Accuracy script not found: {args.accuracy_script}")

    accuracy_dir = args.output_dir / "accuracy"
    performance_dir = args.output_dir / "performance"

    accuracy_cmd = [
        py,
        (THIS_DIR / "run_resnet50_accuracy.py").as_posix(),
        "--mlperf-binary",
        args.mlperf_binary,
        "--accuracy-script",
        args.accuracy_script.as_posix(),
        "--program-path",
        args.program_b1.as_posix(),
        "--dataset-dir",
        args.dataset_dir.as_posix(),
        "--output-dir",
        accuracy_dir.as_posix(),
        "--dataset-type",
        "resnet50",
        "--offset",
        "0",
        "--samples",
        str(args.accuracy_samples),
        "--dtype",
        "int32",
    ]
    run(accuracy_cmd)

    performance_cmds = []
    for batch_size, program_path in ((1, args.program_b1), (8, args.program_b8)):
        perf_cmd = [
            py,
            (THIS_DIR / "run_resnet50_performance.py").as_posix(),
            "--mlperf-binary",
            args.mlperf_binary,
            "--program-path",
            program_path.as_posix(),
            "--artifacts-dir",
            program_path.parent.as_posix(),
            "--model-name",
            "resnet50_mlperf",
            "--batch-size",
            str(batch_size),
            "--runs",
            str(args.performance_runs),
            "--output-dir",
            performance_dir.as_posix(),
        ]
        run(perf_cmd)
        performance_cmds.append(perf_cmd)

    summary = {
        "pipeline": "vendor_resnet50_mlperf_pmi",
        "mlperf_binary": args.mlperf_binary,
        "program_b1": args.program_b1.as_posix(),
        "program_b8": args.program_b8.as_posix(),
        "dataset_dir": args.dataset_dir.as_posix(),
        "accuracy_script": args.accuracy_script.as_posix(),
        "accuracy_samples": args.accuracy_samples,
        "performance_runs": args.performance_runs,
        "accuracy": {
            "command": accuracy_cmd,
            "summary": load_json(accuracy_dir / "summary.json"),
        },
        "performance": {
            "commands": performance_cmds,
            "b1": load_json(performance_dir / "b1/summary.json"),
            "b8": load_json(performance_dir / "b8/summary.json"),
        },
    }
    summary_path = args.output_dir / "results_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved vendor MLPerf summary: {summary_path}")


if __name__ == "__main__":
    main()
