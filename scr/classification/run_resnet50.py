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
        description="Run the end-to-end INT8/TPU classification workflow for resnet50_mlperf"
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT / "models/classification/resnet50.onnx",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=REPO_ROOT / "data/calibration/imagenet",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=REPO_ROOT / "data/evaluation/imagenet",
    )
    parser.add_argument("--mlperf-binary", type=str, default="mlperf")
    parser.add_argument(
        "--accuracy-script",
        type=Path,
        default=THIS_DIR / "accuracy-imagenet.py",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/classification",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=REPO_ROOT / "experiments/classification",
    )
    parser.add_argument("--model-name", type=str, default="resnet50_mlperf")
    parser.add_argument("--accuracy-samples", type=int, default=0)
    parser.add_argument("--performance-runs", type=int, default=3)
    parser.add_argument("--compile-preset", type=str, default="O1", choices=["O1", "O5", "DEFAULT"])
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--skip-performance", action="store_true")
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
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.experiments_dir.mkdir(parents=True, exist_ok=True)

    build_summary_path = args.artifacts_dir / "build_summary.json"
    accuracy_dir = args.experiments_dir / "accuracy"
    performance_dir = args.experiments_dir / "performance"
    output_json = args.output_json or (args.experiments_dir / "results_summary.json")

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    if not args.skip_build:
        build_cmd = [
            py,
            (THIS_DIR / "build_resnet50_program.py").as_posix(),
            "--model-path",
            args.model_path.as_posix(),
            "--calibration-dir",
            args.calibration_dir.as_posix(),
            "--artifacts-dir",
            args.artifacts_dir.as_posix(),
            "--model-name",
            args.model_name,
            "--compile-preset",
            args.compile_preset,
            "--batch-sizes",
            "1",
            "8",
        ]
        run(build_cmd)
    else:
        build_cmd = None

    if not args.skip_accuracy:
        accuracy_cmd = [
            py,
            (THIS_DIR / "run_resnet50_accuracy.py").as_posix(),
            "--mlperf-binary",
            args.mlperf_binary,
            "--accuracy-script",
            args.accuracy_script.as_posix(),
            "--program-path",
            (args.artifacts_dir / f"{args.model_name}_b1.tpu").as_posix(),
            "--dataset-dir",
            args.evaluation_dir.as_posix(),
            "--output-dir",
            accuracy_dir.as_posix(),
            "--samples",
            str(args.accuracy_samples),
        ]
        run(accuracy_cmd)
    else:
        accuracy_cmd = None

    performance_cmds = []
    if not args.skip_performance:
        for batch_size in (1, 8):
            perf_cmd = [
                py,
                (THIS_DIR / "run_resnet50_performance.py").as_posix(),
                "--mlperf-binary",
                args.mlperf_binary,
                "--artifacts-dir",
                args.artifacts_dir.as_posix(),
                "--model-name",
                args.model_name,
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
        "model_name": args.model_name,
        "model_path": args.model_path.as_posix(),
        "calibration_dir": args.calibration_dir.as_posix(),
        "evaluation_dir": args.evaluation_dir.as_posix(),
        "mlperf_binary": args.mlperf_binary,
        "accuracy_script": args.accuracy_script.as_posix(),
        "artifacts_dir": args.artifacts_dir.as_posix(),
        "experiments_dir": args.experiments_dir.as_posix(),
        "build": {
            "command": build_cmd,
            "summary": load_json(build_summary_path) if not args.skip_build else {"skipped": True},
        },
        "accuracy": {
            "command": accuracy_cmd,
            "summary": load_json(accuracy_dir / "summary.json") if not args.skip_accuracy else {"skipped": True},
        },
        "performance": {
            "commands": performance_cmds,
            "b1": load_json(performance_dir / "b1/summary.json") if not args.skip_performance else {"skipped": True},
            "b8": load_json(performance_dir / "b8/summary.json") if not args.skip_performance else {"skipped": True},
        },
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved classification summary: {output_json}")


if __name__ == "__main__":
    main()
