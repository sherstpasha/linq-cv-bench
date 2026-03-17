import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vendor resnet50_mlperf accuracy and performance for comparison"
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--mlperf-binary", type=str, default="mlperf")
    parser.add_argument(
        "--program-b1",
        type=Path,
        default=Path("linq_files/tpu_programs/resnet50_mlperf_b1_o5_128x128_asic.tpu"),
    )
    parser.add_argument(
        "--program-b8",
        type=Path,
        default=Path("linq_files/tpu_programs/resnet50_mlperf_b8_o5_128x128_asic.tpu"),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "data/evaluation/imagenet",
    )
    parser.add_argument("--accuracy-samples", type=int, default=1000)
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=REPO_ROOT / "experiments/classification_vendor_mlperf",
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
    output_json = args.output_json or (args.experiments_dir / "results_summary.json")

    accuracy_summary = args.experiments_dir / "accuracy" / "summary.json"
    performance_dir = args.experiments_dir / "performance"

    accuracy_cmd: Optional[List[str]] = [
        py,
        (THIS_DIR / "run_resnet50_mlperf_accuracy.py").as_posix(),
        "--mlperf-binary",
        args.mlperf_binary,
        "--program-path",
        args.program_b1.as_posix(),
        "--dataset-dir",
        args.dataset_dir.as_posix(),
        "--samples",
        str(args.accuracy_samples),
        "--output-dir",
        (args.experiments_dir / "accuracy").as_posix(),
    ]
    run(accuracy_cmd)

    performance_cmds: List[List[str]] = []
    for batch_size, program_path in ((1, args.program_b1), (8, args.program_b8)):
        cmd = [
            py,
            (THIS_DIR / "run_resnet50_performance.py").as_posix(),
            "--mlperf-binary",
            args.mlperf_binary,
            "--program-path",
            program_path.as_posix(),
            "--model-name",
            "resnet50_mlperf",
            "--batch-size",
            str(batch_size),
            "--runs",
            "3",
            "--output-dir",
            performance_dir.as_posix(),
        ]
        run(cmd)
        performance_cmds.append(cmd)

    summary = {
        "pipeline": "resnet50_mlperf_vendor",
        "program_b1": args.program_b1.as_posix(),
        "program_b8": args.program_b8.as_posix(),
        "dataset_dir": args.dataset_dir.as_posix(),
        "mlperf_binary": args.mlperf_binary,
        "experiments_dir": args.experiments_dir.as_posix(),
        "accuracy": {
            "command": accuracy_cmd,
            "summary": load_json(accuracy_summary),
        },
        "performance": {
            "commands": performance_cmds,
            "b1": load_json(performance_dir / "b1" / "summary.json"),
            "b8": load_json(performance_dir / "b8" / "summary.json"),
        },
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved vendor MLPerf comparison summary: {output_json}")


if __name__ == "__main__":
    main()
