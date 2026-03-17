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
        description="Run ResNet-50 ONNX accuracy and performance on CPU or CUDA"
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "models/classification/resnet50.onnx")
    parser.add_argument("--evaluation-dir", type=Path, default=REPO_ROOT / "data/evaluation/imagenet")
    parser.add_argument("--provider", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--accuracy-batch-size", type=int, default=8)
    parser.add_argument("--accuracy-samples", type=int, default=5000)
    parser.add_argument("--accuracy-warmup-batches", type=int, default=3)
    parser.add_argument("--performance-samples-b1", type=int, default=500)
    parser.add_argument("--performance-samples-b8", type=int, default=1000)
    parser.add_argument("--performance-warmup-batches", type=int, default=3)
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=REPO_ROOT / "experiments/classification_onnx",
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
    performance_b1_summary = args.experiments_dir / "performance" / "b1" / "summary.json"
    performance_b8_summary = args.experiments_dir / "performance" / "b8" / "summary.json"

    accuracy_cmd: Optional[List[str]] = [
        py,
        (THIS_DIR / "run_resnet50_onnx_accuracy.py").as_posix(),
        "--model-path",
        args.model_path.as_posix(),
        "--dataset-dir",
        args.evaluation_dir.as_posix(),
        "--provider",
        args.provider,
        "--batch-size",
        str(args.accuracy_batch_size),
        "--samples",
        str(args.accuracy_samples),
        "--warmup-batches",
        str(args.accuracy_warmup_batches),
        "--predictions-out",
        (args.experiments_dir / "accuracy" / "predictions.jsonl").as_posix(),
        "--summary-out",
        accuracy_summary.as_posix(),
    ]
    run(accuracy_cmd)

    performance_cmds: List[List[str]] = []
    for batch_size, sample_count, out_path in (
        (1, args.performance_samples_b1, performance_b1_summary),
        (8, args.performance_samples_b8, performance_b8_summary),
    ):
        cmd = [
            py,
            (THIS_DIR / "run_resnet50_onnx_performance.py").as_posix(),
            "--model-path",
            args.model_path.as_posix(),
            "--dataset-dir",
            args.evaluation_dir.as_posix(),
            "--provider",
            args.provider,
            "--batch-size",
            str(batch_size),
            "--samples",
            str(sample_count),
            "--warmup-batches",
            str(args.performance_warmup_batches),
            "--summary-out",
            out_path.as_posix(),
        ]
        run(cmd)
        performance_cmds.append(cmd)

    summary = {
        "pipeline": "resnet50_onnx",
        "model_path": args.model_path.as_posix(),
        "evaluation_dir": args.evaluation_dir.as_posix(),
        "provider": args.provider,
        "experiments_dir": args.experiments_dir.as_posix(),
        "accuracy": {
            "command": accuracy_cmd,
            "summary": load_json(accuracy_summary),
        },
        "performance": {
            "commands": performance_cmds,
            "b1": load_json(performance_b1_summary),
            "b8": load_json(performance_b8_summary),
        },
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved ONNX classification summary: {output_json}")


if __name__ == "__main__":
    main()
