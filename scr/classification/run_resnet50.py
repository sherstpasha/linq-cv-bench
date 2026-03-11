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
        description="Run the end-to-end INT8/TPU classification workflow for ResNet-50"
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
        default=THIS_DIR / "evaluate_resnet50_accuracy.py",
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
    parser.add_argument("--model-name", type=str, default="resnet50")
    parser.add_argument("--accuracy-samples", type=int, default=5000)
    parser.add_argument("--performance-runs", type=int, default=3)
    parser.add_argument("--compile-preset", type=str, default="O1", choices=["O1", "O5", "DEFAULT"])
    parser.add_argument("--export-model-if-missing", action="store_true")
    parser.add_argument("--reexport-model", action="store_true")
    parser.add_argument("--export-opset", type=int, default=13)
    parser.add_argument("--no-pretrained", action="store_true")
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


def load_model_export_metadata(model_path: Path) -> Dict:
    metadata_path = model_path.with_suffix(".json")
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def export_contract_matches(metadata: Dict) -> bool:
    return (
        metadata.get("input_layout") == "nhwc"
        and metadata.get("input_value_range") == "uint8"
    )


def main() -> None:
    args = parse_args()
    py = args.python.as_posix()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.experiments_dir.mkdir(parents=True, exist_ok=True)

    build_summary_path = args.artifacts_dir / "build_summary.json"
    accuracy_dir = args.experiments_dir / "accuracy"
    performance_dir = args.experiments_dir / "performance"
    output_json = args.output_json or (args.experiments_dir / "results_summary.json")
    export_cmd = None
    model_export_metadata = load_model_export_metadata(args.model_path)

    needs_export = args.reexport_model or not args.model_path.exists()
    if args.model_path.exists() and model_export_metadata:
        if not export_contract_matches(metadata=model_export_metadata):
            if args.reexport_model:
                needs_export = True
            else:
                raise RuntimeError(
                    "Existing ONNX export uses a different input contract. "
                    f"Current file: layout={model_export_metadata.get('input_layout')}, "
                    f"value_range={model_export_metadata.get('input_value_range')}. "
                    "Expected: layout=nhwc, value_range=uint8. "
                    "Use --reexport-model or choose another --model-path."
                )

    if needs_export:
        export_cmd = [
            py,
            (THIS_DIR / "export_resnet50_to_onnx.py").as_posix(),
            "--output",
            args.model_path.as_posix(),
            "--opset",
            str(args.export_opset),
        ]
        if args.no_pretrained:
            export_cmd.append("--no-pretrained")
        run(export_cmd)
        model_export_metadata = load_model_export_metadata(args.model_path)

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
        "model_export": {
            "command": export_cmd,
            "executed": export_cmd is not None,
            "metadata": model_export_metadata or None,
        },
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
