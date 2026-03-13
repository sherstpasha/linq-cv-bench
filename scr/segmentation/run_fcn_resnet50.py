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
        description="Export FCN-ResNet50 to ONNX, build TPU program, run direct TPU accuracy, and run MLPerf performance"
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT / "experiments/segmentation/fcn_resnet50.onnx",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=REPO_ROOT / "data/calibration/VOCdevkit/VOC2012/JPEGImages",
    )
    parser.add_argument(
        "--voc-root",
        type=Path,
        default=REPO_ROOT / "data/evaluation/VOCdevkit/VOC2012",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/segmentation",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=REPO_ROOT / "experiments/segmentation",
    )
    parser.add_argument("--model-name", type=str, default="fcn_resnet50")
    parser.add_argument("--compile-preset", type=str, default="O1", choices=["O1", "O5", "DEFAULT"])
    parser.add_argument("--height", type=int, default=520)
    parser.add_argument("--width", type=int, default=520)
    parser.add_argument("--accuracy-batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup-images", type=int, default=5)
    parser.add_argument("--num-classes", type=int, default=21)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--num-calibration-images", type=int, default=0)
    parser.add_argument("--calibration-chunk-size", type=int, default=64)
    parser.add_argument("--percentile", type=float, default=100.0)
    parser.add_argument("--batch-axis", type=int, default=0)
    parser.add_argument("--export-opset", type=int, default=18)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--reexport-model", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--skip-performance", action="store_true")
    parser.add_argument("--mlperf-binary", type=str, default="mlperf")
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

    output_json = args.output_json or (args.experiments_dir / "results_summary.json")
    build_summary_path = args.artifacts_dir / "build_summary.json"
    accuracy_summary_path = args.experiments_dir / "accuracy" / "summary.json"
    performance_dir = args.experiments_dir / "performance"

    export_cmd: Optional[List[str]] = None
    if args.reexport_model or not args.model_path.exists():
        export_cmd = [
            py,
            (THIS_DIR / "export_fcn_resnet50_to_onnx.py").as_posix(),
            "--output",
            args.model_path.as_posix(),
            "--opset",
            str(args.export_opset),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
        ]
        if args.no_pretrained:
            export_cmd.append("--no-pretrained")
        run(export_cmd)

    build_cmd: Optional[List[str]] = None
    if not args.skip_build:
        build_cmd = [
            py,
            (THIS_DIR / "build_fcn_resnet50_program.py").as_posix(),
            "--python",
            py,
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
            "--percentile",
            str(args.percentile),
            "--batch-axis",
            str(args.batch_axis),
            "--num-calibration-images",
            str(args.num_calibration_images),
            "--calibration-chunk-size",
            str(args.calibration_chunk_size),
            "--compile-preset",
            args.compile_preset,
            "--batch-sizes",
            "1",
            "8",
        ]
        run(build_cmd)

    accuracy_cmd: Optional[List[str]] = None
    if not args.skip_accuracy:
        accuracy_cmd = [
            py,
            (THIS_DIR / "run_fcn_resnet50_accuracy.py").as_posix(),
            "--program-path",
            (args.artifacts_dir / f"{args.model_name}_b{args.accuracy_batch_size}.tpu").as_posix(),
            "--voc-root",
            args.voc_root.as_posix(),
            "--predictions-dir",
            (args.experiments_dir / "accuracy" / "predictions").as_posix(),
            "--summary-out",
            accuracy_summary_path.as_posix(),
            "--batch-size",
            str(args.accuracy_batch_size),
            "--warmup-images",
            str(args.warmup_images),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--num-classes",
            str(args.num_classes),
            "--ignore-index",
            str(args.ignore_index),
        ]
        if args.limit > 0:
            accuracy_cmd += ["--limit", str(args.limit)]
        run(accuracy_cmd)

    performance_cmds: List[List[str]] = []
    if not args.skip_performance:
        for batch_size in (1, 8):
            cmd = [
                py,
                (THIS_DIR / "run_fcn_resnet50_performance.py").as_posix(),
                "--mlperf-binary",
                args.mlperf_binary,
                "--artifacts-dir",
                args.artifacts_dir.as_posix(),
                "--model-name",
                args.model_name,
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
        "pipeline": "fcn_resnet50_segmentation",
        "model_name": args.model_name,
        "model_path": args.model_path.as_posix(),
        "calibration_dir": args.calibration_dir.as_posix(),
        "voc_root": args.voc_root.as_posix(),
        "mlperf_binary": args.mlperf_binary,
        "artifacts_dir": args.artifacts_dir.as_posix(),
        "experiments_dir": args.experiments_dir.as_posix(),
        "model_export": {
            "command": export_cmd,
            "executed": export_cmd is not None,
        },
        "build": {
            "command": build_cmd,
            "summary": load_json(build_summary_path) if not args.skip_build else {"skipped": True},
        },
        "accuracy": {
            "command": accuracy_cmd,
            "summary": load_json(accuracy_summary_path) if not args.skip_accuracy else {"skipped": True},
        },
        "performance": {
            "commands": performance_cmds,
            "b1": load_json(performance_dir / "b1" / "summary.json")
            if not args.skip_performance
            else {"skipped": True},
            "b8": load_json(performance_dir / "b8" / "summary.json")
            if not args.skip_performance
            else {"skipped": True},
        },
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved segmentation summary: {output_json}")


if __name__ == "__main__":
    main()
