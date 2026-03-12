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
        description="Download SSD-MobileNetV1 ONNX, build TPU artifacts, and run direct TPU detection"
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT / "models/detection/ssd_mobilenet_v1_10.onnx",
    )
    parser.add_argument(
        "--model-url",
        type=str,
        default=(
            "https://github.com/onnx/models/raw/main/"
            "validated/vision/object_detection_segmentation/"
            "ssd-mobilenetv1/model/ssd_mobilenet_v1_10.onnx"
        ),
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
        default=REPO_ROOT / "artifacts/detection/ssd_mobilenet_v1",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=REPO_ROOT / "experiments/detection/ssd_mobilenet_v1",
    )
    parser.add_argument("--model-name", type=str, default="ssd_mobilenet_v1")
    parser.add_argument("--image-size", type=int, default=300)
    parser.add_argument("--compile-preset", type=str, default="O1", choices=["O1", "O5", "DEFAULT"])
    parser.add_argument("--num-calibration-images", type=int, default=500)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--run-batch-size", type=int, default=1)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--re-download-model", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
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
    run_summary_path = args.experiments_dir / "summary.json"

    download_cmd: Optional[List[str]] = None
    if args.re_download_model or not args.model_path.exists():
        download_cmd = [
            py,
            (THIS_DIR / "download_ssd_mobilenet_v1_model.py").as_posix(),
            "--output",
            args.model_path.as_posix(),
            "--url",
            args.model_url,
        ]
        if args.re_download_model:
            download_cmd.append("--force")
        run(download_cmd)

    build_cmd: Optional[List[str]] = None
    if not args.skip_build:
        build_cmd = [
            py,
            (THIS_DIR / "build_ssd_mobilenet_v1_program.py").as_posix(),
            "--model-path",
            args.model_path.as_posix(),
            "--calibration-dir",
            args.calibration_dir.as_posix(),
            "--artifacts-dir",
            args.artifacts_dir.as_posix(),
            "--model-name",
            args.model_name,
            "--image-size",
            str(args.image_size),
            "--compile-preset",
            args.compile_preset,
            "--num-calibration-images",
            str(args.num_calibration_images),
            "--batch-sizes",
            "1",
            "8",
        ]
        run(build_cmd)

    program_path = args.artifacts_dir / f"{args.model_name}_b{args.run_batch_size}.tpu"
    if not program_path.exists():
        raise FileNotFoundError(f"TPU program not found: {program_path}")

    run_cmd = [
        py,
        (THIS_DIR / "run_ssd_mobilenet_v1_tpu.py").as_posix(),
        "--program-path",
        program_path.as_posix(),
        "--build-summary",
        build_summary_path.as_posix(),
        "--img-dir",
        args.img_dir.as_posix(),
        "--ann-file",
        args.ann_file.as_posix(),
        "--predictions-out",
        (args.experiments_dir / "predictions.json").as_posix(),
        "--summary-out",
        run_summary_path.as_posix(),
        "--limit",
        str(args.limit),
        "--score-threshold",
        str(args.score_threshold),
    ]
    run(run_cmd)

    summary = {
        "pipeline": "ssd_mobilenet_v1_detection",
        "model_name": args.model_name,
        "model_path": args.model_path.as_posix(),
        "calibration_dir": args.calibration_dir.as_posix(),
        "img_dir": args.img_dir.as_posix(),
        "ann_file": args.ann_file.as_posix(),
        "artifacts_dir": args.artifacts_dir.as_posix(),
        "experiments_dir": args.experiments_dir.as_posix(),
        "download": {
            "command": download_cmd,
            "executed": download_cmd is not None,
            "metadata": load_json(args.model_path.with_suffix(".json")),
        },
        "build": {
            "command": build_cmd,
            "summary": load_json(build_summary_path) if not args.skip_build else {"skipped": True},
        },
        "run": {
            "command": run_cmd,
            "summary": load_json(run_summary_path),
        },
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved detection summary: {output_json}")


if __name__ == "__main__":
    main()
