import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HP1M classification retest with self-build O5 and optional vendor reference"
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--mlperf-binary", type=str, default="mlperf")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT / "models/classification/resnet50_HP1M_O5.onnx",
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
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/classification_HP1M_O5",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=REPO_ROOT / "experiments/classification_HP1M_O5",
    )
    parser.add_argument("--accuracy-samples", type=int, default=5000)
    parser.add_argument("--reexport-model", action="store_true")
    parser.add_argument("--run-vendor-reference", action="store_true")
    parser.add_argument(
        "--vendor-experiments-dir",
        type=Path,
        default=REPO_ROOT / "experiments/classification_vendor_mlperf_HP1M_O5_REF",
    )
    parser.add_argument(
        "--vendor-program-b1",
        type=Path,
        default=Path("/home/smallpc_user/linq_files/tpu_programs/resnet50_mlperf_b1_o5_128x128_asic.tpu"),
    )
    parser.add_argument(
        "--vendor-program-b8",
        type=Path,
        default=Path("/home/smallpc_user/linq_files/tpu_programs/resnet50_mlperf_b8_o5_128x128_asic.tpu"),
    )
    return parser.parse_args()


def run(cmd: List[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    py = args.python.as_posix()

    cmd = [
        py,
        (REPO_ROOT / "scr/classification/run_resnet50.py").as_posix(),
        "--python",
        py,
        "--mlperf-binary",
        args.mlperf_binary,
        "--model-path",
        args.model_path.as_posix(),
        "--calibration-dir",
        args.calibration_dir.as_posix(),
        "--evaluation-dir",
        args.evaluation_dir.as_posix(),
        "--artifacts-dir",
        args.artifacts_dir.as_posix(),
        "--experiments-dir",
        args.experiments_dir.as_posix(),
        "--compile-preset",
        "O5",
        "--accuracy-samples",
        str(args.accuracy_samples),
    ]
    if args.reexport_model:
        cmd.append("--reexport-model")
    run(cmd)

    if args.run_vendor_reference:
        vendor_cmd = [
            py,
            (REPO_ROOT / "scr/classification/run_resnet50_mlperf_vendor.py").as_posix(),
            "--python",
            py,
            "--mlperf-binary",
            args.mlperf_binary,
            "--program-b1",
            args.vendor_program_b1.as_posix(),
            "--program-b8",
            args.vendor_program_b8.as_posix(),
            "--dataset-dir",
            args.evaluation_dir.as_posix(),
            "--accuracy-samples",
            str(args.accuracy_samples),
            "--experiments-dir",
            args.vendor_experiments_dir.as_posix(),
        ]
        run(vendor_cmd)


if __name__ == "__main__":
    main()
