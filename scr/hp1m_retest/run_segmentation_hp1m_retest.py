import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HP1M segmentation retest with self-build O5"
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--mlperf-binary", type=str, default="mlperf")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT / "experiments/segmentation_HP1M_O5/fcn_resnet50.onnx",
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
        default=REPO_ROOT / "artifacts/segmentation_HP1M_O5",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=REPO_ROOT / "experiments/segmentation_HP1M_O5",
    )
    parser.add_argument("--reexport-model", action="store_true")
    parser.add_argument("--height", type=int, default=520)
    parser.add_argument("--width", type=int, default=520)
    return parser.parse_args()


def run(cmd: List[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    py = args.python.as_posix()

    cmd = [
        py,
        (REPO_ROOT / "scr/segmentation/run_fcn_resnet50.py").as_posix(),
        "--python",
        py,
        "--mlperf-binary",
        args.mlperf_binary,
        "--model-path",
        args.model_path.as_posix(),
        "--calibration-dir",
        args.calibration_dir.as_posix(),
        "--voc-root",
        args.voc_root.as_posix(),
        "--artifacts-dir",
        args.artifacts_dir.as_posix(),
        "--experiments-dir",
        args.experiments_dir.as_posix(),
        "--compile-preset",
        "O5",
        "--height",
        str(args.height),
        "--width",
        str(args.width),
    ]
    if args.reexport_model:
        cmd.append("--reexport-model")
    run(cmd)


if __name__ == "__main__":
    main()
