import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all HP1M retests")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--mlperf-binary", type=str, default="mlperf")
    parser.add_argument("--reexport-models", action="store_true")
    parser.add_argument("--run-vendor-classification-reference", action="store_true")
    return parser.parse_args()


def run(cmd: List[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    py = args.python.as_posix()

    classification_cmd = [
        py,
        (REPO_ROOT / "scr/hp1m_retest/run_classification_hp1m_retest.py").as_posix(),
        "--python",
        py,
        "--mlperf-binary",
        args.mlperf_binary,
    ]
    if args.reexport_models:
        classification_cmd.append("--reexport-model")
    if args.run_vendor_classification_reference:
        classification_cmd.append("--run-vendor-reference")
    run(classification_cmd)

    segmentation_cmd = [
        py,
        (REPO_ROOT / "scr/hp1m_retest/run_segmentation_hp1m_retest.py").as_posix(),
        "--python",
        py,
        "--mlperf-binary",
        args.mlperf_binary,
    ]
    if args.reexport_models:
        segmentation_cmd.append("--reexport-model")
    run(segmentation_cmd)

    detection_cmd = [
        py,
        (REPO_ROOT / "scr/hp1m_retest/run_detection_hp1m_retest.py").as_posix(),
        "--python",
        py,
        "--mlperf-binary",
        args.mlperf_binary,
    ]
    run(detection_cmd)


if __name__ == "__main__":
    main()
