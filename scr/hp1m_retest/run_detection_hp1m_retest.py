import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HP1M detection retest with valid vendor tiny_yolo3 settings"
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--mlperf-binary", type=str, default="mlperf")
    parser.add_argument(
        "--program-path",
        type=Path,
        default=Path("/home/smallpc_user/linq_files/tpu_programs/tiny_yolo3_b8_o5_128x128_asic.tpu"),
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
        "--experiments-dir",
        type=Path,
        default=REPO_ROOT / "experiments/detection_tiny_yolo3_HP1M_VALID",
    )
    parser.add_argument("--qps", type=int, default=2000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=5000)
    return parser.parse_args()


def run(cmd: List[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    py = args.python.as_posix()
    cmd = [
        py,
        (REPO_ROOT / "scr/detection/run_tiny_yolo3_vendor.py").as_posix(),
        "--python",
        py,
        "--mlperf-binary",
        args.mlperf_binary,
        "--program-path",
        args.program_path.as_posix(),
        "--img-dir",
        args.img_dir.as_posix(),
        "--ann-file",
        args.ann_file.as_posix(),
        "--experiments-dir",
        args.experiments_dir.as_posix(),
        "--qps",
        str(args.qps),
        "--runs",
        str(args.runs),
        "--limit",
        str(args.limit),
    ]
    run(cmd)


if __name__ == "__main__":
    main()
