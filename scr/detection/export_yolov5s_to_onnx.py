import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_URL = "https://github.com/ultralytics/yolov5.git"
DEFAULT_REF = "v7.0"
DEFAULT_WEIGHTS_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export classic YOLOv5s from official YOLOv5 repo to ONNX")
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=REPO_ROOT / "third_party/yolov5",
    )
    parser.add_argument("--repo-ref", type=str, default=DEFAULT_REF)
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO_ROOT / "models/detection/yolov5s.pt",
    )
    parser.add_argument("--weights-url", type=str, default=DEFAULT_WEIGHTS_URL)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "models/detection/yolov5s.onnx",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--clone-if-missing", action="store_true")
    parser.add_argument("--install-requirements", action="store_true")
    parser.add_argument("--skip-download-weights", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], cwd: Optional[Path] = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def patch_torch_load_compat(repo_dir: Path) -> None:
    experimental_py = repo_dir / "models/experimental.py"
    if not experimental_py.exists():
        return

    text = experimental_py.read_text(encoding="utf-8")
    old = "ckpt = torch.load(attempt_download(w), map_location='cpu')  # load"
    new = "ckpt = torch.load(attempt_download(w), map_location='cpu', weights_only=False)  # load"
    if old in text and new not in text:
        experimental_py.write_text(text.replace(old, new), encoding="utf-8")
        print(f"Patched torch.load compatibility in: {experimental_py}")


def ensure_repo(repo_dir: Path, repo_ref: str, clone_if_missing: bool) -> None:
    if repo_dir.exists():
        return
    if not clone_if_missing:
        raise FileNotFoundError(
            f"YOLOv5 repo not found: {repo_dir}. Use --clone-if-missing or clone it manually."
        )
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--branch", repo_ref, "--depth", "1", REPO_URL, repo_dir.as_posix()])


def ensure_weights(weights: Path, weights_url: str, skip_download: bool) -> None:
    if weights.exists():
        return
    if skip_download:
        raise FileNotFoundError(f"Weights not found: {weights}")
    weights.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading weights: {weights_url}")
    urllib.request.urlretrieve(weights_url, weights.as_posix())


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise RuntimeError("--batch-size must be > 0")

    ensure_repo(args.repo_dir, args.repo_ref, args.clone_if_missing)
    patch_torch_load_compat(args.repo_dir)
    ensure_weights(args.weights, args.weights_url, args.skip_download_weights)

    if args.install_requirements:
        run(
            [
                args.python,
                "-m",
                "pip",
                "install",
                "setuptools",
                "wheel",
            ]
        )
        run(
            [
                args.python,
                "-m",
                "pip",
                "install",
                "-r",
                (args.repo_dir / "requirements.txt").as_posix(),
            ]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_cmd = [
        args.python,
        (args.repo_dir / "export.py").as_posix(),
        "--weights",
        args.weights.as_posix(),
        "--include",
        "onnx",
        "--imgsz",
        str(args.imgsz),
        str(args.imgsz),
        "--opset",
        str(args.opset),
        "--batch-size",
        str(args.batch_size),
    ]
    run(export_cmd, cwd=args.repo_dir)

    exported = args.weights.with_suffix(".onnx")
    if not exported.exists():
        raise FileNotFoundError(f"Expected exported ONNX was not created: {exported}")

    if exported.resolve() != args.output.resolve():
        shutil.copyfile(exported, args.output)

    metadata = {
        "output": args.output.as_posix(),
        "repo_dir": args.repo_dir.as_posix(),
        "repo_ref": args.repo_ref,
        "weights": args.weights.as_posix(),
        "weights_url": args.weights_url,
        "imgsz": args.imgsz,
        "opset": args.opset,
        "batch_size": args.batch_size,
        "export_command": export_cmd,
        "model_family": "yolov5s",
        "source": "official_yolov5_repo",
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Exported ONNX model to: {args.output}")
    print(f"Saved export metadata: {metadata_path}")


if __name__ == "__main__":
    main()
