import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SCR = REPO_ROOT / "scr"


def run(cmd: List[str]) -> None:
    print("$", " ".join(str(x) for x in cmd))
    env = os.environ.copy()
    env["YOLO_AUTOINSTALL"] = "False"
    subprocess.run(cmd, check=True, env=env)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"error": f"missing file: {path.as_posix()}"}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_environment_info(py: str) -> Dict[str, Any]:
    probe = r'''
import importlib
import importlib.metadata
import json
import os
import platform
import sys


def pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None

pytpu_info = {"import_error": None, "devices": []}
try:
    import pytpu as tpu
    pytpu_info["devices"] = tpu.Device.list_devices()
except Exception as e:
    pytpu_info["import_error"] = str(e)

out = {
    "python": {
        "version": sys.version,
        "version_info": list(sys.version_info[:3]),
        "executable": sys.executable,
    },
    "platform": {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
    },
    "packages": {
        "tensorflow": pkg_version("tensorflow"),
        "onnx": pkg_version("onnx"),
        "tpu_framework": pkg_version("tpu_framework"),
        "pytpu": pkg_version("pytpu"),
        "pycocotools": pkg_version("pycocotools"),
    },
    "pytpu": pytpu_info,
    "env": {
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "yolo_autoinstall": os.environ.get("YOLO_AUTOINSTALL"),
    },
}
print(json.dumps(out))
'''
    proc = subprocess.run([py, "-c", probe], capture_output=True, text=True)
    if proc.returncode != 0:
        return {
            "error": "failed to collect environment info",
            "python": py,
            "returncode": proc.returncode,
            "stderr": proc.stderr.strip(),
        }
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "error": "failed to parse environment info output",
            "python": py,
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-2000:],
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run H1 TPU classification + detection + segmentation and save one JSON")
    p.add_argument("--python", type=Path, default=None, help="Python interpreter for child scripts (default: current)")
    p.add_argument("--experiments-dir", type=Path, default=REPO_ROOT / "experiments", help="Root output directory")
    p.add_argument("--experiment-suffix", type=str, default="h1tpu", help="Suffix used by task runners")
    p.add_argument("--batch-size", type=int, default=8)

    p.add_argument("--compile-preset", choices=["O1", "O5", "DEFAULT"], default="O1")
    p.add_argument("--device", type=str, default=None, help="TPU device path (e.g. /dev/tpu3)")

    p.add_argument("--classification-limit", type=int, default=0)
    p.add_argument("--detection-limit", type=int, default=0)
    p.add_argument("--segmentation-limit", type=int, default=0)

    p.add_argument("--skip-detection", action="store_true")
    p.add_argument("--skip-segmentation", action="store_true")
    p.add_argument("--skip-classification", action="store_true")

    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--environment-json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise RuntimeError("--batch-size must be > 0")

    py = str(args.python) if args.python else sys.executable
    os.environ["YOLO_AUTOINSTALL"] = "False"

    provider_tag = "h1tpu"
    experiment_tag = f"experiment_{provider_tag}_b{args.batch_size}"

    exp_dir = args.experiments_dir / experiment_tag
    cls_dir = exp_dir / "classification"
    det_dir = exp_dir / "detection"
    seg_dir = exp_dir / "segmentation"

    output_json = args.output_json or (exp_dir / "results_summary.json")
    environment_json = args.environment_json or (exp_dir / "environment.json")

    print(f"Using Python: {py}")
    print(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV', '(not set)')}")
    print("YOLO_AUTOINSTALL: False")
    print(f"Experiments dir: {exp_dir}")
    print(f"Experiment tag: {experiment_tag}")

    environment = collect_environment_info(py)

    class_suffix = f"{args.experiment_suffix}_b{args.batch_size}"
    det_seg_suffix = args.experiment_suffix
    det_seg_tag = f"experiment_{det_seg_suffix}_b{args.batch_size}"

    classification_status: Dict[str, Any] = {"skipped": args.skip_classification}
    detection_status: Dict[str, Any] = {"skipped": args.skip_detection}
    segmentation_status: Dict[str, Any] = {"skipped": args.skip_segmentation}

    if not args.skip_classification:
        cmd = [
            py,
            (SCR / "classification/run_full_h1_classification.py").as_posix(),
            "--output-dir",
            cls_dir.as_posix(),
            "--experiment-suffix",
            class_suffix,
            "--compile-preset",
            args.compile_preset,
            "--batch-size",
            str(args.batch_size),
        ]
        if args.classification_limit > 0:
            cmd += ["--limit", str(args.classification_limit)]
        run(cmd)
        classification_status.update(
            {
                "timing": load_json(cls_dir / f"inference_timing_{class_suffix}.json"),
                "metrics": load_json(cls_dir / f"metrics_{class_suffix}.json"),
                "run_params": load_json(cls_dir / f"run_params_{class_suffix}.json"),
            }
        )

    if not args.skip_detection:
        cmd = [
            py,
            (SCR / "detection/run_full_h1_detection.py").as_posix(),
            "--output-dir",
            det_dir.as_posix(),
            "--experiment-suffix",
            det_seg_suffix,
            "--compile-preset",
            args.compile_preset,
            "--batch-size",
            str(args.batch_size),
        ]
        if args.device:
            cmd += ["--device", args.device]
        if args.detection_limit > 0:
            cmd += ["--limit", str(args.detection_limit)]
        run(cmd)
        detection_status.update(
            {
                "timing": load_json(det_dir / f"inference_timing_{det_seg_tag}.json"),
                "metrics": load_json(det_dir / f"metrics_{det_seg_tag}.json"),
                "run_params": load_json(det_dir / f"run_params_{det_seg_tag}.json"),
            }
        )

    if not args.skip_segmentation:
        cmd = [
            py,
            (SCR / "segmentation/run_full_h1_segmentation.py").as_posix(),
            "--output-dir",
            seg_dir.as_posix(),
            "--experiment-suffix",
            det_seg_suffix,
            "--compile-preset",
            args.compile_preset,
            "--batch-size",
            str(args.batch_size),
        ]
        if args.device:
            cmd += ["--device", args.device]
        if args.segmentation_limit > 0:
            cmd += ["--limit", str(args.segmentation_limit)]
        run(cmd)
        segmentation_status.update(
            {
                "timing": load_json(seg_dir / f"inference_timing_{det_seg_tag}.json"),
                "metrics": load_json(seg_dir / f"metrics_{det_seg_tag}.json"),
                "run_params": load_json(seg_dir / f"run_params_{det_seg_tag}.json"),
            }
        )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_tag": experiment_tag,
        "batch_size": args.batch_size,
        "backend": "TPU_H1",
        "compile_preset": args.compile_preset,
        "device": args.device,
        "environment": environment,
        "classification": classification_status,
        "detection": detection_status,
        "segmentation": segmentation_status,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    environment_json.parent.mkdir(parents=True, exist_ok=True)
    with environment_json.open("w", encoding="utf-8") as f:
        json.dump(environment, f, indent=2)

    print(f"Saved summary to: {output_json.as_posix()}")
    print(f"Saved environment to: {environment_json.as_posix()}")


if __name__ == "__main__":
    main()
