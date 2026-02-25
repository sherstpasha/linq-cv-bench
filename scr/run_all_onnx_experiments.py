import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
SCR = REPO_ROOT / "scr"


def run(cmd):
    print("$", " ".join(str(x) for x in cmd))
    env = os.environ.copy()
    env["YOLO_AUTOINSTALL"] = "False"
    subprocess.run(cmd, check=True, env=env)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"error": f"missing file: {path.as_posix()}"}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ONNX classification + detection + segmentation and save one JSON")
    p.add_argument("--python", type=Path, default=None, help="Python interpreter for child scripts (default: current)")
    p.add_argument("--classification-providers", type=str, default=None, help="ONNX providers for classification infer")
    p.add_argument("--detection-providers", type=str, default=None, help="ONNX providers for detection infer")
    p.add_argument("--segmentation-providers", type=str, default=None, help="ONNX providers for segmentation infer")
    p.add_argument("--classification-limit", type=int, default=0)
    p.add_argument("--detection-limit", type=int, default=0)
    p.add_argument("--segmentation-limit", type=int, default=0)
    p.add_argument("--skip-classification-export", action="store_true")
    p.add_argument("--skip-detection-export", action="store_true")
    p.add_argument("--skip-segmentation-export", action="store_true")
    p.add_argument("--output-json", type=Path, default=REPO_ROOT / "experiments/results_summary.json")
    return p.parse_args()


def ensure_module(py: str, module_name: str) -> None:
    proc = subprocess.run([py, "-c", f"import {module_name}"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Module '{module_name}' is missing in interpreter {py}. "
            f"Install dependencies in that env: `{py} -m pip install -r {REPO_ROOT / 'requirements.txt'}`"
        )


def main() -> None:
    args = parse_args()
    os.environ["YOLO_AUTOINSTALL"] = "False"
    py = str(args.python) if args.python else sys.executable
    print(f"Using Python: {py}")
    print(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV', '(not set)')}")
    print("YOLO_AUTOINSTALL: False")
    ensure_module(py, "onnxruntime")
    ensure_module(py, "torch")
    ensure_module(py, "ultralytics")
    ensure_module(py, "pycocotools")

    if not args.skip_classification_export:
        run([py, (SCR / "classification/export_resnet50_to_onnx.py").as_posix()])
    cls_infer = [py, (SCR / "classification/infer_resnet50_onnx.py").as_posix()]
    if args.classification_providers:
        cls_infer += ["--providers", args.classification_providers]
    if args.classification_limit > 0:
        cls_infer += ["--limit", str(args.classification_limit)]
    run(cls_infer)
    run([py, (SCR / "classification/metrics.py").as_posix()])

    if not args.skip_detection_export:
        run([py, (SCR / "detection/export_yolov5su_to_onnx.py").as_posix()])
    det_infer = [py, (SCR / "detection/infer_yolov5_onnx.py").as_posix()]
    det_metrics = [py, (SCR / "detection/metrics.py").as_posix()]
    if args.detection_providers:
        det_infer += ["--providers", args.detection_providers]
    if args.detection_limit > 0:
        det_infer += ["--limit", str(args.detection_limit)]
        det_metrics += ["--limit", str(args.detection_limit)]
    run(det_infer)
    run(det_metrics)

    if not args.skip_segmentation_export:
        run([py, (SCR / "segmentation/export_fcn_resnet50_to_onnx.py").as_posix()])
    seg_infer = [py, (SCR / "segmentation/infer_fcn_resnet50_onnx.py").as_posix()]
    seg_metrics = [py, (SCR / "segmentation/metrics.py").as_posix()]
    if args.segmentation_providers:
        seg_infer += ["--providers", args.segmentation_providers]
    if args.segmentation_limit > 0:
        seg_infer += ["--limit", str(args.segmentation_limit)]
        seg_metrics += ["--limit", str(args.segmentation_limit)]
    run(seg_infer)
    run(seg_metrics)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": {
            "timing": load_json(REPO_ROOT / "experiments/classification/inference_timing.json"),
            "metrics": load_json(REPO_ROOT / "experiments/classification/metrics.json"),
        },
        "detection": {
            "timing": load_json(REPO_ROOT / "experiments/detection/inference_timing.json"),
            "metrics": load_json(REPO_ROOT / "experiments/detection/metrics.json"),
        },
        "segmentation": {
            "timing": load_json(REPO_ROOT / "experiments/segmentation/inference_timing.json"),
            "metrics": load_json(REPO_ROOT / "experiments/segmentation/metrics.json"),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary to: {args.output_json.as_posix()}")


if __name__ == "__main__":
    main()
