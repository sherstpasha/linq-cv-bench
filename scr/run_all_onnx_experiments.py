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


def normalize_provider_tag(providers: str) -> str:
    parts = [p.strip() for p in providers.split(",") if p.strip()]
    if not parts:
        return "auto"
    out = []
    for p in parts:
        low = p.lower()
        if "cuda" in low:
            out.append("cuda")
        elif "cpu" in low:
            out.append("cpu")
        elif "coreml" in low:
            out.append("coreml")
        else:
            out.append(low.replace("executionprovider", ""))
    return "-".join(out)


def collect_environment_info(py: str) -> Dict[str, Any]:
    probe = r"""
import importlib
import importlib.metadata
import json
import os
import platform
import sys

def safe_import(name):
    try:
        return importlib.import_module(name), None
    except Exception as e:
        return None, str(e)

def pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None

onnxruntime, ort_err = safe_import("onnxruntime")
torch, torch_err = safe_import("torch")

gpu = {
    "cuda_available": None,
    "device_count": 0,
    "devices": [],
}
if torch is not None:
    try:
        gpu["cuda_available"] = bool(torch.cuda.is_available())
        if gpu["cuda_available"]:
            gpu["device_count"] = int(torch.cuda.device_count())
            for i in range(gpu["device_count"]):
                props = torch.cuda.get_device_properties(i)
                gpu["devices"].append(
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "total_memory_mb": int(props.total_memory // (1024 * 1024)),
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                )
    except Exception as e:
        gpu["error"] = str(e)

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
        "onnxruntime": pkg_version("onnxruntime"),
        "onnxruntime_gpu": pkg_version("onnxruntime-gpu"),
        "torch": pkg_version("torch"),
        "torchvision": pkg_version("torchvision"),
        "ultralytics": pkg_version("ultralytics"),
        "pycocotools": pkg_version("pycocotools"),
    },
    "onnxruntime": {
        "available_providers": onnxruntime.get_available_providers() if onnxruntime else [],
        "import_error": ort_err,
    },
    "gpu": gpu,
    "env": {
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "yolo_autoinstall": os.environ.get("YOLO_AUTOINSTALL"),
    },
    "import_errors": {
        "torch": torch_err,
        "onnxruntime": ort_err,
    },
}
print(json.dumps(out))
"""
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
    p = argparse.ArgumentParser(description="Run ONNX classification + detection + segmentation and save one JSON")
    p.add_argument("--python", type=Path, default=None, help="Python interpreter for child scripts (default: current)")
    p.add_argument(
        "--experiments-dir",
        type=Path,
        default=REPO_ROOT / "experiments",
        help="Root directory for models, predictions, timing and metrics outputs",
    )
    p.add_argument("--providers", type=str, default=None, help="ONNX providers for all tasks")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for supported ONNX inference scripts")
    p.add_argument("--classification-limit", type=int, default=0)
    p.add_argument("--detection-limit", type=int, default=0)
    p.add_argument("--segmentation-limit", type=int, default=0)
    p.add_argument("--skip-classification-export", action="store_true")
    p.add_argument("--skip-detection-export", action="store_true")
    p.add_argument("--skip-segmentation-export", action="store_true")
    p.add_argument("--output-json", type=Path, default=None, help="Summary JSON path (default: <experiments-dir>/results_summary.json)")
    p.add_argument(
        "--environment-json",
        type=Path,
        default=None,
        help="Environment JSON path (default: <experiments-dir>/environment.json)",
    )
    return p.parse_args()


def ensure_module(py: str, module_name: str) -> None:
    proc = subprocess.run([py, "-c", f"import {module_name}"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Module '{module_name}' is missing in interpreter {py}. "
            f"Install dependencies in that env: `{py} -m pip install -r {REPO_ROOT / 'requirements.txt'}`"
        )


def ensure_path_exists(path: Path, hint: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Missing required path: {path}. {hint}")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise RuntimeError("--batch-size must be > 0")
    os.environ["YOLO_AUTOINSTALL"] = "False"
    providers = args.providers or ""
    providers_lc = providers.lower()
    detection_coreml_mode = "coreml" in providers_lc
    provider_tag = normalize_provider_tag(providers)
    experiment_tag = f"experiment_{provider_tag}_b{args.batch_size}"
    exp_dir = args.experiments_dir / experiment_tag
    cls_dir = exp_dir / "classification"
    det_dir = exp_dir / "detection"
    seg_dir = exp_dir / "segmentation"
    output_json = args.output_json or (exp_dir / "results_summary.json")
    environment_json = args.environment_json or (exp_dir / "environment.json")

    cls_model = cls_dir / "resnet50.onnx"
    cls_preds = cls_dir / "predictions.jsonl"
    cls_timing = cls_dir / "inference_timing.json"
    cls_metrics = cls_dir / "metrics.json"

    det_model = det_dir / "yolov5su.onnx"
    det_preds = det_dir / "predictions.json"
    det_timing = det_dir / "inference_timing.json"
    det_metrics_json = det_dir / "metrics.json"

    seg_model = seg_dir / "fcn_resnet50.onnx"
    seg_preds_dir = seg_dir / "predictions"
    seg_timing = seg_dir / "inference_timing.json"
    seg_metrics_json = seg_dir / "metrics.json"

    py = str(args.python) if args.python else sys.executable
    print(f"Using Python: {py}")
    print(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV', '(not set)')}")
    print("YOLO_AUTOINSTALL: False")
    print(f"Experiments dir: {exp_dir}")
    print(f"Experiment tag: {experiment_tag}")
    if detection_coreml_mode and args.batch_size != 1:
        print(
            "CoreML provider detected: detection will use batch_size=1 and static ONNX export "
            f"(requested global batch_size={args.batch_size})."
        )
    ensure_path_exists(REPO_ROOT / "data/evaluation/imagenet/val_map.txt", "Run split_datasets_for_calibration.py first.")
    ensure_path_exists(
        REPO_ROOT / "data/evaluation/MSCOCO2017/annotations/instances_val2017.json",
        "Run split_datasets_for_calibration.py first.",
    )
    ensure_path_exists(
        REPO_ROOT / "data/evaluation/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt",
        "Run split_datasets_for_calibration.py first.",
    )
    ensure_module(py, "onnxruntime")
    ensure_module(py, "torch")
    ensure_module(py, "ultralytics")
    ensure_module(py, "pycocotools")
    environment = collect_environment_info(py)

    if not args.skip_classification_export:
        run(
            [
                py,
                (SCR / "classification/export_resnet50_to_onnx.py").as_posix(),
                "--output",
                cls_model.as_posix(),
                "--batch-size",
                str(args.batch_size),
            ]
        )
    cls_infer = [
        py,
        (SCR / "classification/infer_resnet50_onnx.py").as_posix(),
        "--model-path",
        cls_model.as_posix(),
        "--predictions-out",
        cls_preds.as_posix(),
        "--timing-out",
        cls_timing.as_posix(),
        "--batch-size",
        str(args.batch_size),
    ]
    if providers:
        cls_infer += ["--providers", providers]
    if args.classification_limit > 0:
        cls_infer += ["--limit", str(args.classification_limit)]
    run(cls_infer)
    run(
        [
            py,
            (SCR / "classification/metrics.py").as_posix(),
            "--predictions",
            cls_preds.as_posix(),
            "--output-json",
            cls_metrics.as_posix(),
        ]
    )

    if not args.skip_detection_export:
        det_export = [py, (SCR / "detection/export_yolov5su_to_onnx.py").as_posix(), "--output", det_model.as_posix()]
        if args.batch_size > 1 and not detection_coreml_mode:
            det_export.append("--dynamic")
        run(det_export)
    det_batch_size = 1 if detection_coreml_mode else args.batch_size
    det_infer = [
        py,
        (SCR / "detection/infer_yolov5_onnx.py").as_posix(),
        "--model-path",
        det_model.as_posix(),
        "--predictions-out",
        det_preds.as_posix(),
        "--timing-out",
        det_timing.as_posix(),
        "--batch-size",
        str(det_batch_size),
    ]
    det_metrics = [
        py,
        (SCR / "detection/metrics.py").as_posix(),
        "--predictions",
        det_preds.as_posix(),
        "--output-json",
        det_metrics_json.as_posix(),
    ]
    if providers:
        det_infer += ["--providers", providers]
    if args.detection_limit > 0:
        det_infer += ["--limit", str(args.detection_limit)]
        det_metrics += ["--limit", str(args.detection_limit)]
    run(det_infer)
    run(det_metrics)

    if not args.skip_segmentation_export:
        run([py, (SCR / "segmentation/export_fcn_resnet50_to_onnx.py").as_posix(), "--output", seg_model.as_posix()])
    seg_infer = [
        py,
        (SCR / "segmentation/infer_fcn_resnet50_onnx.py").as_posix(),
        "--model-path",
        seg_model.as_posix(),
        "--predictions-dir",
        seg_preds_dir.as_posix(),
        "--timing-out",
        seg_timing.as_posix(),
    ]
    seg_metrics = [
        py,
        (SCR / "segmentation/metrics.py").as_posix(),
        "--predictions-dir",
        seg_preds_dir.as_posix(),
        "--output-json",
        seg_metrics_json.as_posix(),
    ]
    if providers:
        seg_infer += ["--providers", providers]
    if args.segmentation_limit > 0:
        seg_infer += ["--limit", str(args.segmentation_limit)]
        seg_metrics += ["--limit", str(args.segmentation_limit)]
    run(seg_infer)
    run(seg_metrics)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_tag": experiment_tag,
        "batch_size": args.batch_size,
        "detection_batch_size": det_batch_size,
        "detection_coreml_mode": detection_coreml_mode,
        "providers": providers or None,
        "environment": environment,
        "classification": {
            "timing": load_json(cls_timing),
            "metrics": load_json(cls_metrics),
        },
        "detection": {
            "timing": load_json(det_timing),
            "metrics": load_json(det_metrics_json),
        },
        "segmentation": {
            "timing": load_json(seg_timing),
            "metrics": load_json(seg_metrics_json),
        },
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
