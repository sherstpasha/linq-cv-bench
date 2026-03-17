import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from onnx_runtime_utils import load_val_rows


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vendor MLPerf accuracy for resnet50_mlperf")
    parser.add_argument("--mlperf-binary", type=str, default="mlperf")
    parser.add_argument(
        "--program-path",
        type=Path,
        default=Path("linq_files/tpu_programs/resnet50_mlperf_b1_o5_128x128_asic.tpu"),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "data/evaluation/imagenet",
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "experiments/classification_vendor_mlperf/accuracy",
    )
    return parser.parse_args()


def decode_prediction(hex_data: str) -> Tuple[int, str]:
    raw = bytes.fromhex(hex_data)
    scalar_dtypes = [np.int64, np.int32, np.uint64, np.uint32]
    for dtype in scalar_dtypes:
        if len(raw) == np.dtype(dtype).itemsize:
            value = int(np.frombuffer(raw, dtype=dtype)[0])
            return value, f"{np.dtype(dtype).name}_scalar"

    vector_dtypes = [np.float32, np.float16, np.int8, np.uint8, np.int32, np.int64]
    for dtype in vector_dtypes:
        itemsize = np.dtype(dtype).itemsize
        if len(raw) > itemsize and len(raw) % itemsize == 0:
            arr = np.frombuffer(raw, dtype=dtype)
            return int(arr.argmax()), f"{np.dtype(dtype).name}_vector"

    raise RuntimeError(f"Unsupported MLPerf accuracy payload length: {len(raw)}")


def evaluate_predictions(rows: List[Tuple[str, int]], items: List[Dict[str, Any]]) -> Dict[str, Any]:
    dedup: Dict[int, Dict[str, Any]] = {}
    for item in items:
        idx = int(item["qsl_idx"])
        dedup.setdefault(idx, item)

    decoded: Dict[int, Tuple[int, str]] = {}
    for idx, item in dedup.items():
        decoded[idx] = decode_prediction(str(item["data"]))

    scored: Dict[int, Dict[str, int]] = {0: {"good": 0}, 1: {"good": 0}}
    prediction_format = None
    total = 0
    for idx, (_, label) in enumerate(rows):
        if idx not in decoded:
            continue
        pred, fmt = decoded[idx]
        prediction_format = prediction_format or fmt
        for shift in (0, 1):
            if pred == label - shift:
                scored[shift]["good"] += 1
        total += 1

    if total == 0:
        raise RuntimeError("No MLPerf predictions matched evaluation rows")

    label_shift = 1 if scored[1]["good"] >= scored[0]["good"] else 0
    good = scored[label_shift]["good"]
    return {
        "prediction_format": prediction_format,
        "label_shift": label_shift,
        "good_top1": good,
        "total": total,
        "top1_accuracy": (good / total) * 100.0,
    }


def main() -> None:
    args = parse_args()
    if not args.program_path.exists():
        raise FileNotFoundError(f"TPU program not found: {args.program_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    val_map = args.dataset_dir / "val_map.txt"
    rows = load_val_rows(val_map)
    if args.samples > 0:
        rows = rows[: args.samples]

    cmd = [
        args.mlperf_binary,
        "-s",
        "offline",
        "-m",
        "accuracy",
        "-o",
        "0",
        "-p",
        args.program_path.as_posix(),
        "-t",
        "resnet50",
        "-d",
        args.dataset_dir.as_posix(),
        "-n",
        str(len(rows)),
    ]
    process = subprocess.run(cmd, cwd=args.output_dir, capture_output=True, text=True)
    (args.output_dir / "mlperf_stdout.txt").write_text(process.stdout, encoding="utf-8")
    (args.output_dir / "mlperf_stderr.txt").write_text(process.stderr, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(
            f"MLPerf accuracy failed with exit code {process.returncode}. "
            f"See {args.output_dir / 'mlperf_stderr.txt'}"
        )

    accuracy_log = args.output_dir / "mlperf_log_accuracy.json"
    if not accuracy_log.exists():
        raise FileNotFoundError(f"MLPerf accuracy log not found: {accuracy_log}")

    items = json.loads(accuracy_log.read_text(encoding="utf-8"))
    metrics = evaluate_predictions(rows, items)
    summary = {
        "pipeline": "resnet50_mlperf_vendor_accuracy",
        "program_path": args.program_path.as_posix(),
        "dataset_dir": args.dataset_dir.as_posix(),
        "val_map": val_map.as_posix(),
        "requested_samples": args.samples,
        "effective_samples": len(rows),
        "mlperf_accuracy_log": accuracy_log.as_posix(),
        "mlperf_command": cmd,
        **metrics,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved vendor MLPerf accuracy summary: {summary_path}")


if __name__ == "__main__":
    main()
