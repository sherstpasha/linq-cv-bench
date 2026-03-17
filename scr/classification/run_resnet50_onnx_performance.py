import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from onnx_runtime_utils import (
    create_session,
    infer_runtime_contract,
    load_optional_export_metadata,
    load_val_rows,
    make_batch,
    preprocess_image,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ResNet-50 ONNX performance on CPU or CUDA")
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "models/classification/resnet50.onnx")
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "data/evaluation/imagenet")
    parser.add_argument("--val-map", type=Path, default=None)
    parser.add_argument("--provider", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=0, help="0 uses methodology-like defaults")
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=REPO_ROOT / "experiments/classification_onnx/performance/b8/summary.json",
    )
    return parser.parse_args()


def resolve_samples(batch_size: int, samples: int) -> int:
    if samples > 0:
        return samples
    if batch_size == 1:
        return 500
    if batch_size == 8:
        return 1000
    return max(batch_size * 32, batch_size)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise RuntimeError("--batch-size must be > 0")
    val_map = args.val_map or (args.dataset_dir / "val_map.txt")
    rows = load_val_rows(val_map)
    effective_samples = min(len(rows), resolve_samples(args.batch_size, args.samples))
    rows = rows[:effective_samples]
    if not rows:
        raise RuntimeError("No evaluation rows selected")

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    export_metadata = load_optional_export_metadata(args.model_path)
    session, resolved_provider = create_session(args.model_path, args.provider)
    runtime_contract = infer_runtime_contract(session, export_metadata)
    input_name = runtime_contract["input_name"]

    batch_count = (len(rows) + args.batch_size - 1) // args.batch_size
    infer_time = 0.0
    measured_images = 0

    for batch_idx in range(batch_count):
        start = batch_idx * args.batch_size
        end = min((batch_idx + 1) * args.batch_size, len(rows))
        batch_rows = rows[start:end]
        batch_tensors = [preprocess_image(args.dataset_dir / image_name, runtime_contract) for image_name, _ in batch_rows]
        x_batch = make_batch(batch_tensors, args.batch_size)

        t0 = time.perf_counter()
        session.run(None, {input_name: x_batch})
        t1 = time.perf_counter()

        if batch_idx >= args.warmup_batches:
            infer_time += t1 - t0
            measured_images += len(batch_rows)

    measured_batches = max(0, batch_count - args.warmup_batches)
    avg_batch_latency_ms = (
        (infer_time / measured_batches) * 1000.0 if measured_batches > 0 else 0.0
    )
    throughput = measured_images / infer_time if infer_time > 0 else 0.0
    summary = {
        "pipeline": "onnx_performance",
        "model_path": args.model_path.as_posix(),
        "dataset_dir": args.dataset_dir.as_posix(),
        "val_map": val_map.as_posix(),
        "provider_requested": args.provider,
        "provider_resolved": resolved_provider,
        "available_providers": session.get_providers(),
        "runtime_contract": runtime_contract,
        "batch_size": args.batch_size,
        "requested_samples": args.samples,
        "effective_samples": len(rows),
        "warmup_batches": args.warmup_batches,
        "measured_inference_sec": infer_time,
        "throughput_img_per_sec": throughput,
        "avg_batch_latency_ms": avg_batch_latency_ms,
    }
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved ONNX performance summary: {args.summary_out}")


if __name__ == "__main__":
    main()
