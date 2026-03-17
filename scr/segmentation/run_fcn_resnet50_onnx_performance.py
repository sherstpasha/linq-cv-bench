import argparse
import json
import time
from pathlib import Path

from onnx_runtime_utils import (
    build_samples,
    create_session,
    infer_runtime_contract,
    infer_static_batch_size,
    load_optional_export_metadata,
    make_batch,
    preprocess_image,
    resolve_spatial_size,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FCN-ResNet50 ONNX performance on CPU or CUDA")
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "experiments/segmentation/fcn_resnet50.onnx")
    parser.add_argument("--voc-root", type=Path, default=REPO_ROOT / "data/evaluation/VOCdevkit/VOC2012")
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--provider", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=0, help="0 uses methodology-like defaults")
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--height", type=int, default=520)
    parser.add_argument("--width", type=int, default=520)
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=REPO_ROOT / "experiments/segmentation_onnx/performance/b8/summary.json",
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

    split_file = args.split_file or (args.voc_root / "ImageSets/Segmentation/val.txt")
    samples = build_samples(args.voc_root, split_file, 0)
    effective_samples = min(len(samples), resolve_samples(args.batch_size, args.samples))
    samples = samples[:effective_samples]
    if not samples:
        raise RuntimeError("No VOC samples selected")

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    export_metadata = load_optional_export_metadata(args.model_path)
    session, resolved_provider = create_session(args.model_path, args.provider)
    runtime_contract = infer_runtime_contract(session, export_metadata)
    effective_batch_size = infer_static_batch_size(runtime_contract, args.batch_size)
    height, width = resolve_spatial_size(runtime_contract, args.height, args.width)
    input_name = runtime_contract["input_name"]

    batch_count = (len(samples) + effective_batch_size - 1) // effective_batch_size
    infer_time = 0.0
    measured_images = 0

    for batch_idx in range(batch_count):
        start = batch_idx * effective_batch_size
        end = min((batch_idx + 1) * effective_batch_size, len(samples))
        batch_samples = samples[start:end]
        batch_tensors = [
            preprocess_image(image_path, runtime_contract, height=height, width=width)
            for _, image_path, _ in batch_samples
        ]
        x_batch = make_batch(batch_tensors, effective_batch_size)

        t0 = time.perf_counter()
        session.run(None, {input_name: x_batch})
        t1 = time.perf_counter()

        if batch_idx >= args.warmup_batches:
            infer_time += t1 - t0
            measured_images += len(batch_samples)

    measured_batches = max(0, batch_count - args.warmup_batches)
    avg_batch_latency_ms = (
        (infer_time / measured_batches) * 1000.0 if measured_batches > 0 else 0.0
    )
    throughput = measured_images / infer_time if infer_time > 0 else 0.0
    summary = {
        "pipeline": "onnx_performance",
        "model_path": args.model_path.as_posix(),
        "voc_root": args.voc_root.as_posix(),
        "split_file": split_file.as_posix(),
        "provider_requested": args.provider,
        "provider_resolved": resolved_provider,
        "available_providers": session.get_providers(),
        "runtime_contract": runtime_contract,
        "input_size": [height, width],
        "batch_size": effective_batch_size,
        "requested_batch_size": args.batch_size,
        "requested_samples": args.samples,
        "effective_samples": len(samples),
        "warmup_batches": args.warmup_batches,
        "measured_inference_sec": infer_time,
        "throughput_img_per_sec": throughput,
        "avg_batch_latency_ms": avg_batch_latency_ms,
    }
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved ONNX performance summary: {args.summary_out}")


if __name__ == "__main__":
    main()
