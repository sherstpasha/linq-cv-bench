import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from onnx_runtime_utils import (
    create_session,
    infer_static_batch_size,
    infer_runtime_contract,
    load_optional_export_metadata,
    load_val_rows,
    make_batch,
    pick_logits,
    preprocess_image,
    top5_indices,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ResNet-50 ONNX accuracy on CPU or CUDA")
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "models/classification/resnet50.onnx")
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "data/evaluation/imagenet")
    parser.add_argument("--val-map", type=Path, default=None)
    parser.add_argument("--provider", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=0, help="0 means all rows from val_map.txt")
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=REPO_ROOT / "experiments/classification_onnx/accuracy/predictions.jsonl",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=REPO_ROOT / "experiments/classification_onnx/accuracy/summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    val_map = args.val_map or (args.dataset_dir / "val_map.txt")
    rows = load_val_rows(val_map)
    if args.samples > 0:
        rows = rows[: args.samples]
    if not rows:
        raise RuntimeError("No evaluation rows selected")
    if args.batch_size <= 0:
        raise RuntimeError("--batch-size must be > 0")

    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)

    export_metadata = load_optional_export_metadata(args.model_path)
    session, resolved_provider = create_session(args.model_path, args.provider)
    runtime_contract = infer_runtime_contract(session, export_metadata)
    effective_batch_size = infer_static_batch_size(runtime_contract, args.batch_size)

    input_name = runtime_contract["input_name"]
    predictions: List[Dict[str, object]] = []
    good_top1 = 0
    good_top5 = 0
    infer_time = 0.0
    measured_images = 0
    batch_count = (len(rows) + effective_batch_size - 1) // effective_batch_size

    for batch_idx in range(batch_count):
        start = batch_idx * effective_batch_size
        end = min((batch_idx + 1) * effective_batch_size, len(rows))
        batch_rows = rows[start:end]

        batch_tensors = [preprocess_image(args.dataset_dir / image_name, runtime_contract) for image_name, _ in batch_rows]
        x_batch = make_batch(batch_tensors, effective_batch_size)

        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: x_batch})
        t1 = time.perf_counter()

        output_dict = {output.name: value for output, value in zip(session.get_outputs(), outputs)}
        logits = pick_logits(session, output_dict, runtime_contract.get("output_name"))
        logits = np.asarray(logits, dtype=np.float32)[: len(batch_rows)]
        top1 = logits.argmax(axis=1)
        top5 = top5_indices(logits)

        for (image_name, label), pred1, pred5 in zip(batch_rows, top1.tolist(), top5.tolist()):
            label0 = label - 1
            is_top1 = pred1 == label0
            is_top5 = label0 in pred5
            good_top1 += int(is_top1)
            good_top5 += int(is_top5)
            predictions.append(
                {
                    "image": image_name,
                    "label": label0,
                    "top1": int(pred1),
                    "top5": [int(x) for x in pred5],
                    "is_top1": bool(is_top1),
                    "is_top5": bool(is_top5),
                }
            )

        if batch_idx >= args.warmup_batches:
            infer_time += t1 - t0
            measured_images += len(batch_rows)

    with args.predictions_out.open("w", encoding="utf-8") as file:
        for row in predictions:
            file.write(json.dumps(row, ensure_ascii=True) + "\n")

    total = len(rows)
    measured_batches = max(0, batch_count - args.warmup_batches)
    avg_batch_latency_ms = (
        (infer_time / measured_batches) * 1000.0 if measured_batches > 0 else 0.0
    )
    throughput = measured_images / infer_time if infer_time > 0 else 0.0
    summary = {
        "pipeline": "onnx_accuracy",
        "model_path": args.model_path.as_posix(),
        "dataset_dir": args.dataset_dir.as_posix(),
        "val_map": val_map.as_posix(),
        "provider_requested": args.provider,
        "provider_resolved": resolved_provider,
        "available_providers": session.get_providers(),
        "runtime_contract": runtime_contract,
        "batch_size": effective_batch_size,
        "requested_batch_size": args.batch_size,
        "requested_samples": args.samples,
        "effective_samples": total,
        "warmup_batches": args.warmup_batches,
        "measured_inference_sec": infer_time,
        "throughput_img_per_sec": throughput,
        "avg_batch_latency_ms": avg_batch_latency_ms,
        "predictions_file": args.predictions_out.as_posix(),
        "top1_accuracy": (good_top1 / total) * 100.0,
        "top5_accuracy": (good_top5 / total) * 100.0,
        "good_top1": good_top1,
        "good_top5": good_top5,
        "total": total,
    }
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved ONNX accuracy summary: {args.summary_out}")


if __name__ == "__main__":
    main()
