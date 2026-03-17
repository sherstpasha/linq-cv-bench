import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from metrics_utils import VOC_CLASSES, summarize_confusion, update_confusion
from onnx_runtime_utils import (
    build_samples,
    create_session,
    infer_runtime_contract,
    infer_static_batch_size,
    load_optional_export_metadata,
    make_batch,
    pick_output,
    preprocess_image,
    resolve_spatial_size,
    to_logits_shape,
    logits_to_mask,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FCN-ResNet50 ONNX accuracy on VOC using CPU or CUDA")
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "experiments/segmentation/fcn_resnet50.onnx")
    parser.add_argument("--voc-root", type=Path, default=REPO_ROOT / "data/evaluation/VOCdevkit/VOC2012")
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--provider", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--height", type=int, default=520)
    parser.add_argument("--width", type=int, default=520)
    parser.add_argument("--num-classes", type=int, default=21)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=REPO_ROOT / "experiments/segmentation_onnx/accuracy/predictions",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=REPO_ROOT / "experiments/segmentation_onnx/accuracy/summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise RuntimeError("--batch-size must be > 0")

    split_file = args.split_file or (args.voc_root / "ImageSets/Segmentation/val.txt")
    samples = build_samples(args.voc_root, split_file, args.limit)
    args.predictions_dir.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)

    export_metadata = load_optional_export_metadata(args.model_path)
    session, resolved_provider = create_session(args.model_path, args.provider)
    runtime_contract = infer_runtime_contract(session, export_metadata)
    effective_batch_size = infer_static_batch_size(runtime_contract, args.batch_size)
    height, width = resolve_spatial_size(runtime_contract, args.height, args.width)
    input_name = runtime_contract["input_name"]

    conf = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    infer_time = 0.0
    measured_images = 0
    batch_count = (len(samples) + effective_batch_size - 1) // effective_batch_size
    resized_predictions = 0

    for batch_idx in range(batch_count):
        start = batch_idx * effective_batch_size
        end = min((batch_idx + 1) * effective_batch_size, len(samples))
        batch_samples = samples[start:end]

        tensors: List[np.ndarray] = []
        original_sizes: List[Tuple[int, int]] = []
        for _, image_path, _ in batch_samples:
            with Image.open(image_path) as image:
                original_sizes.append((image.height, image.width))
            tensors.append(preprocess_image(image_path, runtime_contract, height=height, width=width))
        x_batch = make_batch(tensors, effective_batch_size)

        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: x_batch})
        t1 = time.perf_counter()

        output_dict: Dict[str, np.ndarray] = {
            output.name: value for output, value in zip(session.get_outputs(), outputs)
        }
        logits = pick_output(session, output_dict, runtime_contract.get("output_name"))
        logits = to_logits_shape(logits, expected_batch=len(batch_samples))[: len(batch_samples)]

        for (image_id, _, gt_path), sample_logits, orig_hw in zip(batch_samples, logits, original_sizes):
            pred = logits_to_mask(sample_logits, num_classes=args.num_classes)
            pred_img = Image.fromarray(pred)
            if pred.shape != orig_hw:
                pred_img = pred_img.resize((orig_hw[1], orig_hw[0]), resample=Image.NEAREST)
                resized_predictions += 1
            pred_arr = np.asarray(pred_img, dtype=np.uint8)
            pred_img.save(args.predictions_dir / f"{image_id}.png")

            gt = np.asarray(Image.open(gt_path), dtype=np.uint8)
            update_confusion(conf, gt, pred_arr, args.num_classes, args.ignore_index)

        if batch_idx >= args.warmup_batches:
            infer_time += t1 - t0
            measured_images += len(batch_samples)

    measured_batches = max(0, batch_count - args.warmup_batches)
    avg_batch_latency_ms = (
        (infer_time / measured_batches) * 1000.0 if measured_batches > 0 else 0.0
    )
    throughput = measured_images / infer_time if infer_time > 0 else 0.0
    summary = {
        "pipeline": "onnx_accuracy",
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
        "requested_limit": args.limit,
        "images_in_split": len(samples),
        "warmup_batches": args.warmup_batches,
        "measured_inference_sec": infer_time,
        "throughput_img_per_sec": throughput,
        "avg_batch_latency_ms": avg_batch_latency_ms,
        "predictions_dir": args.predictions_dir.as_posix(),
        "resized_predictions": resized_predictions,
        **summarize_confusion(conf, VOC_CLASSES[: args.num_classes]),
    }
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved ONNX accuracy summary: {args.summary_out}")


if __name__ == "__main__":
    main()
