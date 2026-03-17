import argparse
import json
import time
from pathlib import Path

from pycocotools.coco import COCO

from onnx_runtime_utils import (
    create_session,
    infer_static_batch_size,
    infer_tiny_yolo3_contract,
    make_batch,
    preprocess_image,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tiny YOLOv3 ONNX performance on COCO using CPU or CUDA")
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "models/detection/tiny-yolov3-11.onnx")
    parser.add_argument("--img-dir", type=Path, default=REPO_ROOT / "data/evaluation/MSCOCO2017/val2017")
    parser.add_argument(
        "--ann-file",
        type=Path,
        default=REPO_ROOT / "data/evaluation/MSCOCO2017/annotations/instances_val2017.json",
    )
    parser.add_argument("--provider", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--samples", type=int, default=500, help="0 uses 500 by default")
    parser.add_argument("--warmup-images", type=int, default=10)
    parser.add_argument("--box-order", choices=["yxyx", "xyxy"], default="yxyx")
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=REPO_ROOT / "experiments/detection/tiny_yolo3_onnx/performance/b1/summary.json",
    )
    return parser.parse_args()


def resolve_samples(total_images: int, requested_samples: int) -> int:
    if requested_samples > 0:
        return min(total_images, requested_samples)
    return min(total_images, 500)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise RuntimeError("--batch-size must be > 0")

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    coco = COCO(args.ann_file.as_posix())
    img_ids = coco.getImgIds()
    img_ids = img_ids[: resolve_samples(len(img_ids), args.samples)]
    if not img_ids:
        raise RuntimeError("No COCO images selected")

    session, resolved_provider = create_session(args.model_path, args.provider)
    runtime_contract = infer_tiny_yolo3_contract(session, args.box_order)
    effective_batch_size = infer_static_batch_size(runtime_contract, args.batch_size)
    warmup_batches = max(0, (args.warmup_images + effective_batch_size - 1) // effective_batch_size)

    image_input_name = runtime_contract["image_input_name"]
    image_shape_input_name = runtime_contract["image_shape_input_name"]
    batch_count = (len(img_ids) + effective_batch_size - 1) // effective_batch_size
    infer_time = 0.0
    measured_images = 0

    for batch_idx in range(batch_count):
        start = batch_idx * effective_batch_size
        end = min((batch_idx + 1) * effective_batch_size, len(img_ids))
        batch_ids = img_ids[start:end]

        image_tensors = []
        image_shapes = []
        for image_id in batch_ids:
            info = coco.loadImgs(image_id)[0]
            image_tensor, image_shape, _ = preprocess_image(args.img_dir / info["file_name"], runtime_contract)
            image_tensors.append(image_tensor)
            image_shapes.append(image_shape)

        x_batch = make_batch(image_tensors, effective_batch_size)
        shape_batch = make_batch(image_shapes, effective_batch_size)

        t0 = time.perf_counter()
        session.run(None, {image_input_name: x_batch, image_shape_input_name: shape_batch})
        t1 = time.perf_counter()

        if batch_idx >= warmup_batches:
            infer_time += t1 - t0
            measured_images += len(batch_ids)

    measured_batches = max(0, batch_count - warmup_batches)
    avg_batch_latency_ms = (infer_time / measured_batches) * 1000.0 if measured_batches > 0 else 0.0
    throughput = measured_images / infer_time if infer_time > 0 else 0.0
    summary = {
        "pipeline": "tiny_yolo3_onnx_performance",
        "model_path": args.model_path.as_posix(),
        "img_dir": args.img_dir.as_posix(),
        "ann_file": args.ann_file.as_posix(),
        "provider_requested": args.provider,
        "provider_resolved": resolved_provider,
        "available_providers": session.get_providers(),
        "runtime_contract": runtime_contract,
        "batch_size": effective_batch_size,
        "requested_batch_size": args.batch_size,
        "requested_samples": args.samples,
        "effective_samples": len(img_ids),
        "warmup_images": args.warmup_images,
        "warmup_batches": warmup_batches,
        "measured_inference_sec": infer_time,
        "throughput_img_per_sec": throughput,
        "avg_batch_latency_ms": avg_batch_latency_ms,
    }
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved tiny_yolo3 ONNX performance summary: {args.summary_out}")


if __name__ == "__main__":
    main()
