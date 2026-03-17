import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pycocotools.coco import COCO

from onnx_runtime_utils import (
    compute_coco_metrics,
    create_input_feed,
    create_session,
    decode_tiny_yolo3_nms_outputs,
    infer_static_batch_size,
    infer_tiny_yolo3_contract,
    load_optional_export_metadata,
    make_batch,
    pick_output_map,
    preprocess_image,
)
from run_tiny_yolo3_accuracy import decode_predictions, parse_anchors, parse_masks


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANCHORS = "10,14 23,27 37,58 81,82 135,169 344,319"
DEFAULT_MASKS = "3,4,5|0,1,2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tiny YOLOv3 ONNX accuracy on COCO using CPU or CUDA")
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "models/detection/tiny-yolov3-11.onnx")
    parser.add_argument("--img-dir", type=Path, default=REPO_ROOT / "data/evaluation/MSCOCO2017/val2017")
    parser.add_argument(
        "--ann-file",
        type=Path,
        default=REPO_ROOT / "data/evaluation/MSCOCO2017/annotations/instances_val2017.json",
    )
    parser.add_argument("--provider", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=5000, help="0 means full COCO eval split")
    parser.add_argument("--warmup-images", type=int, default=10)
    parser.add_argument("--score-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--box-order", choices=["yxyx", "xyxy"], default="yxyx")
    parser.add_argument("--anchors", type=str, default=DEFAULT_ANCHORS)
    parser.add_argument("--masks", type=str, default=DEFAULT_MASKS)
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=REPO_ROOT / "experiments/detection/tiny_yolo3_onnx/accuracy/predictions.json",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=REPO_ROOT / "experiments/detection/tiny_yolo3_onnx/accuracy/summary.json",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=REPO_ROOT / "experiments/detection/tiny_yolo3_onnx/accuracy/metrics.json",
    )
    parser.add_argument(
        "--metrics-text",
        type=Path,
        default=REPO_ROOT / "experiments/detection/tiny_yolo3_onnx/accuracy/metrics.txt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise RuntimeError("--batch-size must be > 0")

    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_text.parent.mkdir(parents=True, exist_ok=True)

    coco = COCO(args.ann_file.as_posix())
    img_ids = coco.getImgIds()
    if args.limit > 0:
        img_ids = img_ids[: args.limit]
    if not img_ids:
        raise RuntimeError("No COCO images selected")

    export_metadata = load_optional_export_metadata(args.model_path)
    session, resolved_provider = create_session(args.model_path, args.provider)
    runtime_contract = infer_tiny_yolo3_contract(session, export_metadata, args.box_order)
    effective_batch_size = infer_static_batch_size(runtime_contract, args.batch_size)
    warmup_batches = max(0, (args.warmup_images + effective_batch_size - 1) // effective_batch_size)

    anchors = parse_anchors(str(runtime_contract.get("anchors") or args.anchors))
    masks = parse_masks(str(runtime_contract.get("masks") or args.masks))
    predictions: List[Dict[str, Any]] = []
    infer_time = 0.0
    measured_images = 0
    decode_probe: Optional[Dict[str, Any]] = None

    batch_count = (len(img_ids) + effective_batch_size - 1) // effective_batch_size
    for batch_idx in range(batch_count):
        start = batch_idx * effective_batch_size
        end = min((batch_idx + 1) * effective_batch_size, len(img_ids))
        batch_ids = img_ids[start:end]

        image_tensors = []
        image_shapes = []
        metas = []
        for image_id in batch_ids:
            info = coco.loadImgs(image_id)[0]
            image_tensor, image_shape, meta = preprocess_image(args.img_dir / info["file_name"], runtime_contract)
            meta["image_id"] = int(image_id)
            image_tensors.append(image_tensor)
            if image_shape is not None:
                image_shapes.append(image_shape)
            metas.append(meta)

        x_batch = make_batch(image_tensors, effective_batch_size)
        shape_batch = make_batch(image_shapes, effective_batch_size) if image_shapes else None
        t0 = time.perf_counter()
        outputs = session.run(None, create_input_feed(runtime_contract, x_batch, shape_batch))
        t1 = time.perf_counter()
        output_map = pick_output_map(session, outputs)

        if runtime_contract["mode"] == "modelzoo_nms":
            batch_predictions, batch_probe = decode_tiny_yolo3_nms_outputs(
                output_map,
                runtime_contract,
                metas,
                score_threshold=args.score_thres,
                max_det=args.max_det,
            )
        elif runtime_contract["mode"] == "yolo_heads":
            batch_predictions, batch_probe = decode_predictions(
                output_map,
                runtime_contract.get("preferred_output_name"),
                metas,
                int(runtime_contract["image_size"]),
                anchors,
                masks,
                int(runtime_contract.get("num_classes") or 80),
                args.score_thres,
                args.iou_thres,
                args.max_det,
            )
        else:
            raise RuntimeError(f"Unsupported tiny_yolo3 ONNX mode: {runtime_contract['mode']}")

        predictions.extend(batch_predictions)
        if decode_probe is None:
            decode_probe = batch_probe

        if batch_idx >= warmup_batches:
            infer_time += t1 - t0
            measured_images += len(batch_ids)

    args.predictions_out.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    metrics = compute_coco_metrics(args.ann_file, args.predictions_out, args.metrics_out, args.metrics_text, args.limit)

    measured_batches = max(0, batch_count - warmup_batches)
    avg_batch_latency_ms = (infer_time / measured_batches) * 1000.0 if measured_batches > 0 else 0.0
    throughput = measured_images / infer_time if infer_time > 0 else 0.0
    summary = {
        "pipeline": "tiny_yolo3_onnx_accuracy",
        "model_path": args.model_path.as_posix(),
        "img_dir": args.img_dir.as_posix(),
        "ann_file": args.ann_file.as_posix(),
        "provider_requested": args.provider,
        "provider_resolved": resolved_provider,
        "available_providers": session.get_providers(),
        "runtime_contract": runtime_contract,
        "batch_size": effective_batch_size,
        "requested_batch_size": args.batch_size,
        "effective_images": len(img_ids),
        "warmup_images": args.warmup_images,
        "warmup_batches": warmup_batches,
        "score_threshold": args.score_thres,
        "iou_threshold": args.iou_thres,
        "max_det": args.max_det,
        "measured_inference_sec": infer_time,
        "throughput_img_per_sec": throughput,
        "avg_batch_latency_ms": avg_batch_latency_ms,
        "detections": len(predictions),
        "predictions_file": args.predictions_out.as_posix(),
        "metrics_json": args.metrics_out.as_posix(),
        "metrics_text": args.metrics_text.as_posix(),
        "metrics": metrics.get("metrics", {}),
        "decode_probe": decode_probe,
    }
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved tiny_yolo3 ONNX accuracy summary: {args.summary_out}")
    print(f"Saved tiny_yolo3 ONNX metrics: {args.metrics_out}")


if __name__ == "__main__":
    main()
