import argparse
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


REPO_ROOT = Path(__file__).resolve().parents[2]
METRIC_KEYS = [
    "AP@[.50:.95]",
    "AP@0.50",
    "AP@0.75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR@1",
    "AR@10",
    "AR@100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute COCO bbox metrics from predictions JSON")
    parser.add_argument(
        "--ann-file",
        type=Path,
        default=REPO_ROOT / "data/evaluation/MSCOCO2017/annotations/instances_val2017.json",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=REPO_ROOT / "experiments/detection_onnx_reference/predictions.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "experiments/detection_onnx_reference/metrics.json",
    )
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    coco = COCO(args.ann_file.as_posix())
    try:
        pred_rows = json.loads(args.predictions.read_text(encoding="utf-8"))
    except Exception as error:
        raise RuntimeError(f"Failed to read predictions JSON: {args.predictions}") from error

    if not isinstance(pred_rows, list):
        raise RuntimeError(f"Predictions file must be a JSON list: {args.predictions}")

    if len(pred_rows) == 0:
        result = {
            "ann_file": args.ann_file.as_posix(),
            "predictions": args.predictions.as_posix(),
            "num_images": args.limit if args.limit > 0 else len(coco.getImgIds()),
            "metrics": {key: 0.0 for key in METRIC_KEYS},
            "note": "Predictions are empty; metrics were set to 0.0.",
        }
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return

    dt = coco.loadRes(args.predictions.as_posix())
    evaluator = COCOeval(coco, dt, "bbox")
    if args.limit > 0:
        evaluator.params.imgIds = coco.getImgIds()[: args.limit]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    values = [float(value) for value in evaluator.stats.tolist()]
    result = {
        "ann_file": args.ann_file.as_posix(),
        "predictions": args.predictions.as_posix(),
        "num_images": len(evaluator.params.imgIds) if evaluator.params.imgIds else len(coco.getImgIds()),
        "metrics": {key: value for key, value in zip(METRIC_KEYS, values)},
    }
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
