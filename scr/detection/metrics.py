import argparse
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

REPO_ROOT = Path(__file__).resolve().parents[2]
METRIC_KEYS = [
    "AP@[.50:.95]", "AP@0.50", "AP@0.75", "AP_small", "AP_medium", "AP_large",
    "AR@1", "AR@10", "AR@100", "AR_small", "AR_medium", "AR_large",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute COCO bbox metrics from predictions JSON")
    parser.add_argument("--ann-file", type=Path, default=REPO_ROOT / "data/evaluation/MSCOCO2017/annotations/instances_val2017.json")
    parser.add_argument("--predictions", type=Path, default=REPO_ROOT / "experiments/detection/predictions.json")
    parser.add_argument("--output-json", type=Path, default=REPO_ROOT / "experiments/detection/metrics.json")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    coco = COCO(args.ann_file.as_posix())
    dt = coco.loadRes(args.predictions.as_posix())
    ev = COCOeval(coco, dt, "bbox")
    if args.limit > 0:
        ev.params.imgIds = coco.getImgIds()[: args.limit]
    ev.evaluate(); ev.accumulate(); ev.summarize()

    vals = [float(x) for x in ev.stats.tolist()]
    result = {
        "ann_file": args.ann_file.as_posix(),
        "predictions": args.predictions.as_posix(),
        "num_images": len(ev.params.imgIds) if ev.params.imgIds else len(coco.getImgIds()),
        "metrics": {k: v for k, v in zip(METRIC_KEYS, vals)},
    }
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
